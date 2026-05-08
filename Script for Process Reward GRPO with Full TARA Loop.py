"""
TB-ViTAR
First and Second Improvements F: PEFT LoRA + Process-Reward GRPO + Full TARA loop
"""

import modal, os, json, random

# ── Volume & paths ─────────────────────────────────────────────────────────────
vol = modal.Volume.from_name("my-dataset", create_if_missing=True)

DATASET_MOUNT_PATH = "/mnt/my-dataset"
OUTPUTS_DIR        = "/mnt/my-dataset/outputs"
SFT_CKPT_DIR       = "/mnt/my-dataset/outputs/qwen2vl_sft_full"
OUTCOME_CKPT_DIR   = "/mnt/my-dataset/outputs/qwen2vl_outcome_grpo"
PROCESS_CKPT_DIR   = "/mnt/my-dataset/outputs/qwen2vl_process_grpo"
VQA_JSON_PATH      = "/mnt/my-dataset/outputs/tbx11k_VQA_balanced.json"

# ── Image ────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "transformers==4.48.3",
        "accelerate>=0.34.2",
        "bitsandbytes>=0.43.0",
        "peft>=0.14.0",
        "trl==0.15.2",
        "Pillow",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "tqdm",
        "qwen-vl-utils",
        "datasets",
    )
)

app = modal.App("tbvitar-nb3", image=image)

# ── Shared utilities ───────────────────────────────────────
SHARED = """
import os, re, json, random, copy, warnings, zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np, pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_MOUNT_PATH  = "/mnt/my-dataset"
DATASET_EXTRACT_DIR = "/mnt/my-dataset/TBX11K_extracted"
OUTPUTS_DIR         = "/mnt/my-dataset/outputs"
SFT_CKPT_DIR        = "/mnt/my-dataset/outputs/qwen2vl_sft_full"
OUTCOME_CKPT_DIR    = "/mnt/my-dataset/outputs/qwen2vl_outcome_grpo"
PROCESS_CKPT_DIR    = "/mnt/my-dataset/outputs/qwen2vl_process_grpo"
VQA_JSON_PATH       = "/mnt/my-dataset/outputs/tbx11k_VQA_balanced.json"
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")
PREFERRED_DTYPE = (torch.bfloat16 if torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 8 else torch.float16)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(PROCESS_CKPT_DIR, exist_ok=True)

def find_tbx11k_root(base):
    for root, dirs, _ in os.walk(base):
        rp = Path(root)
        if (rp/"imgs").exists() and (rp/"annotations").exists(): return rp
    for root, dirs, _ in os.walk(base):
        if "imgs" in dirs: return Path(root)
    raise FileNotFoundError(f"TBX11K not found under {base}")

def setup_dataset():
    os.makedirs(DATASET_EXTRACT_DIR, exist_ok=True)
    if not any(Path(DATASET_EXTRACT_DIR).rglob("*.jpg")):
        for z in list(Path(DATASET_MOUNT_PATH).glob("*.zip"))+list(Path(DATASET_MOUNT_PATH).glob("**/*.zip")):
            print(f"Extracting {z} ...")
            with zipfile.ZipFile(z,"r") as zf: zf.extractall(DATASET_EXTRACT_DIR)
    tbx_root = find_tbx11k_root(DATASET_EXTRACT_DIR)
    xml_dir  = tbx_root / "annotations" / "xml"
    IMAGE_INDEX = {}
    for ext in ("jpg","jpeg","png"):
        for p in tbx_root.rglob(f"*.{ext}"): IMAGE_INDEX[p.stem]=str(p); IMAGE_INDEX[p.name]=str(p)
    LABEL_MAP = {"active_tb":("Active TB",1),"latent_tb":("Latent TB",1),"tb":("Active TB",1),
                 "sick":("Sick-non-TB",0),"sick_non_tb":("Sick-non-TB",0),"non_tb":("Sick-non-TB",0),
                 "healthy":("Healthy",0)}
    rows = []
    for ext in ("jpg","jpeg","png"):
        for p in tbx_root.rglob(f"*.{ext}"):
            cls4="Healthy"; tb=0
            for part in p.parts:
                for key,(c4,t) in LABEL_MAP.items():
                    if key in part.lower(): cls4=c4; tb=t; break
            rows.append({"path":str(p),"stem":p.stem,"cls4":cls4,"tb":tb})
    df = pd.DataFrame(rows).drop_duplicates(subset="stem").reset_index(drop=True)
    print(f"Images:{len(df)}  TB+:{int(df.tb.sum())} ({df.tb.mean()*100:.1f}%)")
    return df, xml_dir, IMAGE_INDEX

def stratified_split(df, seed=SEED):
    tr,tmp = train_test_split(df.index, test_size=0.30, stratify=df["cls4"], random_state=seed)
    va,te  = train_test_split(tmp, test_size=0.50, stratify=df.loc[tmp,"cls4"], random_state=seed)
    return df.loc[tr].copy(), df.loc[va].copy(), df.loc[te].copy()

BINARY_QS = ["Does this chest X-ray show active tuberculosis?",
             "Is tuberculosis present in this radiograph?",
             "Is there evidence of active pulmonary tuberculosis?",
             "Does this chest X-ray show signs of TB infection?"]
LOC_QS    = ["Localize the TB lesion. Provide [x1,y1,x2,y2] (0-1000).",
             "Where is the TB lesion? Output bounding box [x1,y1,x2,y2].",
             "Identify and localize the TB lesion with [x1,y1,x2,y2]."]

def parse_xml_bbox(xml_path):
    try:
        tree=ET.parse(xml_path); root=tree.getroot()
        for obj in root.findall("object"):
            bb=obj.find("bndbox")
            if bb is None: continue
            xmin=int(float(bb.findtext("xmin","0"))); ymin=int(float(bb.findtext("ymin","0")))
            xmax=int(float(bb.findtext("xmax","0"))); ymax=int(float(bb.findtext("ymax","0")))
            size=root.find("size")
            w=max(1,int(float(size.findtext("width","1000")))) if size else 1000
            h=max(1,int(float(size.findtext("height","1000")))) if size else 1000
            x1=int(xmin/w*1000);y1=int(ymin/h*1000);x2=int(xmax/w*1000);y2=int(ymax/h*1000)
            if x2>x1 and y2>y1: return [x1,y1,x2,y2]
    except: pass
    return None

def bbox_zone(box):
    cx=(box[0]+box[2])/2.; cy=(box[1]+box[3])/2.
    side="right" if cx<480 else ("left" if cx>520 else "hilar")
    zone="upper" if cy<350 else ("mid" if cy<650 else "lower")
    return f"{side} {zone}"

def make_pair(row, xml_dir, q_type="binary"):
    stem=Path(row["path"]).stem; xml_path=Path(xml_dir)/f"{stem}.xml"
    bbox=parse_xml_bbox(xml_path) if xml_path.exists() else None
    is_tb=int(row["tb"])
    if q_type=="binary":
        q=random.choice(BINARY_QS)
        if is_tb:
            if bbox:
                zone=bbox_zone(bbox)
                ans=(f"<think>Suspicious opacity in the {zone} consistent with TB.</think> "
                     f"<act>{bbox}</act> "
                     f"<rethink>The {zone} shows consolidation typical of active tuberculosis.</rethink> "
                     f"<answer>Yes, active tuberculosis in the {zone} at {bbox}.</answer>")
            else:
                ans=("<think>Radiographic opacity consistent with tuberculosis.</think> "
                     "<act>No TB coordinates available.</act> "
                     "<rethink>Opacity pattern consistent with active TB infiltrate.</rethink> "
                     "<answer>Yes, this chest X-ray shows active tuberculosis.</answer>")
        else:
            ans=("<think>No focal opacity, cavitation, or infiltrate suggesting TB.</think> "
                 "<act>No TB lesion.</act> "
                 "<rethink>Lung fields clear, no TB pattern identified.</rethink> "
                 "<answer>No, this chest X-ray does not show active tuberculosis.</answer>")
    elif q_type=="localization" and bbox and is_tb:
        q=random.choice(LOC_QS); zone=bbox_zone(bbox)
        ans=(f"<think>Lesion visible in the {zone}.</think> "
             f"<act>{bbox}</act> "
             f"<rethink>Confirmed TB-pattern opacity in the {zone}.</rethink> "
             f"<answer>Yes, TB lesion at {bbox} in the {zone}.</answer>")
    else: return None
    return {"image":row["path"],"conversations":[{"from":"human","value":q},{"from":"gpt","value":ans}]}

def build_balanced_vqa(df_split, xml_dir, seed=SEED):
    rng=random.Random(seed)
    tb_pos=df_split[df_split["tb"]==1].to_dict("records")
    tb_neg=df_split[df_split["tb"]==0].to_dict("records")
    neg_sample=rng.sample(tb_neg, min(len(tb_pos),len(tb_neg)))
    pairs=[]
    for row in tb_pos:
        p=make_pair(row,xml_dir,"binary");
        if p: pairs.append(p)
        p=make_pair(row,xml_dir,"localization");
        if p: pairs.append(p)
    for row in neg_sample:
        p=make_pair(row,xml_dir,"binary");
        if p: pairs.append(p)
    rng.shuffle(pairs); return pairs

def load_or_build_vqa(df, xml_dir, out_path):
    if os.path.exists(out_path):
        with open(out_path) as f: data=json.load(f)
        yes_n=sum(1 for p in data if "<answer>Yes" in p["conversations"][1]["value"])
        no_n =sum(1 for p in data if "<answer>No"  in p["conversations"][1]["value"])
        print(f"Loaded {len(data)} VQA pairs  YES:{yes_n} NO:{no_n}")
        return data
    train_df,val_df,test_df=stratified_split(df)
    all_pairs=(build_balanced_vqa(train_df,xml_dir)+
               build_balanced_vqa(val_df,xml_dir)+
               build_balanced_vqa(test_df,xml_dir))
    with open(out_path,"w") as f: json.dump(all_pairs,f)
    print(f"Built {len(all_pairs)} VQA pairs")
    return all_pairs

BBOX_RE  = re.compile(r"\\[\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\]")
_YES_PAT = re.compile(r"\\b(yes|positive|active\\s+tb)\\b",re.I)
_NO_PAT  = re.compile(r"\\b(no|negative|no\\s+evidence|clear|normal)\\b",re.I)

def extract_tag(text,tag):
    m=re.search(rf"<{tag}>\\s*(.*?)\\s*</{tag}>",text,re.I|re.S)
    return m.group(1).strip() if m else ""

def parse_yesno(text):
    ans=extract_tag(text,"answer") or text
    first=ans.split(".")[0]
    if _NO_PAT.search(first): return "no"
    if _YES_PAT.search(first): return "yes"
    if _NO_PAT.search(ans): return "no"
    if _YES_PAT.search(ans): return "yes"
    return None

def parse_all_bboxes(text): return [[int(x) for x in m] for m in BBOX_RE.findall(text)]
def parse_bbox(text): boxes=parse_all_bboxes(text); return boxes[0] if boxes else None

def iou(a,b):
    ax1,ay1,ax2,ay2=[float(v) for v in a]; bx1,by1,bx2,by2=[float(v) for v in b]
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0.,ix2-ix1),max(0.,iy2-iy1); inter=iw*ih
    union=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return float(inter/union) if union>0 else 0.0

def best_iou(pred_boxes,gt_box):
    if gt_box is None or not pred_boxes: return 0.0
    return max(iou(pb,gt_box) for pb in pred_boxes)

def decision_gold(pair):
    gt=parse_yesno(pair["conversations"][1]["value"])
    if gt: return gt
    if parse_bbox(pair["conversations"][1]["value"]): return "yes"
    return None

def evaluate_predictions(pred_texts, gold_pairs):
    y_true,y_pred=[],[]; ious,hits50=[],[]; structured=0; examples=[]
    for pred,pair in zip(pred_texts,gold_pairs):
        gt_box=parse_bbox(pair["conversations"][1]["value"])
        gt_dec=decision_gold(pair)
        pred_boxes=parse_all_bboxes(pred); pred_dec=parse_yesno(pred)
        q=pair["conversations"][0]["value"].lower()
        is_dec=any(k in q for k in ("tuberculosis","tb","show","evidence","present"))
        is_loc=any(k in q for k in ("localize","bounding","bbox","where"))
        if all(extract_tag(pred,t) for t in ("think","act","rethink","answer")): structured+=1
        if is_dec and gt_dec:
            if pred_dec is None and pred_boxes: pred_dec="yes"
            y_true.append(1 if gt_dec=="yes" else 0); y_pred.append(1 if pred_dec=="yes" else 0)
        if is_loc and gt_box:
            v=best_iou(pred_boxes,gt_box); ious.append(v); hits50.append(int(v>=0.5))
        if len(examples)<6:
            examples.append({"q":pair["conversations"][0]["value"],
                             "gold":pair["conversations"][1]["value"],"pred":pred})
    res={"n_structured":structured,"structured_rate":structured/max(1,len(gold_pairs))}
    if y_true:
        yt,yp=np.array(y_true),np.array(y_pred)
        tp=int(((yt==1)&(yp==1)).sum()); tn=int(((yt==0)&(yp==0)).sum())
        fn=int(((yt==1)&(yp==0)).sum()); fp=int(((yt==0)&(yp==1)).sum())
        res.update({"n_yesno":len(yt),"n_yes":int((yt==1).sum()),"n_no":int((yt==0).sum()),
                    "yn_acc":float((yt==yp).mean()),
                    "yn_sens":float(tp/(tp+fn)) if (tp+fn) else float("nan"),
                    "yn_spec":float(tn/(tn+fp)) if (tn+fp) else float("nan"),
                    "tp":tp,"tn":tn,"fp":fp,"fn":fn})
    if ious:
        res.update({"n_loc":len(ious),"mean_iou":float(np.mean(ious)),"iou@0.5":float(np.mean(hits50))})
    res["examples"]=examples
    return res
"""


# ── First and Second Improvements F: Process-Reward GRPO with Full TARA loop ──────────────────────────────────
@app.function(
    gpu="A100-80GB",
    timeout=5*3600,
    volumes={DATASET_MOUNT_PATH: vol},
)
def run_first_and_second_improvement_F():
    exec(SHARED, globals())

    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset as HFDataset

    import shutil
    if os.path.exists(PROCESS_CKPT_DIR):
        shutil.rmtree(PROCESS_CKPT_DIR)
        os.makedirs(PROCESS_CKPT_DIR, exist_ok=True)
        print("Cleared old Process-GRPO checkpoint — retraining with improved config.")

    # ── Dataset ────────────────────────────────────────────────────────────
    df,xml_dir,IMAGE_INDEX = setup_dataset()
    train_df,val_df,test_df = stratified_split(df)
    all_vqa = load_or_build_vqa(df, xml_dir, VQA_JSON_PATH)
    train_paths = set(train_df["path"]); test_paths = set(test_df["path"])
    vqa_train = [p for p in all_vqa if p["image"] in train_paths]
    vqa_test  = [p for p in all_vqa if p["image"] in test_paths]

    # Strictly balanced eval: exactly 80 YES binary + 80 NO binary + loc on top
    random.seed(SEED)
    bin_yes_pool = [p for p in vqa_test
                    if decision_gold(p)=="yes"
                    and "localize" not in p["conversations"][0]["value"].lower()
                    and "bounding" not in p["conversations"][0]["value"].lower()]
    bin_no_pool  = [p for p in vqa_test
                    if decision_gold(p)=="no"
                    and "localize" not in p["conversations"][0]["value"].lower()
                    and "bounding" not in p["conversations"][0]["value"].lower()]
    loc_pool     = [p for p in vqa_test
                    if parse_bbox(p["conversations"][1]["value"]) is not None]
    sel_yes = random.sample(bin_yes_pool, min(80, len(bin_yes_pool)))
    sel_no  = random.sample(bin_no_pool,  min(80, len(bin_no_pool)))
    sel_loc = random.sample(loc_pool,     min(40, len(loc_pool)))
    seen=set(); eval_subset=[]
    for p in sel_yes+sel_no:
        k=(p["image"],p["conversations"][0]["value"])
        if k not in seen: seen.add(k); eval_subset.append(p)
    for p in sel_loc:
        k=(p["image"],p["conversations"][0]["value"])
        if k not in seen: seen.add(k); eval_subset.append(p)
    random.shuffle(eval_subset)
    n_eval_yes = sum(1 for p in eval_subset if decision_gold(p)=="yes"
                     and "localize" not in p["conversations"][0]["value"].lower())
    n_eval_no  = sum(1 for p in eval_subset if decision_gold(p)=="no")
    print(f"Eval subset:{len(eval_subset)}  binary_yes:{n_eval_yes} binary_no:{n_eval_no} loc:{len(sel_loc)}")

    if os.path.exists(os.path.join(OUTCOME_CKPT_DIR, "config.json")):
        load_from = OUTCOME_CKPT_DIR
        print("Loading from Outcome-GRPO checkpoint (incremental)")
    elif os.path.exists(os.path.join(SFT_CKPT_DIR, "config.json")):
        load_from = SFT_CKPT_DIR
        print("Loading from SFT checkpoint")
    else:
        load_from = MODEL_ID
        print("Loading base model")

    processor = AutoProcessor.from_pretrained(load_from)

    _orig_cls_fwd = Qwen2VLForConditionalGeneration.forward
    def _patched_cls_fwd(self, *args, **kwargs):
        kwargs.pop("logits_to_keep", None)
        return _orig_cls_fwd(self, *args, **kwargs)
    Qwen2VLForConditionalGeneration.forward = _patched_cls_fwd

    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        load_from, torch_dtype=PREFERRED_DTYPE)
    base_model = base_model.to(DEVICE)
    base_model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"])
    model = get_peft_model(base_model, lora_cfg)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_tot   = sum(p.numel() for p in model.parameters())
    print(f"LoRA trainable:{n_train/1e6:.1f}M / {n_tot/1e6:.1f}M total")

    SYSTEM_TARA = (
        "You are a tuberculosis radiology assistant. Always output exactly four XML tags: "
        "<think>global chest assessment, identify suspicious lung side and zone</think> "
        "<act>one bounding box [x1,y1,x2,y2] in 0-1000, or 'No TB lesion'</act> "
        "<rethink>detailed clinical description of the localized finding</rethink> "
        "<answer>final TB decision starting with Yes or No</answer>"
    )

    # ── Decomposed process reward functions ───────────────────────────────
    # Budget: think(0.10) + spatial(0.20) + clinical(0.15) + answer(0.50) + format(0.05) = 1.00
    # Answer weight increased 0.35->0.50 to directly optimise classification accuracy

    ZONE_KWORDS = {
        "upper": ["upper","apical","apex","apices"],
        "mid":   ["mid","middle","perihilar","hilar"],
        "lower": ["lower","basal","base"],
    }
    SIDE_KWORDS = {
        "left":  ["left"],
        "right": ["right"],
        "hilar": ["hilar","central","bilateral"],
    }
    POS_CLINICAL = ["cavity","cavitary","cavitation","consolidation","opacity",
                    "infiltrate","lesion","nodular","tree-in-bud","tuberculosis",
                    "tb","upper lobe","apical"]
    NEG_CLINICAL = ["no evidence","no tb","no focal","clear","normal",
                    "not tuberculosis","non-tb","no active"]

    def _contains(text, terms):
        t=(text or "").lower(); return any(term in t for term in terms)

    def r_think(think_t, gt_box):
        if gt_box is None or not think_t:
            return 0.10 if _contains(think_t,
                ["no focal","no suspicious","no tb","normal","clear lung"]) else 0.03
        cx=(gt_box[0]+gt_box[2])/2.0; cy=(gt_box[1]+gt_box[3])/2.0
        gt_side="right" if cx<480 else ("left" if cx>520 else "hilar")
        gt_zone="upper" if cy<350 else ("mid" if cy<650 else "lower")
        score=0.0
        if _contains(think_t, SIDE_KWORDS.get(gt_side,[gt_side])): score+=0.04
        if _contains(think_t, ZONE_KWORDS.get(gt_zone,[gt_zone])): score+=0.06
        return score

    def r_spatial(act_t, gt_box):
        if gt_box is None:
            return 0.20 if not parse_all_bboxes(act_t) else 0.0
        pred_boxes=parse_all_bboxes(act_t)
        return 0.20*best_iou(pred_boxes, gt_box) if pred_boxes else 0.0

    def r_clinical(rethink_t, answer_t, gt_dec):
        combined=f"{rethink_t} {answer_t}".lower()
        if gt_dec=="yes":
            hits=sum(1 for t in POS_CLINICAL if t in combined)
            return 0.15 if hits>=3 else (0.10 if hits>=2 else (0.05 if hits>=1 else 0.0))
        if _contains(combined, NEG_CLINICAL): return 0.15
        return 0.05 if parse_yesno(answer_t)=="no" else 0.0

    def r_answer(answer_t, pred_t, gt_dec, gt_box):
        pred_dec   = parse_yesno(answer_t) or parse_yesno(pred_t)
        pred_boxes = parse_all_bboxes(answer_t) or parse_all_bboxes(pred_t)
        pred_pos   = (pred_dec=="yes") or (pred_dec is None and bool(pred_boxes))
        score=0.0
        if gt_dec=="yes":
            if pred_pos:
                score+=0.38
                if gt_box and pred_boxes: score+=0.07*best_iou(pred_boxes, gt_box)
            else:
                score-=0.38  # symmetric heavy FN penalty
        else:
            score+=0.38 if not pred_pos else -0.38  # symmetric FP penalty
        if parse_yesno(answer_t) is not None: score+=0.05
        return score
        # Budget: think(0.10)+spatial(0.20)+clinical(0.15)+answer(0.50)+format(0.05)=1.00
        # Normalization range: [-0.38-0.38, 0.10+0.20+0.15+0.50+0.05] = [-0.76, 1.00]
        # Map [-0.76, 1.00] -> [0,1]: (total+0.76)/1.76

    def r_format(pred_t):
        n=sum(1 for t in ("think","act","rethink","answer") if extract_tag(pred_t,t))
        return 0.05*(n/4)

    # ── GRPO training dataset ────────
    GRPO_POS_N = int(os.environ.get("GRPO_POS_N","200"))
    GRPO_NEG_N = int(os.environ.get("GRPO_NEG_N","200"))
    pos_pool   = [p for p in vqa_train if decision_gold(p)=="yes"]
    neg_pool   = [p for p in vqa_train if decision_gold(p)=="no"]
    grpo_pairs = (random.sample(pos_pool, min(GRPO_POS_N,len(pos_pool)))+
                  random.sample(neg_pool, min(GRPO_NEG_N,len(neg_pool))))
    random.shuffle(grpo_pairs)
    pairs_list = grpo_pairs
    print(f"GRPO train prompts:{len(grpo_pairs)}  "
          f"pos:{min(GRPO_POS_N,len(pos_pool))} neg:{min(GRPO_NEG_N,len(neg_pool))} (balanced 1:1)")

    def make_hf_row(pair):
        q=pair["conversations"][0]["value"]
        messages=[{"role":"system","content":SYSTEM_TARA},
                  {"role":"user",  "content":q}]
        prompt=processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return {"prompt":prompt, "image_path":pair["image"]}

    hf_dataset = HFDataset.from_list([make_hf_row(p) for p in grpo_pairs])

    reward_log = []

    def process_reward_fn(completions, prompts=None, **kwargs):
        rewards=[]
        batch_idx=list(range(len(completions)))
        if "idx" in kwargs: batch_idx=kwargs["idx"]
        batch_pairs=[pairs_list[min(i,len(pairs_list)-1)] for i in batch_idx]
        for pred,pair in zip(completions, batch_pairs):
            try:
                gt_dec  = decision_gold(pair)
                gt_box  = parse_bbox(pair["conversations"][1]["value"])
                think_t   = extract_tag(pred,"think")
                act_t     = extract_tag(pred,"act")
                rethink_t = extract_tag(pred,"rethink")
                answer_t  = extract_tag(pred,"answer") or pred
                rt = r_think(think_t, gt_box)
                rs = r_spatial(act_t, gt_box)
                rc = r_clinical(rethink_t, answer_t, gt_dec)
                ra = r_answer(answer_t, pred, gt_dec, gt_box)
                rf = r_format(pred)
                total = rt+rs+rc+ra+rf
                normalized = float(max(0.0, min(1.0, (total+0.76)/1.76)))
                rewards.append(normalized)
                reward_log.append(normalized)
                if len(reward_log)%40==0:
                    w=reward_log[-40:]
                    print(f"[GRPO-F step ~{len(reward_log)//4}] "
                          f"mean_reward={sum(w)/len(w):.3f} "
                          f"min={min(w):.3f} max={max(w):.3f}")
            except: rewards.append(0.0)
        return rewards

    USE_BF16 = (torch.cuda.is_available() and
                torch.cuda.get_device_capability()[0]>=8)

    grpo_config = GRPOConfig(
        output_dir=PROCESS_CKPT_DIR,
        learning_rate=float(os.environ.get("GRPO_LR","3e-7")),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        num_generations=2,
        max_completion_length=192,
        max_prompt_length=768,
        temperature=1.0,
        beta=0.04,
        bf16=USE_BF16,
        fp16=(not USE_BF16),
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=SEED,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        log_completions=False)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[process_reward_fn],
        args=grpo_config,
        train_dataset=hf_dataset,
        processing_class=processor.tokenizer)

    print("Starting Process-Reward GRPO with Full TARA loop (First and Second Improvements F) ...")
    print("Reward budget v2: think=0.10 spatial=0.20 clinical=0.15 answer=0.50 format=0.05")
    trainer.train()

    model.save_pretrained(PROCESS_CKPT_DIR)
    processor.tokenizer.save_pretrained(PROCESS_CKPT_DIR)
    processor.save_pretrained(PROCESS_CKPT_DIR)
    print(f"First and Second Improvements F saved -> {PROCESS_CKPT_DIR}")
    if reward_log:
        last50=reward_log[-50:]
        print(f"First and Second reward stats: mean={sum(last50)/len(last50):.3f} "
              f"min={min(last50):.3f} max={max(last50):.3f} "
              f"n_above_0.6={sum(1 for r in last50 if r>0.6)}")

    # ── Evaluation ─────────────────────────────────────────────────────────
    model.eval()
    EVAL_SYS = (
        "You are a tuberculosis radiology assistant. Output: "
        "<think>...</think><act>...</act><rethink>...</rethink><answer>...</answer>"
    )

    def infer(image_path, question, max_new=192):
        try: img=Image.open(image_path).convert("RGB")
        except: return ""
        msgs=[{"role":"system","content":EVAL_SYS},
              {"role":"user",  "content":[{"type":"image"},{"type":"text","text":question}]}]
        text=processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs=processor(text=[text], images=[img], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out=model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        return processor.batch_decode(
            out[:,inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    preds=[]
    for i,pair in enumerate(eval_subset):
        preds.append(infer(pair["image"], pair["conversations"][0]["value"]))
        if (i+1)%20==0: print(f"Eval {i+1}/{len(eval_subset)}")

    res=evaluate_predictions(preds, eval_subset)
    print("\n[First and Second Improvements F — Process GRPO] RESULTS:")
    print(json.dumps({k:v for k,v in res.items() if k!="examples"}, indent=2))
    with open(os.path.join(PROCESS_CKPT_DIR,"eval_metrics.json"),"w") as f:
        json.dump({k:v for k,v in res.items() if k!="examples"}, f, indent=2)
    with open(os.path.join(PROCESS_CKPT_DIR,"eval_examples.json"),"w") as f:
        json.dump(res.get("examples",[]), f, indent=2)

    vol.commit()
    return {k:v for k,v in res.items() if k!="examples"}


# ── Full comparison: D vs E vs F ──────────────────────────────────────────────
@app.function(
    volumes={DATASET_MOUNT_PATH: vol},
)
def make_full_comparison():
    import json, os
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    checkpoints = {
        "Baseline D (SFT)":          "/mnt/my-dataset/outputs/qwen2vl_sft_full",
        "Improvement E (Outcome GRPO)": "/mnt/my-dataset/outputs/qwen2vl_outcome_grpo",
        "Improvement F (Process GRPO)": "/mnt/my-dataset/outputs/qwen2vl_process_grpo",
    }
    rows=[]
    for name,ckpt in checkpoints.items():
        mp=os.path.join(ckpt,"eval_metrics.json")
        if not os.path.exists(mp): print(f"Missing: {mp}"); continue
        with open(mp) as f: m=json.load(f)
        rows.append({"Model":name,
                     "Accuracy":  round(m.get("yn_acc",  float("nan")),4),
                     "Sensitivity":round(m.get("yn_sens", float("nan")),4),
                     "Specificity":round(m.get("yn_spec", float("nan")),4),
                     "mean_IoU":  round(m.get("mean_iou",float("nan")),4),
                     "IoU@0.5":   round(m.get("iou@0.5", float("nan")),4),
                     "Struct%":   round(m.get("structured_rate",float("nan")),4)})
    if not rows: print("No metrics found."); return
    df=pd.DataFrame(rows)
    print("\n"+"="*70)
    print("FULL VLM COMPARISON  D vs E vs F")
    print("="*70)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUTPUTS_DIR,"full_vlm_comparison.csv"),index=False)

    metrics=["Accuracy","Sensitivity","Specificity","mean_IoU","IoU@0.5","Struct%"]
    x=np.arange(len(df)); w=0.13
    colors=plt.cm.Set2(np.linspace(0,1,len(metrics)))
    fig,ax=plt.subplots(figsize=(14,6))
    for i,m in enumerate(metrics):
        vals=df[m].values
        bars=ax.bar(x+i*w, vals, w, label=m, color=colors[i])
        for bar,val in zip(bars,vals):
            if not np.isnan(val):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6, rotation=45)
    ax.set_xticks(x+w*(len(metrics)//2))
    ax.set_xticklabels(df["Model"], fontsize=9)
    ax.set_ylim(0,1.25); ax.set_ylabel("Score")
    ax.set_title("TB-ViTAR: SFT vs Outcome-GRPO vs Process-GRPO")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    plt.tight_layout()
    out_png=os.path.join(OUTPUTS_DIR,"full_comparison_D_E_F.png")
    plt.savefig(out_png, dpi=150)
    print(f"Plot saved -> {out_png}")

    all_ex={}
    for name,ckpt in checkpoints.items():
        ep=os.path.join(ckpt,"eval_examples.json")
        if os.path.exists(ep):
            with open(ep) as f: all_ex[name]=json.load(f)
    with open(os.path.join(OUTPUTS_DIR,"all_qual_examples.json"),"w") as f:
        json.dump(all_ex,f,indent=2)
    print("Qualitative examples saved.")
    vol.commit()


# ── Entry point ───────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    print("="*60)
    print("First and Second Improvements F: Process-Reward GRPO (TARA loop)")
    print("="*60)
    res_f = run_first_and_second_improvement_F.remote()
    print("\nFirst and Second Improvements F DONE:")
    for k,v in res_f.items():
        print(f"  {k}: {v}")

    print("\n"+"="*60)
    print("Generating full comparison D vs E vs F ...")
    print("="*60)
    make_full_comparison.remote()

    print("\n"+"="*60)
    print("ALL DONE — Results in Modal volume my-dataset/outputs/")
    print("  qwen2vl_process_grpo/eval_metrics.json  <- Improvement F")
    print("  full_vlm_comparison.csv                 <- All three models")
    print("  full_comparison_D_E_F.png               <- Bar chart")
    print("="*60)
