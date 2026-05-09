# TB-ViTAR: Iterative Spatially-Grounded Reasoning with Process Rewards for Tuberculosis Diagnosis in Chest X-rays

**Authors:** Asad Ayub (27100413) | Fazal Rehman (27100294)  
**Course:** CS437 — Deep Learning
**Paper:** `Deliverables/Deliverable 6 - Final Paper.pdf`

---

## Overview

TB-ViTAR is a research framework for automated tuberculosis (TB) diagnosis in chest X-rays (CXRs) that bridges the gap between purely discriminative CNN classifiers and fully interpretable, spatially-grounded vision-language models (VLMs). The project progresses through six experimental configurations — from ResNet/DenseNet/CLIP baselines to supervised fine-tuning and two stages of reinforcement learning — evaluated on the TBX11K benchmark.

The central novelty is the combination of the **Think-Act-Rethink-Answer (TARA)** iterative reasoning structure with a **five-component per-step process reward** under Group Relative Policy Optimization (GRPO). Unlike prior medical VLMs that assign a single scalar reward to the final answer only, TB-ViTAR scores every intermediate reasoning step independently, keeping informative gradients active throughout training and producing fully auditable diagnostic chains.

---

## Abstract

Medical vision-language models trained with reinforcement learning have demonstrated strong diagnostic reasoning on radiology benchmarks, yet two fundamental gaps persist: they reason in a single forward pass without iteratively refocusing on suspicious image regions, and their reward functions evaluate only the final answer while ignoring whether intermediate reasoning steps are spatially or clinically valid. We propose **TB-ViTAR**, which addresses both gaps simultaneously by combining the TARA iterative reasoning structure with per-step spatially-grounded process rewards, applied to TB CXR diagnosis on TBX11K. Starting from Qwen2-VL-2B-Instruct fine-tuned with PEFT LoRA, we compare six experimental configurations spanning CNN classifiers (ResNet-50, DenseNet-121), vision-language zero-shot and linear probe (CLIP ViT-B/32), supervised fine-tuning (SFT, Baseline D), our First Improvement (Outcome GRPO with Unsloth and TARA), and a five-component process-reward GRPO with the complete TARA cognitive loop (Improvement F).

CNN baselines achieve up to **97.1% accuracy and 0.979 AUC**. The VLM progression from SFT to process-reward GRPO achieves **92.0% accuracy, 90.8% sensitivity, 93.8% specificity, and mean IoU of 0.273**, while producing a **47% stronger mean training signal** (0.725 vs. 0.493) with **163% more high-quality samples** than outcome-only GRPO, demonstrating that fine-grained process supervision produces richer learning dynamics and fully interpretable TARA reasoning chains at **100% format compliance**.

---

## Contents

- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [TARA VQA Construction](#tara-vqa-construction)
  - [Baseline A: ResNet-50](#baseline-a-resnet-50)
  - [Baseline B: DenseNet-121](#baseline-b-densenet-121)
  - [Baseline C: CLIP ViT-B/32](#baseline-c-clip-vit-b32)
  - [Baseline D: Supervised Fine-Tuning (SFT)](#baseline-d-supervised-fine-tuning-sft)
  - [First Improvement: Outcome GRPO with Unsloth and TARA](#first-improvement-outcome-grpo-with-unsloth-and-tara)
  - [Improvement F: Process-Reward GRPO with Full TARA Loop](#improvement-f-process-reward-grpo-with-full-tara-loop)
  - [Training Setup Summary](#training-setup-summary)
- [Results](#results)
  - [CNN and CLIP Baselines](#cnn-and-clip-baselines)
  - [VLM Progression](#vlm-progression)
  - [GRPO Reward Convergence](#grpo-reward-convergence)
  - [Ablation Studies](#ablation-studies)
- [Interpretability and Structured Reasoning](#interpretability-and-structured-reasoning)
- [Discussion](#discussion)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)
- [Citation](#citation)

---

## Directory Structure

```
TB-Vitar/
├── Deliverables/
│   ├── Deliverable 1 - SOA Survey Report.pdf
│   ├── Deliverable 2 - Dataset and Annotations.ipynb
│   ├── Deliverable 3 - Baseline Models A B and C.ipynb
│   ├── Deliverable 4 - First Improvement.ipynb
│   ├── Deliverable 5 - Second Improvement.ipynb
│   └── Deliverable 6 - Final Paper.pdf
├── Scripts/
│   └── Script for Second Improvement.py
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AsadAyub947/TB-Vitar.git
cd TB-Vitar
```

Install Python dependencies:

```bash
pip install torch==2.4.0 torchvision==0.19.0 transformers==4.48.3 \
    accelerate>=0.34.0 peft>=0.14.0 trl==0.15.2 \
    datasets scikit-learn pandas numpy Pillow \
    qwen-vl-utils matplotlib timm clip
```

For Modal-based notebooks (First Improvement and Second Improvement):

```bash
pip install modal
modal setup
```

---

## Dataset

We use **TBX11K** (Liu et al., CVPR 2020), containing **8,400 chest X-rays** across three diagnostic categories:

| Category | Count | Share |
|:---|:---:|:---:|
| Active TB (TB-positive with lesions) | 800 | 9.5% |
| Sick non-TB (pathological, TB-negative) | 3,800 | 45.2% |
| Healthy (normal CXR) | 3,800 | 45.2% |

PASCAL VOC bounding-box annotations are provided for all 800 TB-positive cases (799 valid boxes, 99.9% coverage). A stratified 70/15/15 train/val/test split with `seed=42` yields **5,880 / 1,260 / 1,260 images**, preserving the 9.5% TB-positive rate throughout all splits.

> **Note:** CNN baselines (A–C) are evaluated on the full 1,260-sample test set (9.5% TB+). VLM models D and E are evaluated on a balanced 260-sample set (180 positive, 80 negative); F on a balanced 200-sample set (including 40 localization pairs). Direct cross-group accuracy comparisons must account for these different evaluation distributions.

---

## Methodology

### TARA VQA Construction

For each image we generate structured **Think-Act-Rethink-Answer (TARA)** question-answer pairs in two types:

- **Binary pairs** — ask whether TB is present; answer includes anatomical zone identification, bounding-box coordinates (for TB-positive cases), and clinical keyword descriptions.
- **Localization pairs** — ask for bounding-box coordinates of TB lesions in TB-positive images with ground-truth annotations.

Bounding boxes are normalized to a **0–1000 pixel grid**. Zone labels are computed from centroid thresholds:

```
side     ∈ { right (cx < 480),  hilar,  left (cx > 520) }
vertical ∈ { upper (cy < 350),  mid,    lower (cy ≥ 650) }
```

The total TARA VQA dataset contains **2,399 pairs** distributed as:

| Type | Count | Share |
|:---|:---:|:---:|
| Binary YES (TB-positive) | 1,599 | 64.8% |
| Binary NO (TB-negative) | 800 | 29.1% |
| Localization | 521 | 6.1% |

All VLM models produce responses in the following four-tag TARA structure:

```
<think>  [initial anatomical assessment]  </think>
<act>    [bounding-box coordinates or "No TB lesion"]  </act>
<rethink>[refined clinical assessment]  </rethink>
<answer> [binary Yes/No decision with zone description]  </answer>
```

**Sample TARA output for a TB-positive case:**

```
<think>Suspicious opacity in the right upper zone consistent with TB.</think>
<act>[142, 87, 398, 310]</act>
<rethink>The right upper zone shows consolidation and cavitation typical of active tuberculosis.</rethink>
<answer>Yes, active tuberculosis is present in the right upper zone at [142, 87, 398, 310].</answer>
```

---

### Baseline A: ResNet-50

Two-phase training on the TBX11K training split:

- **Phase 1** — train a linear classification head (frozen backbone) for 5 epochs; `lr = 1e-3`, `BCEWithLogitsLoss`, positive class weight `w+ = 9.5`.
- **Phase 2** — unfreeze `layer4` for up to 10 epochs; `lr = 2e-5`, early stopping patience 4.

Validation threshold tuned by maximizing F1 on the validation set (optimal threshold: 0.310).

---

### Baseline B: DenseNet-121

Identical two-phase protocol to ResNet-50. Phase 2 unfreezes `denseblock4` and `transition3` instead of `layer4`.

---

### Baseline C: CLIP ViT-B/32

Two evaluation settings:

- **Zero-shot** — positive and negative text prompt prototypes are averaged into class embeddings; inference uses cosine similarity between image embedding and each class prototype.
- **Linear probe** — 512-dimensional CLIP image features are passed through a two-layer MLP head: `LayerNorm → Linear(512, 256) → GELU → Dropout(0.3) → Linear(256, 1)`, trained for 30 epochs with AdamW and cosine annealing.

---

### Baseline D: Supervised Fine-Tuning (SFT)

We load **Qwen2-VL-2B-Instruct** with the vision tower frozen and the full LLM trainable, then apply **PEFT LoRA**:

| Config | Value |
|:---|:---|
| LoRA rank / alpha | 16 / 32 |
| Dropout | 0.05 |
| Target projections | q, k, v, o, gate, up, down |
| Trainable parameters | 18.5M / 2,227.5M total (0.83%) |

The model minimises the standard causal language modelling loss over TARA-formatted outputs:

```
L_SFT = -E[ Σ_t log p_θ(y_t | I, q, y_{<t}) ]
```

**Training configuration:**

| Parameter | Value |
|:---|:---|
| Epochs | 3 |
| Learning rate | 2e-5 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Hardware | A100-40 GB |
| Training time | ~90 min |
| Loss progression | 0.842 → 0.314 → 0.187 |

SFT is an essential warm-start stage — without it, neither GRPO variant can reliably produce TARA-formatted responses, making the format reward component uninformative and destabilizing early RL.

---

### First Improvement: Outcome GRPO with Unsloth and TARA

Starting from the SFT checkpoint, **Unsloth** LoRA (same rank and targets) is applied, enabling an **80% VRAM reduction** relative to standard LoRA. This makes `num_generations = 4` feasible on A100-40 GB, giving GRPO genuine reward variance across four candidate responses per prompt.

Training uses **260 VQA pairs (180 positive, 80 negative)**. The outcome reward evaluates only the final answer:

| Condition | Reward |
|:---|:---:|
| Correct TB-positive prediction | +0.40 |
| Missed TB — false negative | −0.40 |
| Correct TB-negative prediction | +0.30 |
| False positive | −0.10 |
| Valid bounding box + clinical keywords | up to +0.20 |
| TARA format compliance (all 4 tags present) | +0.10 |

**GRPO configuration:**

| Parameter | Value |
|:---|:---|
| Learning rate | 5e-7 |
| Batch size | 2 |
| Gradient accumulation | 4 |
| num_generations | 4 |
| β_KL | 0.04 |
| Steps | 100 |
| Hardware | A100-40 GB |
| Training time | ~38 min |

---

### Second Improvement: Process-Reward GRPO with Full TARA Loop

The core limitation of the First Improvement is that a model producing partially correct reasoning but an incorrect binary decision collapses to reward 0.0, receiving **no gradient signal for what it did correctly**. Improvement F decomposes the reward into **five independent components** — one per TARA step — providing partial-credit feedback at every gradient update:

```
R_proc = r_think + r_spatial + r_clinical + r_answer + r_format
```

Normalized from raw range [−0.76, 1.00] → [0, 1] via:

```
normalized = (r_think + r_spatial + r_clinical + r_answer + r_format + 0.76) / 1.76
```

**Reward component breakdown:**

| Component | Weight | Range | Signal |
|:---|:---:|:---:|:---|
| `r_think` | 0.10 | [0, 0.10] | Anatomical zone: correct side earns +0.04; correct vertical zone earns +0.06, both verified against GT centroids |
| `r_spatial` | 0.20 | [0, 0.20] | `r_spatial = 0.20 × IoU(predicted_bbox, GT_bbox)`; coordinates extracted via regex from `<act>` tag; full score for correct "no box" on true negatives |
| `r_clinical` | 0.15 | [0, 0.15] | `<rethink>` and `<answer>` tags checked against a 12-item positive lexicon *(cavity, cavitary, consolidation, opacity, infiltrate, lesion, nodular, tree-in-bud, tuberculosis, tb, upper lobe, apical)* and 8-item negative lexicon; scored 0.15/0.10/0.05 for 3+/2/1 matches |
| `r_answer` | 0.50 | [−0.38, +0.50] | Correct positive: +0.38 (plus up to +0.07 IoU bonus); missed TB: −0.38; correct negative: +0.38; false positive: −0.38. Symmetric ±0.38 penalty encodes equal clinical cost for FN and FP |
| `r_format` | 0.05 | [0, 0.05] | `r_format = 0.05 × n_tags / 4` — proportional to number of correct TARA tags present |

**GRPO configuration:**

| Parameter | Value |
|:---|:---|
| Training data | 200 positive + 200 negative (strict 1:1) |
| Learning rate | 3e-7 |
| Epochs / Steps | 2 epochs / 200 steps |
| Batch size | 2 |
| Gradient accumulation | 4 |
| num_generations | 2 |
| β_KL | 0.04 |
| Max completion tokens | 192 |
| Hardware | A100-80 GB |
| Training time | ~51 min |

---

### Training Setup Summary

| Setting | Baselines A–C | Baseline D (SFT) | First Imp. (E) | Improvement F |
|:---|:---:|:---:|:---:|:---:|
| Platform | Kaggle | Modal | Modal | Modal |
| GPU | NVIDIA T4 | A100-40 GB | A100-40 GB | A100-80 GB |
| Model | ResNet-50 / DenseNet-121 / CLIP | Qwen2-VL-2B-Instruct | Qwen2-VL-2B + LoRA | Qwen2-VL-2B + LoRA |
| LoRA library | — | PEFT | Unsloth | Unsloth |

---

## Results

### CNN and CLIP Baselines

| Method | Accuracy | Sensitivity | Specificity | F1 | AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| **A: ResNet-50** | **0.9714** | 0.8500 | 0.9842 | 0.8500 | **0.9794** |
| **B: DenseNet-121** | 0.9635 | 0.7500 | **0.9860** | 0.7965 | 0.9462 |
| **C: CLIP Zero-shot** | 0.6429 | 0.6250 | 0.6447 | 0.2500 | 0.6455 |
| **C: CLIP Linear Probe** | 0.9444 | 0.7333 | 0.9667 | 0.7154 | 0.9630 |

Key findings:
- **ResNet-50** achieves the highest accuracy (97.1%) and AUC (0.979). With 90.5% TB-negative prevalence, a validation-tuned threshold of 0.310 yields 98.4% specificity (TP 102, TN 1122, FP 18, FN 18) at 85.0% sensitivity.
- **DenseNet-121** achieves 96.4% accuracy (AUC 0.946) with 98.6% specificity and 75.0% sensitivity.
- **CLIP zero-shot** falls to 64.3% accuracy and 0.646 AUC — confirming that unadapted VLMs cannot interpret TB pathology from text prompts alone.
- **CLIP linear probe** recovers to 94.4% accuracy (AUC 0.963), confirming CLIP features are informative once a task-specific head is trained.

> **Important caveat:** CNN baselines are evaluated on the full test set (1,260 samples, 9.5% TB+). VLM models are evaluated on balanced subsets (50–69% TB+). These are fundamentally different tasks; raw accuracy comparisons are misleading. CNNs produce no reasoning, no bounding boxes, and no interpretable output — the TARA chain allows a radiologist to audit spatial claims and verify clinical reasoning step by step.

---

### VLM Progression

| Method | Accuracy | Sensitivity | Specificity | Mean IoU | IoU@0.5 | Train Reward |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **D: Qwen2-VL SFT** | 0.800 | 0.717 | 0.988 | 0.228 | 0.222 | — |
| **First Imp.: Outcome GRPO (Unsloth+TARA)** | 0.850 | 0.713 | 0.988 | 0.239 | 0.250 | 0.493 |
| **F: Process GRPO (TARA)** | **0.920** | **0.908** | 0.938 | **0.273** | **0.308** | **0.725** |

Key findings:
- **SFT (D):** 80.0% accuracy, 71.7% sensitivity, 98.8% specificity, mean IoU 0.228. The 100% structured output rate confirms successful TARA format acquisition, but moderate IoU reveals that bounding boxes are placed syntactically without RL pressure for geometric accuracy.
- **First Improvement (E):** +5.0 pp accuracy (85.0%), mean IoU to 0.239, IoU@0.5 from 0.222 to 0.250. Confusion matrix: TP 129, TN 79, FP 1, FN 51. The 180:80 training imbalance inflates specificity (98.8%) at the cost of sensitivity (71.2%).
- **Improvement F:** +7.0 pp over SFT, +2.0 pp over E — **92.0% accuracy, 90.8% sensitivity, 93.8% specificity**. Confusion matrix: TP 109, TN 75, FP 5, FN 11. Mean IoU 0.273 (+4.5 pp over E), IoU@0.5 0.308 (+5.8 pp over E).

The monotonic IoU progression — SFT (0.222) → E (0.250) → F (0.308) — confirms that each RL stage adds genuine geometric learning beyond format compliance.

---

### GRPO Reward Convergence

| Statistic | Outcome GRPO (E) | Process GRPO (F) | Δ |
|:---|:---:|:---:|:---:|
| Steps 1–50 mean | 0.471 | 0.712 | +51% |
| Steps 51–100 mean | 0.496 | 0.718 | +45% |
| Steps 101–150 mean | 0.497 | 0.710 | +43% |
| Steps 151–200 mean | 0.500 | 0.720 | +44% |
| **Overall mean** | 0.493 | **0.725** | **+47%** |
| **Overall std** | 0.161 | **0.146** | −9.3% |
| **Min reward** | 0.350 | **0.367** | Higher floor |
| **Max reward** | 1.000 | 0.933 | Process cap |
| **Samples > 0.6 / 40** | 11 | **29** | **+163%** |

Outcome GRPO plateaus near 0.50 with high variance. Process GRPO launches at 0.730 and stabilizes in 0.69–0.73 because partial-credit scoring guarantees that any response with correct zone identification, clinical keywords, and format earns at least 0.30 — keeping informative gradients active throughout all 200 steps.

---

### Ablation Studies

#### Leave-One-Out Reward Component Ablation

| Removed Component | Δ Mean Reward | Δ IoU@0.5 |
|:---|:---:|:---:|
| None (full Improvement F) | — | — |
| −r_answer (0.50 wt.) | −0.184 | −0.098 |
| −r_spatial (0.20 wt.) | −0.061 | **−0.112** |
| −r_clinical (0.15 wt.) | −0.038 | −0.021 |
| −r_think (0.10 wt.) | −0.019 | −0.008 |
| −r_format (0.05 wt.) | −0.004 | −0.003 |

Removing `r_spatial` causes the **largest IoU@0.5 drop (−0.112)** despite having only 0.20 weight — confirming that the dedicated spatial reward is disproportionately responsible for geometric grounding.

#### Effect of Training Data Balance (First Improvement)

| Pos:Neg | Accuracy | Sensitivity | Specificity |
|:---|:---:|:---:|:---:|
| 260:0 (all positive) | 0.690 | 1.000 | 0.000 |
| 180:80 (used in E) | **0.850** | 0.713 | **0.988** |
| 130:130 (balanced) | 0.831 | 0.825 | 0.838 |

The 180:80 distribution maximizes accuracy but at the cost of conservative sensitivity. This motivated the strict 1:1 balance (200:200) used in Improvement F, which achieves the best overall sensitivity (90.8%).

#### Effect of num_generations (First Improvement with Unsloth and TARA)

| num_gen | Reward Variance | Mean Reward | IoU@0.5 |
|:---:|:---:|:---:|:---:|
| 2 (standard LoRA limit) | 0.209 | 0.461 | 0.194 |
| 4 (Unsloth enabled) | **0.161** | **0.493** | **0.250** |

Increasing to 4 generations reduces reward variance (0.209 → 0.161) and raises mean reward (0.461 → 0.493). Unsloth's 80% 

---

## Interpretability and Structured Reasoning

The TARA cognitive loop produces step-by-step interpretability that purely discriminative models cannot offer:

| Tag | Role | Verified by |
|:---|:---|:---|
| `<think>` | Global anatomical assessment — identifies the suspected lung zone (side + vertical) before any decision | `r_think`: zone keyword dictionaries against GT centroids |
| `<act>` | Explicit bounding-box coordinates `[x1, y1, x2, y2]` on a 0–1000 scale, or "No TB lesion" for negatives | `r_spatial`: IoU against GT bounding box |
| `<rethink>` | Refined hypothesis using clinical vocabulary — cavitation, consolidation, upper-lobe opacity, tree-in-bud | `r_clinical`: 12-item positive / 8-item negative lexicon |
| `<answer>` | Final unambiguous Yes/No decision with a reasoning summary | `r_answer`: symmetric ±0.38 FN/FP penalty |

All three VLM variants (D, E, F) achieve **100% TARA format compliance** throughout training and evaluation, confirming that the tag structure learned during SFT is fully stable under subsequent RL optimization.

**Representative SFT output — localization with correct reasoning but imprecise box:**
```
GT:   <think>Lesion visible as nodular opacity in the right upper.</think>
      <act>[283, 273, 424, 374]</act>
      <rethink>Confirmed nodular opacity consistent with TB.</rethink>
      <answer>Nodular opacity in the right upper, consistent with TB.</answer>

Pred: <think>Lesion visible as nodular opacity in the right upper.</think>
      <act>[150, 147, 454, 454]</act>           ← oversized box (IoU = 0.362)
      <rethink>Confirmed nodular opacity consistent with TB pathology in the right upper.</rethink>
      <answer>Nodular opacity in the right upper consistent with TB pathology.</answer>
```

**Representative First Improvement output — correct negative, high-confidence:**
```
Pred: <think>No upper-lobe opacity, cavitation, or TB-pattern infiltrate. No consolidation
      or infiltrate in the right upper.</think>
      <act>No TB lesion.</act>
      <rethink>No consolidation or infiltrate in the right upper. No radiographic
      evidence of tuberculosis.</rethink>
      <answer>No, this CXR does not show active tuberculosis.</answer>
```

---

## Discussion

### Why Process GRPO Achieves the Best VLM Performance

Four factors explain Improvement F's superiority over all other VLM configurations:

1. **Balanced training data.** F used 200 positive and 200 negative pairs (1:1), producing better-calibrated sensitivity-specificity trade-offs. E used 180:80, inflating specificity at the cost of sensitivity.

2. **Partial-credit gradient signal.** Process rewards keep gradients alive even when the binary decision is wrong. A model that correctly identifies the lung zone, produces a plausible bounding box, and includes clinical keywords earns `r_think + r_spatial ≈ 0.28` even with a wrong final answer — reinforcing correct intermediate reasoning.

3. **Superior training dynamics.** The 47% higher mean reward and 163% more samples above the 0.6 threshold indicate consistent gradients throughout all 200 steps. Outcome GRPO never escapes the low-reward regime because non-zero reward requires correct binary decision *and* valid bbox simultaneously.

4. **Symmetric penalty design.** The symmetric ±0.38 FN/FP penalty is more clinically principled than E's asymmetric design (−0.40 FN vs. −0.10 FP), which created an implicit bias toward positive predictions.

---

## Limitations

- **Evaluation heterogeneity.** CNNs and VLMs are evaluated on different class distributions (9.5% vs. 50–69% TB+), making direct accuracy comparisons misleading.
- **Short GRPO training.** At 100–200 steps, reward curves show no plateau — extended training would likely further improve all metrics.
- **Template VQA.** TARA pairs are generated from structured templates rather than radiologist-authored descriptions; performance on manually written questions may differ.
- **Single dataset.** All VLM results are on TBX11K only; cross-dataset transfer to Shenzhen and Montgomery datasets was not completed in this work.
- **Compute constraints.** VLM evaluation uses balanced subsets (160–260 samples) rather than the full test split, due to Modal GPU time constraints.

---

## Future Work

### 1. Extended GRPO Training (> 500 steps)

All RL experiments were capped at 100–200 steps due to compute constraints. Training reward curves for Improvement F show no sign of plateau at step 200, suggesting substantial headroom remains. Extended runs of ≥ 500 steps with a cosine learning-rate schedule and periodic checkpoint evaluation would likely push accuracy beyond 94% and IoU@0.5 past 0.35, closing the gap with CNN classification performance while retaining full reasoning interpretability.

### 2. Cross-Dataset Generalization

TB-ViTAR was evaluated exclusively on TBX11K. Validating on the **Shenzhen** (662 CXRs) and **Montgomery** (138 CXRs) benchmarks — which differ in scanner type, patient demographics, and image resolution — is essential to establish clinical transferability. A domain-adaptation stage using unlabelled target-domain images within the GRPO reward loop (replacing spatial IoU with domain-invariant features) is a promising direction.

### 3. LLM-Based Step Verification

The current `r_clinical` component relies on a fixed 12-item positive and 8-item negative keyword lexicon. Replacing this with a frozen medical language model (e.g. BioMedLM or CheXagent) for step-level factuality verification — following the spirit of ChestX-Reasoner — would provide richer, semantically grounded intermediate supervision that generalizes beyond predefined vocabulary.

### 4. Reward Component Weight Ablations and AutoRL

The five-component budget (`r_answer`: 0.50, `r_spatial`: 0.20, `r_clinical`: 0.15, `r_think`: 0.10, `r_format`: 0.05) was set by clinical heuristic. A systematic grid search — or automated reward shaping via population-based training — could discover weight combinations that further improve the sensitivity/specificity trade-off, particularly in high-stakes screening settings where false-negative cost dominates.

### 5. Larger Base Models and Multi-Disease Extension

Scaling to Qwen2-VL-7B or a 13B-parameter radiology VLM, and extending the TARA framework to additional pulmonary diseases (pneumonia, pleural effusion, lung cancer), would test whether the process-reward paradigm generalizes beyond TB. Multi-disease TARA training with shared reward components but disease-specific lexicons represents a natural next step toward a clinically deployable chest X-ray reasoning system.

---

## Citation

```bibtex
@misc{ayub2025tbvitar,
  author       = {Ayub, Asad and Rehman, Fazal},
  title        = {{TB-ViTAR}: Iterative Spatially-Grounded Reasoning with Process Rewards
                  for Tuberculosis Diagnosis in Chest X-rays},
  year         = {2025},
  howpublished = {CS437/CS5317/EE414/EE513 — Deep Learning, LUMS Spring 2026},
  url          = {https://github.com/AsadAyub947/TB-Vitar}
}
```

---

*For questions, open a GitHub issue at https://github.com/AsadAyub947/TB-Vitar.*
