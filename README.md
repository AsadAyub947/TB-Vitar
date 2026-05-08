# TB-ViTAR: Spatially-Grounded Process-Reward Reinforcement Learning for Tuberculosis Detection

**Group 44** | Asad Ayub (27100413) & Fazal Rehman (27100294)

---

## Overview

TB-ViTAR (Tuberculosis Visual Thinking and Reasoning) is a progressive framework for automated TB detection and localization in chest X-rays. The project advances from traditional CNN baselines to reinforcement-learning-enhanced Vision-Language Models with structured reasoning chains.

## Key Results

| Method | Accuracy | Sensitivity | Specificity | Mean IoU | IoU@0.5 | Train Reward |
|--------|----------|-------------|-------------|---------|
| A: ResNet-50 | 0.8605 | 0.8577 | 0.8667| — | — | — |
| B: DenseNet-121 | 0.8534 | 0.8608  | 0.8368 | — | — | — |
| C: CLIP Zero-shot | 0.6629 | 0.5896  |  0.8263 | — | — | — |
| C: CLIP Linear Probe | 0.8664  | 0.8381 | 0.9298 | — | — | — |
| D: Qwen2-VL SFT | 0.904 | 0.887 | 0.950 | 0.193 | 0.173 | — |
| **E: Outcome GRPO** | **0.931** | **0.925** | 0.938 | **0.293** | **0.500** | 0.493 |
| F: Process GRPO with Full TARA Loop| 0.920 | 0.908 | 0.938 | 0.273 | 0.308 | **0.725** |

## Repository Structure

```
├── notebooks/
│   ├── Dataset and Annotations.ipynb 
│   ├── Baseline Models A B and C.ipynb 
│   ├── Baseline Models D and E.ipynb 
│   └── Process Reward GRPO with Full TARA Loop.ipynb
├── scripts/
│   ├── Baseline Models D and E.py                        
│   └── Process Reward GRPO with Full TARA Loop.py        
├── report/
│   └── main.tex            
└── README.md
```

## Methodology

### Progressive Architecture

1. **Baselines A-C** (Kaggle): ResNet-50, DenseNet-121, CLIP zero-shot/linear probe
2. **Baseline D** (Modal): Qwen2-VL-2B with PEFT LoRA supervised fine-tuning
3. **Baseline E** (Modal): Outcome-only GRPO on SFT checkpoint
4. **First and Second Improvements F** (Modal): Process-Reward GRPO with TARA cognitive loop

### TARA Cognitive Loop

The model outputs structured reasoning in four XML tags:
```
<think>Global chest assessment, identify suspicious lung zone</think>
<act>[x1,y1,x2,y2] bounding box or 'No TB lesion'</act>
<rethink>Clinical description of finding</rethink>
<answer>Final TB decision starting with Yes or No</answer>
```

### Decomposed Process Rewards

| Component | Weight | Signal |
|-----------|--------|--------|
| Think (zone ID) | 0.10 | Anatomical zone correctness |
| Spatial (IoU) | 0.20 | Bounding box overlap |
| Clinical | 0.15 | Medical keyword verification |
| Answer | 0.50 | Binary decision + FN penalty |
| Format | 0.05 | TARA tag compliance |

## Setup & Reproduction

### Requirements
- Python 3.10+
- PyTorch 2.4.0
- transformers 4.48.3
- trl 0.15.2
- peft >= 0.14.0
- Modal.com account (for notebooks 2-3)
- Kaggle account (for notebooks 0-1)

### Running on Kaggle (Notebooks 0-1)
Upload notebooks to Kaggle with TBX11K dataset attached. Enable GPU T4 accelerator.

### Running on Modal (Notebooks 2-3)
```bash
pip install modal
modal setup
python -m modal run Baseline Models D and E.py 
python -m modal run Process Reward GRPO with Full TARA Loop.py  
```

## Dataset

**TBX11K** (Liu et al., CVPR 2020): 11,200+ chest X-rays with bounding box annotations.
- 4 classes: Active TB, Latent TB, Sick non-TB, Healthy
- TB positive rate: ~69%
- Annotation format: PASCAL VOC XML bounding boxes
- Split: 70/15/15 stratified by 4-class label

## Technical Notes

- Model: Qwen2-VL-2B-Instruct with LoRA (r=16, alpha=32)
- GPU: NVIDIA A100 40/80GB (Modal.com)
- Training time: ~40 min per GRPO run (100 steps)
- All models maintain 100% TARA format compliance under RL training

## Authors

- Asad Ayub (27100413)
- Fazal Rehman (27100294)

## License

This project is for academic purposes (Deep Learning course deliverable).
