# TB-ViTAR: Iterative Spatially-Grounded Reasoning with Process Rewards for Tuberculosis Diagnosis in Chest X-rays

## Overview

This repository contains the codebase for a research project focused on the detection and localization of Tuberculosis (TB) in chest X-rays using a progressive framework that advances from traditional CNN classifiers to reinforcement-learning-enhanced Vision-Language Models (VLMs). The project combines the Think-Act-Rethink-Answer (TARA) structured reasoning paradigm with per-step spatially-grounded process rewards, applied to the TBX11K dataset.

## Abstract

We propose TB-ViTAR, a framework for automated TB chest X-ray diagnosis that combines iterative visual reasoning with decomposed process rewards. Starting from CNN and CLIP baselines (Baselines A–C), we progress through supervised fine-tuning of Qwen2-VL-2B-Instruct with PEFT LoRA (Baseline D), outcome-only GRPO (Baseline E), and a full process-reward GRPO with the TARA cognitive loop (Improvement F). Our outcome GRPO model achieves 93.1% accuracy and 92.5% sensitivity on TBX11K, with IoU@0.5 of 0.500 — a +7.1 pp accuracy gain and +32.7 pp localization improvement over the SFT baseline. Our process-reward GRPO model produces a 47% stronger mean training signal (0.725 vs. 0.493), demonstrating that fine-grained per-step process supervision yields substantially richer and more consistent learning dynamics than outcome-only rewards.

## Contents

- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Preprocessing and VQA Generation](#data-preprocessing-and-vqa-generation)
  - [Baseline Models A–C](#baseline-models-ac)
  - [Baseline D: Supervised Fine-Tuning](#baseline-d-supervised-fine-tuning)
  - [Baseline E: Outcome-Only GRPO](#baseline-e-outcome-only-grpo)
  - [Improvement F: Process-Reward GRPO with TARA Loop](#improvement-f-process-reward-grpo-with-tara-loop)
  - [Training Setup](#training-setup)
- [Results](#results)
- [Interpretability and Structured Reasoning](#interpretability-and-structured-reasoning)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)

## Directory Structure

```
Deliverables/
  ├── Deliverable 1 - SOA Survey Report.pdf 
  ├── Deliverable 2 - Dataset and Annotations.ipynb     
  ├── Deliverable 3 - Baseline Models A B and C.ipynb         
  ├── Deliverable 3 - Baseline Models D and E.ipynb
  ├── Deliverable 4 and 5 - Process Reward GRPO with Full TARA Loop.ipynb            
  └── Deliverable 6 - Final Paper.pdf 

Scripts/
  ├── Baseline Models D and E.py            
  └── Process Reward GRPO with Full TARA Loop.py 
README.md
```

## Installation

Clone this repository:

```bash
git clone https:github.com/AsadAyub947/TB-Vitar.git
cd TB-Vitar
```

Install Python dependencies:

```bash
pip install torch==2.4.0 torchvision==0.19.0 transformers==4.48.3 \
    accelerate>=0.34.0 peft>=0.14.0 trl==0.15.2 \
    datasets scikit-learn pandas numpy Pillow \
    qwen-vl-utils matplotlib timm clip
```

For Modal-based notebooks (Baselines D, E, and Improvement F):

```bash
pip install modal
modal setup
```

## Dataset

The study uses [TBX11K](https://mmcheng.net/tb/) (Liu et al., CVPR 2020), containing 11,200+ chest X-rays with four diagnostic categories and PASCAL VOC bounding-box annotations for TB lesions:

- **Active TB** — TB-positive with visible lesions
- **Latent TB** — TB-positive without acute radiographic findings
- **Sick non-TB** — Pathological but TB-negative
- **Healthy** — Normal chest X-ray

The dataset is split 70/15/15 (train/val/test) using stratified sampling by four-class label. Bounding-box coordinates are normalized to a 0–1000 pixel scale for spatial reward computation.

## Methodology

### Data Preprocessing and VQA Generation

Each image is paired with structured question-answer pairs using the TARA output format. Two types of pairs are generated:

- **Binary pairs**: Ask whether TB is present; the answer includes anatomical zone identification, bounding-box coordinates (if available), and clinical keyword descriptions.
- **Localization pairs**: Ask for bounding-box coordinates of TB lesions in TB-positive images with ground-truth annotations.

Training datasets are balanced at a 1:1 positive-to-negative ratio for binary pairs. Radiological convention is applied for zone labeling (image-left = patient's right lung).

A sample TARA-formatted answer for a TB-positive case:

```
<think>Suspicious opacity in the right upper zone consistent with TB.</think>
<act>[142, 87, 398, 310]</act>
<rethink>The right upper zone shows consolidation and cavitation typical of active tuberculosis.</rethink>
<answer>Yes, active tuberculosis is present in the right upper zone at [142, 87, 398, 310].</answer>
```

### Baseline Models A–C

**ResNet-50 (A) and DenseNet-121 (B)** are trained in two phases on Kaggle (T4 GPU). Phase 1 trains a linear classification head on a frozen backbone for 5 epochs. Phase 2 unfreezes the final convolutional block for up to 10 epochs with early stopping. Validation thresholds are tuned by maximizing F1 score.

**CLIP ViT-B/32 (C)** is evaluated in two settings:
- *Zero-shot*: Cosine similarity between image embeddings and averaged positive/negative text prompt embeddings.
- *Linear probe*: A two-layer MLP head trained on top of frozen CLIP image features for 30 epochs.

### Baseline D: Supervised Fine-Tuning

Qwen2-VL-2B-Instruct is loaded with PEFT LoRA (rank $r=16$, $\alpha=32$, dropout 0.05) targeting all attention and feed-forward projection matrices. This yields 18.5M trainable parameters out of 2,227.5M total (0.83%). Training runs for 3 epochs on Modal A100-40GB with:

```
Optimizer:          AdamW  (lr = 2e-5, betas = (0.9, 0.95))
Batch size:         4 (gradient accumulation = 4)
LoRA targets:       q, k, v, o, gate, up, down projections
```

The model learns the TARA tag structure and produces 100% format-compliant outputs at inference.

### Baseline E: Outcome-Only GRPO

Starting from the SFT checkpoint, GRPO is applied for 100 steps with a simple outcome reward:

| Condition | Reward |
|---|---|
| Correct TB-positive prediction | +0.40 |
| Missed TB (false negative) | −0.40 |
| Correct TB-negative prediction | +0.30 |
| False positive | −0.10 |
| Valid bounding box + clinical keywords | up to +0.20 |
| TARA format compliance | +0.10 |

```
GRPO config:  lr = 5e-7  |  batch = 2  |  grad_accum = 4
              num_generations = 2  |  beta = 0.04  |  temp = 1.0
```

### Improvement F: Process-Reward GRPO with TARA Loop

The key innovation. Five independent reward components evaluate each reasoning step separately:

| Component | Weight | Signal |
|---|:---:|---|
| `r_think` | 0.10 | Anatomical zone identification (side + vertical zone) |
| `r_spatial` | 0.20 | IoU between predicted and ground-truth bounding boxes |
| `r_clinical` | 0.15 | Clinical keyword verification (cavity, consolidation, opacity, etc.) |
| `r_answer` | 0.50 | Binary decision with symmetric ±0.38 FN/FP penalties |
| `r_format` | 0.05 | TARA tag structure compliance |
| **Total** | **1.00** | Normalized from raw range [−0.76, 1.00] → [0, 1] |

The normalization formula applied to raw scores is:

```
normalized = (r_think + r_spatial + r_clinical + r_answer + r_format + 0.76) / 1.76
```

Training uses a strictly balanced dataset (200 positive + 200 negative):

```
GRPO config:  lr = 3e-7  |  2 epochs  |  batch = 2  |  grad_accum = 4
              num_generations = 2  |  beta = 0.04  |  max_completion = 192 tokens
```

### Training Setup

| Setting | Baselines A–C | Baseline D | Baselines E & F |
|---|---|---|---|
| **Platform** | Kaggle | Modal | Modal |
| **GPU** | NVIDIA T4 | A100-40GB | A100-40GB / 80GB |
| **Model** | ResNet-50 / DenseNet-121 / CLIP | Qwen2-VL-2B | Qwen2-VL-2B + LoRA |
| **Framework** | PyTorch + timm | Transformers + TRL | TRL (GRPOTrainer) |
| **Training time** | ~1–2 hr | ~3 hr | ~40 min per run |

## Results

### Main Comparison

| Method | Accuracy | Sensitivity | Specificity | Mean IoU | IoU@0.5 | Train Reward |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **A: ResNet-50** | 0.8605 | 0.8577 | 0.8667 | — | — | — |
| **B: DenseNet-121** | 0.8534 | 0.8608 | 0.8368 | — | — | — |
| **C: CLIP Zero-shot** | 0.6629 | 0.5896 | 0.8263 | — | — | — |
| **C: CLIP Linear Probe** | 0.8664 | 0.8381 | 0.9298 | — | — | — |
| **D: Qwen2-VL SFT** | 0.9040 | 0.8870 | 0.9500 | 0.193 | 0.173 | — |
| **E: Outcome GRPO** | **0.9310** | **0.9250** | 0.9380 | **0.293** | **0.500** | 0.493 |
| **F: Process GRPO (TARA)** | 0.9200 | 0.9080 | 0.9380 | 0.273 | 0.308 | **0.725** |

### GRPO Reward Convergence

| Metric | Outcome GRPO (E) | Process GRPO (F) | Change |
|---|:---:|:---:|:---:|
| **Mean Reward** | 0.493 | **0.725** | +47% |
| **Reward Std** | 0.161 | **0.146** | −9.3% |
| **Samples > 0.6** | 11/40 | **29/40** | +163% |
| **Min Reward** | 0.350 | **0.367** | Higher floor |
| **Max Reward** | 1.000 | 0.933 | Process cap |

### Progressive Improvement (SFT → Outcome GRPO → Process GRPO)

| Metric | D: SFT | E: Outcome GRPO | F: Process GRPO |
|---|:---:|:---:|:---:|
| **Accuracy** | 0.904 | **0.931** | 0.920 |
| **Sensitivity** | 0.887 | **0.925** | 0.908 |
| **Specificity** | **0.950** | 0.938 | 0.938 |
| **Mean IoU** | 0.193 | **0.293** | 0.273 |
| **IoU@0.5** | 0.173 | **0.500** | 0.308 |
| **Structured Rate** | 1.00 | 1.00 | 1.00 |
| **Train Reward** | — | 0.493 | **0.725** |

## Interpretability and Structured Reasoning

The TARA cognitive loop provides step-by-step interpretability that purely discriminative models cannot offer:

- **`<think>`** grounds the global assessment in anatomical language, identifying the suspected lung zone before any decision is made.
- **`<act>`** produces explicit bounding-box coordinates that can be verified against ground-truth lesion locations using IoU — creating a spatially auditable reasoning step.
- **`<rethink>`** refines the initial hypothesis using clinical-specific vocabulary (cavitation, consolidation, upper-lobe opacity, tree-in-bud), verified by the clinical reward component.
- **`<answer>`** provides a final unambiguous Yes/No decision with a reasoning chain that a clinician can follow.

All three VLM variants (D, E, F) achieve 100% TARA format compliance throughout training and evaluation, confirming that the tag structure learned during SFT is stable under subsequent RL optimization.

## Limitations

- VLM evaluation uses a balanced subset of 160–200 samples rather than the full test set, due to GPU time constraints on Modal.
- TARA VQA pairs are generated from structured templates rather than radiologist-authored descriptions, which may introduce annotation artifacts on held-out manually written questions.
- GRPO training was limited to 100–200 steps; longer training would likely further improve all RL-based models.
- Cross-dataset generalization to Shenzhen and Montgomery datasets was not completed in this phase.

## Future Work

- Run Process-Reward GRPO for longer (500+ steps) with the balanced 1:1 training distribution to test whether it eventually surpasses Outcome GRPO on accuracy as well as training dynamics.
- Evaluate cross-dataset generalization to Shenzhen (662 CXRs) and Montgomery (138 CXRs) datasets.
- Develop individualized caption generation per MRI slice rather than template-based VQA pairs.
- Extend to the full four-step iterative TARA architecture with intermediate visual cropping between Think and Rethink rounds.
- Ablate reward component weights to identify the optimal budget allocation for the process reward function.

## References

The detailed list of references is available in the associated research paper (`report/main.tex`) and the SOA Survey (`SOA Survey.pdf`).

Key references include:
- Liu et al. *Rethinking Computer-Aided Tuberculosis Diagnosis.* CVPR 2020. (**TBX11K dataset**)
- Pan et al. *MedVLM-R1: Incentivizing Medical Reasoning Capability of VLMs via Reinforcement Learning.* MICCAI 2025.
- Chen et al. *ViTAR: Think Twice to See More — Iterative Visual Reasoning in Medical VLMs.* arXiv 2025.
- Xu et al. *MedGround-R1: Spatial-Semantic Rewarded GRPO for Medical Image Grounding.* MICCAI 2025.
- Fan et al. *ChestX-Reasoner: Advancing Radiology Foundation Models with Step-by-Step Verification.* arXiv 2025.
- Guo et al. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL.* arXiv 2025.

## Citation

Please cite this repository as follows if you use it in your research:

```
Ayub, A., & Rehman, F. (2025). TB-ViTAR: Iterative Spatially-Grounded Reasoning
with Process Rewards for Tuberculosis Diagnosis in Chest X-rays [Source code].
CS437/CS5317/EE414/EE513 — Deep Learning, LUMS Spring 2026.
```

---

For additional details, refer to the final paper in `Deliverables` or the SOA Survey in `SOA Survey.pdf`. Contact the authors via GitHub issues for any questions.
