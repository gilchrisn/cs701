# CS701 SAR Multi-Task Benchmark

Reference code for the CS701 team assignment. One ViT-B/16 backbone, two tasks on SAR images: **classify** the
image (9 classes) and **detect** its objects.

![One SAR image, one shared backbone, two heads: a class and the boxes](docs/task_pipeline.png)

## Contents

1. [SAR Task Introduction](#1-sar-task-introduction)
   - [1.1 SAR Basics](#11-sar-basics)
   - [1.2 Task](#12-task)
   - [1.3 Metric](#13-metric)
   - [1.4 Small Objects](#14-small-objects)
2. [Quick Start](#2-quick-start)
   - [2.1 Install](#21-install)
   - [2.2 Train Baseline](#22-train-baseline)
   - [2.3 Use Course Data](#23-use-course-data)
   - [2.4 Make Submission](#24-make-submission)
3. [Baseline Results (for reference)](#3-baseline-results-for-reference)
4. [Code Structure](#4-code-structure)
5. [Links](#5-links)
6. [YOLO26l-P2 (our best model)](#6-yolo26l-p2-our-best-model)

## 1. SAR Task Introduction

### 1.1 SAR Basics

![A radar sends microwave pulses sideways, records the echoes, and combines them along its flight path into one long virtual antenna](docs/sar_explainer.gif)

- **Active radar**: satellite or aircraft sends microwave pulses, records echoes. Works day, night, through clouds.
  Looks sideways.
- **Synthetic aperture**: echoes along the flight path combined into one long virtual antenna, so fine resolution.
- **Pixel = echo strength**: 8-bit grey, no colour. Metal and corners bright, calm water dark, grainy speckle.

<details>
<summary>What SAR images look like</summary>

![Speckle, bright metal and corners, dark smooth surfaces, bright and dark slopes](docs/sar_effects.png)

</details>

### 1.2 Task

- Input: one SAR image. Output: its class + a box, class and score for each object.
- Each image: one class label, 1 or more boxes, all of that class.
- 9 classes: aircraft, airport, bridge, car, harbor, oil tank, playground, ship, wind turbine.
- Data, formats, release dates: [dataset card](https://huggingface.co/datasets/doem1997/cs701-sar-course-data).
  Rules: [Codabench](https://www.codabench.org/competitions/18271).
- Limits: one model; at most 130 M parameters in total (frozen ones count; this code: 106-107 M) and at most 40 M
  trainable (heads included: LoRA 21.0 M; full fine-tuning and from scratch, 106 M, are over); each image in once at
  ≤ 512 × 512 px; no ensembles, no test-time augmentation. Your report states parameters (total, trainable) and
  compute; full list on Codabench (Terms).

### 1.3 Metric

- Classification: macro-F1. Detection: COCO mAP.
- Ranking: **Δm**, mean relative change of macro-F1 and mAP vs a reference model (ViT + LoRA, the example command
  in 2.2; mean of 3 seeds):
  `Δm = 100% × ½ [(F1 − F1_ref) / F1_ref + (mAP − mAP_ref) / mAP_ref]`
- Δm = 0: as good as the reference. Δm > 0: better. ViT full fine-tuning: about +13 (over the trainable budget: an
  upper bound).

### 1.4 Small Objects

Half of all boxes are smaller than one 16 × 16 ViT patch (at 512 × 512 input).

![A close-up with the 16 px patch grid: wind turbines of 6-10 px and the 15.9 px median box are smaller than one patch](docs/patch_grid.png)

## 2. Quick Start

### 2.1 Install

```bash
conda create -n cs701bench python=3.12 -y && conda activate cs701bench
pip install -r requirements.txt   # CUDA 12.8 builds of torch 2.9.0 / torchvision 0.24.0
```

### 2.2 Train Baseline

```bash
python train.py --backbone vit --init pretrained --adapt lora   # the Δm reference; one run, about 50 min
bash run_all.sh                                                  # all eight runs, about 7 h
```

- Options: `--backbone vit|terramind`, `--init pretrained|scratch`, `--adapt full|lora|moelora`
  (`python train.py -h`).
- `--adapt full` and `--init scratch` train 106 M parameters: over the 40 M budget, for reference only.
- Pretrained weights download from Hugging Face on first use (ViT 0.4 GB, TerraMind 1.5 GB).
- Times for one RTX PRO 6000 Blackwell GPU. Memory: [docs/DETAILS.md](docs/DETAILS.md#time-and-memory).

### 2.3 Use Course Data

TA Zichen ran this code on fully labelled data in another folder layout. For the
[course data](https://huggingface.co/datasets/doem1997/cs701-sar-course-data), adapt `sarbench/data.py` and `train.py`:

- train on `train/labels.csv` + `train/instances.json`
- hold out part of train for local validation (val and test have no labels)
- predict on the images in `val/images.json` and `test/images.json`

### 2.4 Make Submission

`make_submission.py` has `write_submission` (the function in the briefing video): maps your boxes back to
original image pixels, writes `submission.zip` for Codabench.

## 3. Baseline Results (for reference)

Test split, %, mean ± std over 3 seeds. Δm reference: ViT + LoRA, mean of its 3 seeds (93.99 / 25.41), so its Δm
is 0. Full fine-tuning and from scratch train 106 M parameters with the heads: over the 40 M budget, shown for
reference (full fine-tuning: the upper bound).

| backbone | adaptation | trained backbone params | macro-F1 | mAP | Δm |
|---|---|---:|---:|---:|---:|
| ViT (ImageNet-21k) | full fine-tuning | 86.0 M | 95.8 ± 0.3 | 31.8 ± 0.2 | +13.5 ± 0.4 |
| ViT (ImageNet-21k) | LoRA | 0.88 M | 94.0 ± 0.2 | 25.4 ± 0.5 | 0.0 ± 0.9 |
| ViT (ImageNet-21k) | MoE-LoRA | 1.03 M | 93.4 ± 0.5 | 26.8 ± 0.3 | +2.5 ± 0.8 |
| TerraMind (Earth observation) | full fine-tuning | 85.3 M | 95.6 ± 0.2 | 30.8 ± 0.4 | +11.5 ± 0.7 |
| TerraMind (Earth observation) | LoRA | 0.88 M | 92.6 ± 0.4 | 26.5 ± 0.4 | +1.4 ± 1.0 |
| TerraMind (Earth observation) | MoE-LoRA | 1.03 M | 92.4 ± 0.1 | 27.8 ± 0.3 | +3.8 ± 0.7 |
| ViT, from scratch | full training | 86.0 M | 82.8 ± 0.7 | 17.0 ± 0.3 | −22.5 ± 0.9 |
| TerraMind, from scratch | full training | 85.3 M | 86.1 ± 0.2 | 17.8 ± 0.5 | −19.3 ± 0.8 |

<details>
<summary>Val results</summary>

Val split, %, mean ± std over 3 seeds. Δm reference: the same configuration on val (93.31 / 26.13).

| backbone | adaptation | trained backbone params | macro-F1 | mAP | Δm |
|---|---|---:|---:|---:|---:|
| ViT (ImageNet-21k) | full fine-tuning | 86.0 M | 94.4 ± 0.3 | 32.0 ± 0.4 | +11.8 ± 0.7 |
| ViT (ImageNet-21k) | LoRA | 0.88 M | 93.3 ± 0.4 | 26.1 ± 0.4 | 0.0 ± 0.9 |
| ViT (ImageNet-21k) | MoE-LoRA | 1.03 M | 92.2 ± 0.8 | 27.1 ± 0.2 | +1.3 ± 0.9 |
| TerraMind (Earth observation) | full fine-tuning | 85.3 M | 94.2 ± 0.3 | 31.5 ± 0.1 | +10.7 ± 0.3 |
| TerraMind (Earth observation) | LoRA | 0.88 M | 92.2 ± 0.4 | 27.0 ± 0.5 | +1.0 ± 1.3 |
| TerraMind (Earth observation) | MoE-LoRA | 1.03 M | 91.9 ± 0.4 | 28.8 ± 0.1 | +4.4 ± 0.3 |
| ViT, from scratch | full training | 86.0 M | 83.5 ± 0.9 | 18.2 ± 0.5 | −20.3 ± 1.3 |
| TerraMind, from scratch | full training | 85.3 M | 86.0 ± 0.9 | 19.2 ± 0.4 | −17.3 ± 1.2 |

</details>

Accuracy and AP50: [docs/DETAILS.md](docs/DETAILS.md#results-in-full).

## 4. Code Structure

```
train.py              entry: train + evaluate one configuration
train_yolo.py         entry: train or fine-tune YOLO26l-P2 (section 6)
final_sprint.sh       entry: the final-sprint experiments and the submission model (section 7)
run_all.sh            the eight reference runs (calls train.py)
make_submission.py    write_submission(): predictions -> submission.zip
sarbench/             the code behind train.py
├── data.py           images + boxes at 512 × 512, flips; boxes back to original pixels
├── backbones.py      ViT (ImageNet-21k), TerraMind-1.0-base
├── adapters.py       LoRA, MoE-LoRA on a frozen backbone
├── yolo_kl.py        KL box loss for YOLO (train_yolo.py --kl)
├── model.py          backbone + class head + ViTDet feature pyramid + Faster R-CNN
└── metrics.py        accuracy, macro-F1, COCO box AP
tools/                hold-out split, YOLO data and submission, analysis of finished runs
tests/                checks of sarbench/ (python -m pytest tests)
docs/DETAILS.md       model, training protocol, design notes, time and memory
```

More: [docs/DETAILS.md](docs/DETAILS.md).

## 5. Links

- Data: https://huggingface.co/datasets/doem1997/cs701-sar-course-data
- Leaderboard: https://web-production-c241e.up.railway.app (Codabench, https://www.codabench.org/competitions/18271,
  is down since 2026-10-25)
- Questions: TA Zichen, zichen.tian.2023@phdcs.smu.edu.sg

## 6. YOLO26l-P2 (our best model)

One detector, no classification head: the image label is the class with the largest summed box score in that image.

- Model: Ultralytics YOLO26-large with an extra P2 output (stride 4, for objects of a few pixels), `yolo26l-p2.yaml`,
  initialised from the COCO-pretrained `yolo26l.pt`. 25.7 M parameters, all trained.
- Training: 96 epochs, 512 × 512, batch 16, seed 0, Ultralytics 8.4.182 defaults otherwise; 3.3 h on one RTX A5000.
- Held-out split (15 % of every class, seed 0; 1,408 images), score ≥ 0.001: mAP 42.9, AP50 69.4, macro-F1 93.3.
  Bridge and playground stay low (AP 4.7 and 7.2).

```bash
pip install -r requirements.txt
python tools/make_holdout.py --src data/train --out data/holdout15_seed0        # 85 % train, 15 % held out
python tools/make_yolo_data.py data/holdout15_seed0 data/yolo_holdout15_seed0    # YOLO labels + data.yaml
python train_yolo.py --data data/yolo_holdout15_seed0/data.yaml --model yolo26l-p2.yaml --init yolo26l.pt \
    --out runs/yolo26lp2_96ep
python tools/predict_submission_yolo.py runs/yolo26lp2_96ep/weights/last.pt --images-json data/val/images.json \
    --root data/val --out submission_val.zip
```

- The numbers above are from `last.pt` (epoch 96). Weights (52 MB) are not in git.
- Fine-tune them: `--model yolo26l-p2_96ep.pt`, other Ultralytics arguments as `key=value` at the end
  (`python train_yolo.py -h`). `lr0` needs a named optimizer (`optimizer=AdamW lr0=...`); with the default
  `optimizer=auto` Ultralytics ignores it.
- They were trained on the 85 % split above. Fine-tuning on a different split puts held-out images into training,
  so held-out scores are no longer held-out scores.

## 7. Final sprint

**What the held-out score measures.** About 57 % of held-out images, and about as many official val images, show a scene that
is also in the training images: shifted, flipped or nearly identical. `tools/holdout15_seed0_scene_groups.json` splits the
held-out set into new scenes (G0, 613 images), shifted copies (G1, 359) and near-identical or mirrored copies (G2, 436).
Every earlier run re-scored on these groups shows two patterns:

- Copied scenes reward memorising the training views. Longer training and fully trained weights help. Augmentation that changes
  the view (mosaic, scale jitter, rotations) and frozen backbones hurt.
- New scenes improved only with finer feature maps (the stride-4 P2 output) or multi-scale views. No loss, label-assignment,
  head or post-processing change moved them.

**Experiments.** `bash final_sprint.sh <name>` trains on the 85 % split for 48 epochs, then scores with
`tools/score_yolo_holdout.py` (full held-out, G0 / G1 / G2, and val-weighted = 0.412 G0 + 0.325 G1 + 0.264 G2, the shares of the
official val set). Memory and time are for one RTX A5000 used alone; the 768 px figures are estimates. `yolo26l.pt` (COCO
weights) downloads on first use; if that times out, get it from
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt into this folder.

| name | change from our best recipe | question | memory / time |
|---|---|---|---|
| `control48` | none | reference for every row below | ~11 GB / ~1.8 h |
| `vflip48` | vertical flips, `flipud=0.5` | do the views of val's mirrored copies help? | ~11 GB / ~1.8 h |
| `img768_b16` | 768 px input | do bigger objects help new scenes without hurting copies? | ~13 GB / ~4 h |
| `img768_b8` | 768 px input, batch 8 | the same at batch 8: does batch size matter? | ~7 GB / ~4 h |
| `control48_seed1`, `img768_b16_seed1` | seed 1 | seed noise for YOLO, never measured so far | as above |

A change passes if its val-weighted mAP beats `control48` by at least 0.0068 and its G0 mAP is not lower by more than 0.0049
(the RMS differences between 8 pairs of earlier runs that differ only in seed).

**Submission model.** `bash final_sprint.sh final [winning change]`, e.g. `bash final_sprint.sh final --imgsz 768`: all 9,392
labelled images, 96 epochs, KL box loss, then the stripped weights predict the val set at score >= 0.00001 into
`runs/final/submission_val.zip`. About 4 h at 512 px on one RTX A5000 (about 9.5 h at 768 px, estimated).
