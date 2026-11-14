# CS701 Team 1: SAR classification and detection

Our solution to the CS701 SAR course project (SMU, AY2026-27 Term 1). One SAR image in, its class (9 classes) and its
boxes (class, score) out; ranked by Δm, the mean relative change of macro-F1 and COCO mAP against the course reference
model (ViT-B/16 + LoRA). The submitted model is Ultralytics YOLO26l-P2 (8.4.182) with a KL box loss and flips that follow the
dataset's own label rule. It scored Δm 54.24 on the val leaderboard (5th of 9) and +58.26 on test (6th). This repo holds
the code that reproduces the numbers in our report, plus the two final checkpoints. Course material and everything not used
by the report were removed from the tree; the git history keeps them.

Terms used below. **Held-out split**: 15 % of every class of the 9,392 labelled images (1,408 images, seed 0), never trained on
by the "85 %" model; all held-out numbers are COCO mAP@[.5:.95] on it. **Score cut-off**: detections below this score are
dropped before scoring; the report numbers use 0.05 (the course models only stored boxes at or above 0.05), the YOLO scorer
prints at 0.001. **G0 / G1 / G2** (overlap groups): held-out images whose scene is new (G0, 613 images), a shifted
or overlapping crop of a training image (G1, 359), or a near-duplicate or mirrored copy of one (G2, 436).

## Install

```
pip install -r requirements.txt     # torch 2.9.0, ultralytics 8.4.182, pycocotools, timm (course model)
```
Checked with Python 3.12, torch 2.9.0, one RTX 4070 laptop GPU (8 GB) for scoring; training the final model took 10.7 h
on one RTX A5000 (24 GB) that was also running a second job. `yolo26l.pt` (COCO weights) downloads on first use; run training from a folder that holds it.

## Data

Download the [course data](https://huggingface.co/datasets/doem1997/cs701-sar-course-data) and unpack it so that
`data/train/{images,instances.json,labels.csv}`, `data/val/{images,images.json}` and `data/test/{images,images.json}` exist.
Then build the splits (every command below is run from the repo root):

```
python tools/make_holdout.py --src data/train --out data/holdout15_seed0            # 85 % train / 15 % held out
python tools/make_yolo_data.py data/holdout15_seed0 data/yolo_holdout15_seed0       # YOLO labels + data.yaml
sed 's#^train: images/train$#train: [images/train, images/val]#' data/yolo_holdout15_seed0/data.yaml > data/yolo_holdout15_seed0/data_full.yaml   # all 9,392 images
```

## Checkpoints (weights/)

| file | trained on | use |
|---|---|---|
| `yolo26l-p2_final_alldata.pt` | all 9,392 labelled images | the submitted model |
| `yolo26l-p2_final_85pct.pt` | the 85 % split | its held-out twin: every held-out score in the report |

Both are 144-epoch checkpoints with the KL variance rows already removed, so plain Ultralytics loads them:
`YOLO('weights/yolo26l-p2_final_85pct.pt')` (checked: task detect, 25.7 M parameters, 512 px, 9 classes).

## 1. Final model: training

Recipe: `yolo26l-p2.yaml` from the COCO weights `yolo26l.pt`, mosaic and scale jitter off, horizontal and vertical flips whose
boxes follow the dataset rule x' = W - 1 - x (`yolo/flipconv_mod.py`; Ultralytics uses x' = W - x), KL box loss
(`yolo/kl_mod.py`), 144 epochs, batch 16, 512 px, seed 0.

```
python yolo/train_variants.py kl+flipconv final85 data/yolo_holdout15_seed0/data.yaml 16 runs 144        # 85 % split
python yolo/train_variants.py kl+flipconv finalall data/yolo_holdout15_seed0/data_full.yaml 16 runs 144   # all images (submitted)
python yolo/kl_mod.py strip runs/final85/weights/last.pt          # writes last_stripped.pt; always strip before predicting
```
Expected: the 85 % model reaches held-out mAP 0.5375 (table in section 3). Retraining will differ slightly (GPU nondeterminism);
the two shipped checkpoints are the runs behind the report.

## 2. Submission

`tools/predict_submission_yolo.py` predicts at 512 px with at most 100 boxes per image, drops boxes of non-positive size, and
sets the image class to the class with the largest summed box score. Output is the Codabench zip.

```
python tools/predict_submission_yolo.py weights/yolo26l-p2_final_alldata.pt --images-json data/val/images.json --root data/val --out submission_val.zip --conf 0.00001
```
Expected: val Δm 54.24 (macro-F1 92.87, mAP 54.60) on Codabench; test +58.26. Spot check on the first 64 val images: 5,758 boxes,
and all 64 image classes equal those of the stored submission.

## 3. Held-out scoring: overall, by object size, by overlap group

```
python tools/score_yolo_holdout.py weights/yolo26l-p2_final_85pct.pt --data data/holdout15_seed0 --out runs/eval85
```
Prints mAP overall and per overlap group at score cut-off 0.001: measured here 0.5434 overall, G0 0.3077, G1 0.6320, G2 0.7130,
val-weighted 0.5198 (val-weighted = 0.412 G0 + 0.325 G1 + 0.264 G2, the group shares of the official val set).

The report's numbers use cut-off 0.05, size bins, and every model of the chain; `analysis/chain_metrics.py` computes them from
each run's `predictions_val.json` and writes `analysis/results/chain_metrics.json`, which holds the stored report numbers:

```
python analysis/chain_metrics.py --data data --only epochs_144_final_recipe_85pct --predictions epochs_144_final_recipe_85pct=runs/eval85/predictions_val.json
```
Expected for the final 85 % model (stored / measured here from the shipped checkpoint, differences are GPU noise):

| | mAP | AP50 | AP75 | box side <8 / 8-16 / 16-32 / >32 px | G0 / G1 / G2 |
|---|---|---|---|---|---|
| stored | 0.5375 | 0.727 | 0.578 | 0.263 / 0.394 / 0.431 / 0.614 | 0.301 / 0.627 / 0.708 |
| measured | 0.5374 | 0.727 | 0.578 | 0.263 / 0.394 / 0.431 / 0.614 | 0.301 / 0.627 / 0.708 |

With all run folders present, `python analysis/chain_metrics.py --data data --runs runs` rebuilds every row of the chain
(course model 0.2318, Cascade 0.2624 and 0.3358, YOLO rows up to 0.5375, post-deadline fine-tune 0.5589 and 0.5644).
Run on the stored predictions it reproduced `chain_metrics.json` exactly.

## 4. Overlap groups

The held-out groups are stored in `tools/holdout15_seed0_scene_groups.json` (COCO image ids; G0 613, G1 359, G2 436) and used by
the scorer above. `analysis/overlap_groups.py` applies the same rules to the val and test images against all 9,392 labelled
images (G2: edge-image cosine >= 0.8 under identity, horizontal or vertical flip; G1: ORB + RANSAC with >= 30 inliers spread
over >= 8 of 64 cells; G0: the rest):

```
python analysis/overlap_groups.py --data data --out analysis/results/test_groups.json --procs 20     # CPU only
```
Expected (stored in `analysis/results/test_groups.txt`): val G0 581 / G1 353 / G2 492 of 1,426; test G0 863 / G1 473 / G2 729 of 2,065. A rerun here (CPU, 20 processes, about 20 min) gave exactly these counts.
The original group script (a scratch file, `groups3_mirror.py`) is lost. This rerun matches its val G0 (581 vs 587) but not its
G1 / G2 boundary (G2 492 vs 376); the cause was not traced. Held-out groups come from the original script, not from this one.

## 5. Flip-rule evidence

`analysis/flip_convention.py` compares held-out boxes with their aligned training copies. In horizontally flipped copies the
held-out box edges sit 1 px left of the exactly mirrored box (82 % of 90 boxes, left and right edge), in 180-degree copies 1 px
up and left (86 % of 526), in unflipped copies 0 px (83 % of 1,812): the dataset flips with x' = W - 1 - x.
`analysis/copies.py` finds the copies (it needs the candidate lists of an earlier same-scene search, which are lost, so its stored
output `analysis/results/copies_pairs.csv`, `copies_boxes.csv`, `copies_summary.txt` is used instead: 682 of 795 G1/G2 held-out
images align to a training copy).

```
BUDGET_PRED=runs/eval85/predictions_val.json DATA_ROOT=data python analysis/flip_convention.py     # writes analysis/results/flip_convention.txt
```
Expected offsets as above (they do not depend on the predictions). The last line, the mAP an exact flip rule could gain on these
copies, was +0.0247 with the 96-epoch mosaic-off model's predictions (stored in `analysis/results/flip_convention.txt`); it
depends on which predictions file is given. `analysis/budget.py`, `loose_profile.py` and `loose_tests.py` are helpers it imports.

## 6. Post-deadline fine-tune (not submitted)

From the 144-epoch checkpoint, 48 more epochs with HSV jitter and translation off, flips on, MuSGD lr 0.002, 1 warm-up epoch,
batch 16, seed 0. `INIT_FROM` must be the stripped checkpoint.

```
INIT_FROM=weights/yolo26l-p2_final_85pct.pt python yolo/train_variants.py kl+flipconv+exact ft48 data/yolo_holdout15_seed0/data.yaml 16 runs 48
python yolo/kl_mod.py strip runs/ft48/weights/last.pt
python tools/score_yolo_holdout.py runs/ft48/weights/last_stripped.pt --data data/holdout15_seed0 --out runs/ft48_eval --conf 0.00001
```
Expected held-out mAP (cut-off 0.05): 0.5644 after 48 epochs, 0.5589 after 24, against 0.5375 before; the gain is in G2
(0.708 to 0.771). This model was trained after the Phase 1 deadline and was not submitted.

## 7. Course model and Cascade R-CNN

`train.py` + `sarbench/` are the course reference code (ViT-B/16 backbone, class head, ViTDet pyramid, torchvision Faster R-CNN
heads) with our `--cascade` option (one extra RoI stage at IoU 0.6). They write `predictions_val.json` (score >= 0.05) for
`analysis/chain_metrics.py`. Held-out runs, seed 0:

```
python train.py --data data/holdout15_seed0 --out runs/2026-10-08_ref24_s0 --backbone vit --init pretrained --adapt lora --epochs 24 --batch-size 16 --accum 4 --workers 4 --seed 0
python train.py --data data/holdout15_seed0 --out runs/2026-10-09_cascade24_s0 --backbone vit --init pretrained --adapt lora --cascade --epochs 24 --batch-size 16 --accum 4 --workers 4 --seed 0
python train.py --data data/holdout15_seed0 --out runs/2026-10-18_cascade48_s0 --backbone vit --init pretrained --adapt lora --cascade --epochs 48 --batch-size 16 --accum 4 --workers 4 --seed 0
python train.py --data data/holdout15_seed0 --out runs/2026-10-19_ft24_c48_s0_laptop --backbone vit --init pretrained --adapt lora --cascade --batch-size 16 --accum 4 --workers 4 --lr 3e-5 --seed 0 --init-from runs/2026-10-18_cascade48_s0/model.pt --epochs 24
```
Expected held-out mAP (cut-off 0.05): course model 0.2318, Cascade 24 epochs 0.2624, Cascade 48 + 24 epochs 0.3358. The course
code also builds other backbones and adapters (`--backbone`, `--adapt`); the report does not use them.
`tools/predict_submission.py` writes a Codabench zip from such a run; `tools/infer.py` re-predicts a finished run.

## Layout

```
weights/        the two final checkpoints
yolo/           train_variants.py (recipe), kl_mod.py (KL box loss), flipconv_mod.py (dataset flip rule)
tools/          make_holdout, make_yolo_data, predict_submission_yolo, score_yolo_holdout, analyze (AP code used by chain_metrics),
                predict_submission / infer (course model), holdout15_seed0_scene_groups.json
analysis/       chain_metrics, overlap_groups, flip_convention, copies, budget (+ results/ with the stored outputs)
train.py, sarbench/, make_submission.py   course model and Cascade R-CNN
tests/          checks of sarbench/ (python -m pytest tests; 20 pass without data; the 11 that read the course dataset folder dataset/SARFact-Course-20K fail here, which we do not have)
```
