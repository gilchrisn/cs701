# CS701 Team 1: SAR classification and detection

This repository holds the code and checkpoints behind our CS701 final report (SMU, AY2026-27 Term 1). Our model is the
Ultralytics YOLO26l-P2 detector trained without mosaic and scale jitter, with boxes flipped by the dataset's own flip convention and with
the KL box loss. The image class is the class whose boxes have the largest total score. It scored a Δm of 54.24 on the validation
leaderboard (5th of 9 teams) and 58.26 on the test leaderboard (6th).

All "held-out" numbers below are COCO mAP on our held-out split: 15% of the training images of each class (1,408 images, seed 0),
counting detections with score ≥ 0.05.

## Install

```
pip install -r requirements.txt
```
We used Python 3.12, PyTorch 2.9.0 and Ultralytics 8.4.182. Training the final model took 10.7 h on one RTX A5000 (24 GB).
The COCO weights `yolo26l.pt` download on first use.

## Data

Download the [course data](https://huggingface.co/datasets/doem1997/cs701-sar-course-data) into `data/`, so that
`data/train/`, `data/val/` and `data/test/` exist. Then build the held-out split and the YOLO labels (all commands run from the
repository root):

```
python tools/make_holdout.py --src data/train --out data/holdout15_seed0
python tools/make_yolo_data.py data/holdout15_seed0 data/yolo_holdout15_seed0
sed 's#^train: images/train$#train: [images/train, images/val]#' data/yolo_holdout15_seed0/data.yaml > data/yolo_holdout15_seed0/data_full.yaml
```
The last line writes a second config that trains on all 9,392 images (used for the submitted model).

## Checkpoints

| file | trained on | use |
|---|---|---|
| `weights/yolo26l-p2_final_alldata.pt` | all 9,392 training images | the submitted model |
| `weights/yolo26l-p2_final_85pct.pt` | the 85% split | the same model scored on the held-out split |

Both load in plain Ultralytics: `YOLO('weights/yolo26l-p2_final_85pct.pt')`.

## 1. Score the final model on the held-out split

```
python tools/score_yolo_holdout.py weights/yolo26l-p2_final_85pct.pt --data data/holdout15_seed0 --out runs/final85_eval
python analysis/chain_metrics.py --data data --runs runs --only epochs_144_final_recipe_85pct
```
Expected: held-out mAP 0.5375 (we measured 0.5374 from the shipped checkpoint; the difference is GPU nondeterminism).

## 2. Write a submission

```
python tools/predict_submission_yolo.py weights/yolo26l-p2_final_alldata.pt --images-json data/val/images.json --root data/val --out submission_val.zip --conf 0.00001
```
This writes the leaderboard zip (`classification.csv` and `detection.json`). On the validation leaderboard it scored a Δm of
54.24 (macro-F1 92.87, mAP 54.60).

## 3. Reproduce every step of the report

Each command trains one step of Fig. 3 in the report on the 85% split. The variant names are explained at the top of
`yolo/train_variants.py`.

```
# YOLO26l-P2 steps (96 epochs unless stated)
python yolo/train_variants.py ultralytics  yolo_mosaic_on  data/yolo_holdout15_seed0/data.yaml 16 runs 96
python yolo/train_variants.py none         yolo_mosaic_off data/yolo_holdout15_seed0/data.yaml 16 runs 96
python yolo/train_variants.py kl           yolo_kl         data/yolo_holdout15_seed0/data.yaml 16 runs 96
python yolo/train_variants.py kl+flipconv  yolo_kl_flip    data/yolo_holdout15_seed0/data.yaml 16 runs 96
python yolo/train_variants.py kl+flipconv  final85         data/yolo_holdout15_seed0/data.yaml 16 runs 144

# after Phase 1 (not submitted): 24 or 48 more epochs from the final model, with brightness and shift jitter off
INIT_FROM=weights/yolo26l-p2_final_85pct.pt python yolo/train_variants.py kl+flipconv+exact ft48 data/yolo_holdout15_seed0/data.yaml 16 runs 48

# course model and Cascade R-CNN (course code in train.py and sarbench/, with our --cascade option)
python train.py --data data/holdout15_seed0 --out runs/course24 --backbone vit --init pretrained --adapt lora --epochs 24 --batch-size 16 --accum 4 --workers 4 --seed 0
python train.py --data data/holdout15_seed0 --out runs/cascade24 --backbone vit --init pretrained --adapt lora --cascade --epochs 24 --batch-size 16 --accum 4 --workers 4 --seed 0
python train.py --data data/holdout15_seed0 --out runs/cascade48 --backbone vit --init pretrained --adapt lora --cascade --epochs 48 --batch-size 16 --accum 4 --workers 4 --seed 0
python train.py --data data/holdout15_seed0 --out runs/cascade48_ft24 --backbone vit --init pretrained --adapt lora --cascade --epochs 24 --lr 3e-5 --init-from runs/cascade48/model.pt --batch-size 16 --accum 4 --workers 4 --seed 0
```

To score a YOLO step trained with `kl`, first remove the extra KL outputs from the checkpoint, then predict on the held-out split:
```
python yolo/kl_mod.py strip runs/yolo_kl/weights/last.pt
python tools/score_yolo_holdout.py runs/yolo_kl/weights/last_stripped.pt --data data/holdout15_seed0 --out runs/yolo_kl_eval
```
(For steps without `kl`, score `last.pt` directly.) The course and Cascade runs write their predictions themselves. Then
`python analysis/chain_metrics.py --data data --runs runs` scores every step by object size and by overlap group and writes
`analysis/results/chain_metrics.json`. Our stored results, held-out mAP:

| step | all | no match in train | overlapping crop | near-duplicate |
|---|---|---|---|---|
| course model | 0.232 | 0.150 | 0.266 | 0.264 |
| Cascade R-CNN, 24 epochs | 0.262 | 0.163 | 0.306 | 0.305 |
| Cascade R-CNN, 48 + 24 epochs | 0.336 | 0.170 | 0.387 | 0.401 |
| YOLO26l-P2 | 0.422 | 0.290 | 0.500 | 0.467 |
| mosaic and scale jitter off | 0.472 | 0.269 | 0.557 | 0.596 |
| + KL box loss | 0.479 | 0.273 | 0.571 | 0.591 |
| + flip convention | 0.516 | 0.313 | 0.608 | 0.652 |
| + 144 epochs (submitted) | 0.538 | 0.301 | 0.627 | 0.708 |
| brightness and shift off, 24 epochs | 0.559 | 0.307 | 0.643 | 0.758 |
| brightness and shift off, 48 epochs | 0.564 | 0.310 | 0.644 | 0.771 |

Retraining gives slightly different numbers; two runs that differ only in the seed differ by about 0.0055.

## 4. Overlap groups

`tools/holdout15_seed0_scene_groups.json` stores the overlap group of every held-out image; the scorers above read it.
`analysis/overlap_groups.py` finds the groups of the validation and test images (CPU only, about 20 minutes):

```
python analysis/overlap_groups.py --data data --out analysis/results/test_groups.json --procs 20
```
- near-duplicate: the edge map of the image, after both are shrunk, has cosine similarity ≥ 0.8 with that of a training image,
  under identity, horizontal flip or vertical flip;
- overlapping crop: otherwise, at least 30 corner points (ORB) match a training image under one shift fitted with RANSAC, spread
  over at least 8 of 64 grid cells;
- no match in train: all other images.

Expected (stored in `analysis/results/test_groups.txt`): validation 581 / 353 / 492 and test 863 / 473 / 729 images with no
match / overlapping crop / near-duplicate. The held-out groups were made earlier with the same rules by a script we did not keep;
on the validation set it agrees with this script on the no-match group (587 against 581 images) but divides the rest differently
between crops and near-duplicates.

## 5. The dataset's flip convention

```
python analysis/flip_convention.py --data data/holdout15_seed0
```
For each held-out image aligned to a training copy (`analysis/results/copies_pairs.csv`), this maps the copy's boxes onto the
held-out image with the exact flip and measures how far each box edge is from the held-out box. Expected
(`analysis/results/flip_convention.txt`): in horizontally flipped copies, 82% of left and 81% of right edges are exactly 1 px off;
in unflipped copies the edges agree. So the dataset flips boxes with x' = W - 1 - x, while Ultralytics uses x' = W - x.
`copies_pairs.csv` was made by an earlier alignment step (each candidate copy tried under the 8 flips and rotations and a shift);
its inputs were not kept, so the pairs are stored here as data.

## 6. Every figure and number of the report

Run from the repository root, after the held-out split of "Data" exists. Each figure script writes `figures/out/<name>.pdf` and `.png`
(`--out` changes the folder; fig2 and fig3 need no `--data`).

| report | command | reads |
|---|---|---|
| Fig. 1, a held-out image, its flipped training copy, and overlap shares by split | `python figures/fig1_near_duplicates.py` | held-out split, `analysis/results/copies_pairs.csv`, `analysis/results/test_groups.txt`, `tools/holdout15_seed0_scene_groups.json` |
| Fig. 2, pipeline diagram | `python figures/fig2_pipeline.py` | nothing |
| Fig. 3, held-out mAP after each step | `python figures/fig3_journey.py` | `analysis/results/chain_metrics.json` (made by Section 3's `chain_metrics.py` command) |
| Fig. 4, eleven changes against their control | `python figures/fig4_failed_changes.py` | `figures/data/failed_changes.csv`, `figures/data/seed_pairs.csv` |
| Fig. 5, one box edge of a flipped copy | `python figures/fig5_flip_pixels.py` | held-out split, `analysis/results/copies_pairs.csv` |
| Fig. 6, a crop, a rotated near-duplicate, overlap and AP per class | `python figures/fig6_copies.py` | held-out split, `analysis/results/copies_pairs.csv`, `analysis/results/chain_metrics.json` |
| Table of held-out mAP per step (Sec. 3) | `python analysis/chain_metrics.py --data data --runs runs` | the predictions of Section 3 |
| 1 px flip offsets (Sec. 5) | `python analysis/flip_convention.py --data data/holdout15_seed0` | held-out split, `copies_pairs.csv` |

Fig. 4 compares runs we trained in October on the 85% split; the repository does not contain those runs. `figures/data/failed_changes.csv`
stores the held-out mAP of each change and of its control (columns `mAP_change`, `mAP_control`; the last column names the file in our run store each
number came from, or the note in our log for the three that were only logged). The figure plots the difference. To redo
one row, train the change with the recipe of the step it modifies and score it as in Section 3.
`figures/data/seed_pairs.csv` has the same for the eight pairs of runs that differ only in the seed.

The figures use DejaVu Sans. Matplotlib 3.10.8 rendered Figs. 1, 2, 3 and 5 of the report and 3.11.2 rendered Figs. 4 and 6; with the
version we used for each, the scripts reproduce the report's PNGs pixel for pixel. With the other version the text is a few percent wider or narrower.

Numbers in the text of Sec. 5:

```
python analysis/copy_stats.py --data data/holdout15_seed0
python analysis/copy_labels.py --data data/holdout15_seed0 --preds runs/final85_eval/predictions_val.json
python analysis/seed_variation.py
```
- `copy_stats.py` (writes `analysis/results/copy_stats.txt`): 795 held-out copies, 682 aligned (correlation >= 0.6, overlap >= 25%); median
  correlation 0.96 for near-duplicates (375 aligned); 334 of them whole copies (no shift, overlap >= 0.95), 276 unrotated and 58 rotated by 180
  degrees; crops: 307 aligned, 224 shifted (median larger shift component 96 px, median overlap 0.69), 83 whole copies; the share of each overlap
  group per class. Median correlation of the crops is 0.998, which the report writes as 0.99.
- `copy_labels.py` (writes `analysis/results/copy_labels.txt`): per class and group, the median IoU between a held-out box and the same box
  mapped from the training copy (1.000 for near-duplicates in most classes, 0.620 for wind turbines in crops). Then, on the 276
  unrotated, unshifted near-duplicates, the training copy's boxes used as predictions score 0.9075 mAP. The model on the same images
  scores 0.6975 with predictions made at `--conf 0.00001` (`score_yolo_holdout.py ... --conf 0.00001`; the default `--conf 0.001` gives 0.6971), and 0.7674
  for the exact-copy fine-tune (`--preds-ft`, 24 epochs). Using the copy's boxes is not allowed as a method; this is only an upper bound for
  memorising the copies better.
- `seed_variation.py`: the eight pairs and their root mean square, 0.55 points of held-out mAP.

We checked these numbers against the stored predictions of our October-November runs; the results were identical to the values above.
We did not re-run them from the shipped checkpoints.

## Layout

```
weights/    the two final checkpoints
yolo/       train_variants.py (training recipe), kl_mod.py (KL box loss), flipconv_mod.py (dataset flip convention)
tools/      make_holdout.py, make_yolo_data.py, score_yolo_holdout.py, predict_submission_yolo.py,
            analyze.py (AP code used by chain_metrics.py), holdout15_seed0_scene_groups.json
analysis/   chain_metrics.py, overlap_groups.py, flip_convention.py, copy_stats.py, copy_labels.py, seed_variation.py, results/ (stored outputs)
figures/    fig1 to fig6 scripts for the report's figures, _style.py (shared look), data/ (small inputs that cannot be recomputed here)
train.py, sarbench/, make_submission.py   course model and Cascade R-CNN; make_submission.py writes the leaderboard zip
```
