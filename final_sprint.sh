#!/usr/bin/env bash
# Final-sprint experiments (README section 7), one function each: bash final_sprint.sh <experiment> [extra train_yolo.py arguments]
# Paths: DATA (default data), RUNS (default runs), GPU (default 0). Needs data/holdout15_seed0 and data/yolo_holdout15_seed0 (section 6).
set -euo pipefail
cd "$(dirname "$0")"
DATA=${DATA:-data} RUNS=${RUNS:-runs} GPU=${GPU:-0}
Y=$DATA/yolo_holdout15_seed0/data.yaml
# Our best recipe: mosaic and scale augmentation off. MuSGD pinned: optimizer=auto picks AdamW below 10,000 steps (all 48-epoch runs).
BEST="mosaic=0 scale=0 optimizer=MuSGD lr0=0.01 momentum=0.9"
train() { local out=$1; shift; python train_yolo.py --data "$Y" --model yolo26l-p2.yaml --init yolo26l.pt --device "$GPU" --out "$RUNS/$out" "$@"; }
score() { python tools/score_yolo_holdout.py "$RUNS/$1/weights/last.pt" --data "$DATA/holdout15_seed0" --out "$RUNS/$1/eval" --device "$GPU" | tee "$RUNS/$1/score.txt"; }

control48()        { train screen48_control        --epochs 48 $BEST "$@" && score screen48_control; }
vflip48()          { train screen48_vflip          --epochs 48 $BEST flipud=0.5 "$@" && score screen48_vflip; }
img768_b8()        { train screen48_img768_b8      --epochs 48 --imgsz 768 --batch 8 $BEST "$@" && score screen48_img768_b8; }
img768_b16()       { train screen48_img768_b16     --epochs 48 --imgsz 768 $BEST "$@" && score screen48_img768_b16; }
# Second seeds: no YOLO setup has two seeds yet, so these measure the noise every decision above is judged against.
control48_seed1()  { train screen48_control_seed1  --epochs 48 --seed 1 $BEST "$@" && score screen48_control_seed1; }
img768_b16_seed1() { train screen48_img768_b16_seed1 --epochs 48 --imgsz 768 --seed 1 $BEST "$@" && score screen48_img768_b16_seed1; }
# For 8 GB GPUs: the control at batch 8 with two seeds. Seed 0 vs 1 = seed noise; batch 8 vs control48 (batch 16) = batch-size effect.
control48_b8()       { train screen48_control_b8       --epochs 48 --batch 8 $BEST "$@" && score screen48_control_b8; }
control48_b8_seed1() { train screen48_control_b8_seed1 --epochs 48 --batch 8 --seed 1 $BEST "$@" && score screen48_control_b8_seed1; }

# The submission model: all 9,392 labelled images, 96 epochs, KL box loss, plus the winning change as extra arguments,
# e.g. bash final_sprint.sh final --imgsz 768   or   bash final_sprint.sh final flipud=0.5
final() {
  sed 's#^train: images/train$#train: [images/train, images/val]#' "$Y" > "$DATA/yolo_holdout15_seed0/data_all.yaml"
  python train_yolo.py --data "$DATA/yolo_holdout15_seed0/data_all.yaml" --model yolo26l-p2.yaml --init yolo26l.pt --device "$GPU" \
      --out "$RUNS/final" --epochs 96 --kl $BEST "$@"
  python -m sarbench.yolo_kl strip "$RUNS/final/weights/last.pt"
  python tools/predict_submission_yolo.py "$RUNS/final/weights/last_stripped.pt" --images-json "$DATA/val/images.json" --root "$DATA/val" \
      --out "$RUNS/final/submission_val.zip" --conf 0.00001 --device "$GPU"
}

"$@"
