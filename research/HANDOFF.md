# Research handoff (Team 1, 2026-11-04)

Where the model stands, what moved the score, what the dataset turned out to be, and what is still open.
Numbers: **val** = Codabench Phase 1 leaderboard; **held-out** = our own 15% split of train (`tools/make_holdout.py`, seed 0),
scored with pycocotools. Held-out has predicted the *direction* of every val change so far, not its size.

## 1. Current best

| model | val Δm | val F1 | val mAP | held-out mAP |
|---|---|---|---|---|
| YOLO26l-P2 + KL box loss + flip convention, all images, 144 ep, labels from boxes | **54.24** (5th of 9) | 92.87 | 54.60 | 54.5 (85% split) |
| same + 24-ep **exact-copy fine-tune** (HSV and translate jitter off) | not on val (Phase 1 closed) | – | – | **57.2** (+2.8) |

Phase 2 (test, 2,065 images, 2 uploads, closes 8 Nov 23:59): upload 1 = the 144-ep model (safe, val-proven);
upload 2 = the exact-copy fine-tune.

Recipe (Ultralytics 8.4.182, one 24 GB GPU, ~8 h on all images; run from a folder holding `yolo26l.pt`):
```
python research/scripts/train_variants_2026-10-29.py kl+flipconv <name> <yolo data.yaml> 16 <project> 144
INIT_FROM=<that run>/weights/last_stripped.pt python research/scripts/train_variants_2026-10-29.py kl+flipconv+exact <name> <data.yaml> 16 <project> 24
python research/scripts/kl_mod.py strip <run>/weights/last.pt        # always strip KL weights before predicting
python tools/predict_submission_yolo.py <last_stripped.pt> --images-json <test/images.json> --root <test> --out sub.zip --conf 0.00001
```
`INIT_FROM` must be a KL-stripped checkpoint (the KL patch widens the box head after loading).

## 2. What moved the score (measured)

| change | effect | where |
|---|---|---|
| YOLO26l-P2 instead of ViT/Cascade | +10 held-out mAP | architecture |
| mosaic + scale jitter off | +5.0 held-out | copies of training scenes |
| **flip with the dataset's own convention** (x' = W - 1 - x; `flipconv_mod.py`) | **+6.9 val mAP** (the older model was also a restarted run) | the labels of flipped copies |
| 144 instead of 96 epochs | +0.7 val mAP | copies (new scenes slightly down) |
| exact-copy fine-tune (no HSV / translate jitter) | +2.8 held-out, new scenes also up | copies |

Tested and **no gain**: larger YOLO (x), ConvNeXt-B, D-FINE-L, RF-DETR-S, DINOv3 + Cascade (held-out 36.0), strip pooling,
refinement stage, DFL, NWD, alpha-IoU, SPD-Conv, speckle aug, 90-degree rotations, mosaic in the first half, every NMS / WBF /
voting / score-recalibration setting, a classification head inside YOLO (+0.3 F1). A DINOv2 classifier fused with the box scores
gives +3 F1 (val 54.30) but is two models (possible ensemble; not submitted).

## 3. The dataset is a merge (the "build function")

`data/SOURCES_AND_LICENSES.md`: SARDet-100K, RSAR, M4-SAR and HRSID, pooled and re-split by SARFact. Visible in our labels
(`research/scripts/label_fingerprint.py`): whole-pixel boxes for every 256-px JPG / PNG image and for 512-px aircraft and cars;
fractional boxes for 512-px airports, bridges, harbors, playgrounds, wind turbines (rotated or optical-derived labels converted to
horizontal boxes). From the papers (text only): M4-SAR boxes were drawn on **optical** images and moved to SAR by geographic
coordinates only ("coarse rather than pixel-level"), so on new scenes a box can sit off the radar signature or on an object the
radar does not show (our worst cells: new-scene bridges and playgrounds, 94-100% with no candidate box).

About half of held-out and val are **copies of training scenes** (`copies.py`): 545 of 682 matches in the same orientation (417 of them unshifted:
other polarisation or duplicate), 137 flipped (mostly 180°, labelled with x' = W - 1 - x), shifts of ±96 / ±68 px (probably 512-px
tiles cut from 608 / 580-px images), never rescaled. Every gain in section 2 except the architecture swap came from reproducing these copies' labels more exactly.

## 4. Open directions (dataset-specific)

1. **Memorisation of copies.** The exact-copy fine-tune was the largest held-out gain after the flip fix. Untested: longer, or the
   whole 144-ep run without HSV / translate jitter; keeping brightness jitter (VV vs VH copies differ in brightness) but no translate.
2. **Polarisation pairs.** If near-identical copies are VV/VH pairs, an augmentation that maps one polarisation's brightness to the
   other's would mirror the build exactly.
3. **Tile shifts of 68 px** (not a multiple of the 32-px stride) are seen differently from 96-px shifts; a translate augmentation
   restricted to those offsets mirrors that step.
4. **F1** (lowest on the board; every top team has ~95): a single model that classifies as well as a DINO backbone.
5. Error-budget tooling: `budget.py` splits the whole mAP gap over class x error type x scene group x source (exact Shapley).

Course rules that bind every idea: no other SAR data (the public source datasets also hold the val/test labels, so do not download
them even to look), one model, no test-time augmentation, each image in once at <= 512 px.
