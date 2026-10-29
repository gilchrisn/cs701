"""Score a YOLO model on the held-out split the way every run in README section 7 is scored.
Predicts the held-out images (the model's own training input size, score >= --conf, at most 100 boxes per image), writes
<out>/predictions_val.json (COCO format, category_id = YOLO class + 1, boxes in original pixels) and prints COCO mAP@[.5:.95]:
- full: all 1,408 held-out images;
- G0 / G1 / G2: held-out images whose scene is new / a shifted copy of a training image / a near-identical or mirrored copy
  (tools/holdout15_seed0_scene_groups.json, COCO image ids);
- val-weighted: 0.412 G0 + 0.325 G1 + 0.264 G2, the shares of the official val set, so the closest held-out stand-in for the
  leaderboard.
KL-trained weights must be stripped first (python -m sarbench.yolo_kl strip <weights>).
    python tools/score_yolo_holdout.py runs/screen48_control/weights/last.pt --data data/holdout15_seed0 --out runs/screen48_control/eval
Scoring an existing predictions file only: --predictions <file> (weights argument ignored)."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

GROUPS = Path(__file__).with_name('holdout15_seed0_scene_groups.json')
WEIGHTS = {'G0': 587 / 1426, 'G1': 463 / 1426, 'G2': 376 / 1426}

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('weights', type=Path)
p.add_argument('--data', type=Path, required=True, help='held-out folder from tools/make_holdout.py (detection/instances_val.json)')
p.add_argument('--out', type=Path, required=True)
p.add_argument('--conf', type=float, default=0.001)
p.add_argument('--device', default='0')
p.add_argument('--predictions', type=Path, help='score this COCO predictions file instead of predicting')
args = p.parse_args()
quiet = lambda: contextlib.redirect_stdout(io.StringIO())

gt_path = args.data.expanduser() / 'detection' / 'instances_val.json'
if args.predictions:
    preds = json.loads(args.predictions.read_text())
else:
    from ultralytics import YOLO
    model = YOLO(str(args.weights.expanduser()))
    images, preds = json.loads(gt_path.read_text())['images'], []
    for k in range(0, len(images), 32):
        batch = images[k:k + 32]
        results = model.predict([str(args.data.expanduser() / i['file_name']) for i in batch], imgsz=model.overrides.get('imgsz', 512),
                                conf=args.conf, max_det=100, device=args.device, verbose=False)
        for im, r in zip(batch, results):
            for (x1, y1, x2, y2), s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                preds.append({'image_id': im['id'], 'category_id': int(c) + 1, 'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': s})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'predictions_val.json').write_text(json.dumps(preds))

with quiet():
    gt = COCO(str(gt_path))
    ev = COCOeval(gt, gt.loadRes(preds), 'bbox')
    ev.params.imgIds = sorted(gt.getImgIds())
    ev.evaluate()


def coco_map(keep=None):  # accumulate over a subset of images by blanking the others' per-image results
    e = copy.copy(ev)
    n = len(ev._paramsEval.imgIds)
    e.evalImgs = [x if keep is None or ev._paramsEval.imgIds[i % n] in keep else None for i, x in enumerate(ev.evalImgs)]
    with quiet():
        e.accumulate()
        e.summarize()
    return e.stats[0]


groups = {k: coco_map(set(v)) for k, v in json.loads(GROUPS.read_text()).items()}
print(f'mAP full {coco_map():.4f} | G0 new {groups["G0"]:.4f}  G1 shifted copy {groups["G1"]:.4f}  G2 near-identical {groups["G2"]:.4f} '
      f'| val-weighted {sum(WEIGHTS[k] * groups[k] for k in WEIGHTS):.4f}')
