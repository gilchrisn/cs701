"""Codabench submission zip from an Ultralytics YOLO detector, as ONE model: the detections are the detection answer, and the image label
is the class with the largest summed detection score in that image (no separate classification head). imgsz 512, conf 0.001, at most
100 boxes per image. Images with no detection get label 4 (harbor, the most common training class). Writes through
make_submission.write_submission. Example: python tools/predict_submission_yolo.py best.pt --images-json ~/cs701/data/val/images.json \
    --root ~/cs701/data/val --out submission_val_yolo.zip"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from make_submission import write_submission  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('weights', type=Path)
p.add_argument('--images-json', type=Path, required=True)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--conf', type=float, default=0.001)
p.add_argument('--device', default='0')
args = p.parse_args()
images = json.loads(args.images_json.expanduser().read_text())
images = images['images'] if isinstance(images, dict) else images
model = YOLO(str(args.weights.expanduser()))
preds = {}
for k in range(0, len(images), 32):
    batch = images[k:k + 32]
    results = model.predict([str(args.root.expanduser() / im['file_name']) for im in batch], imgsz=512, conf=args.conf, max_det=100,
                            device=args.device, verbose=False)
    for im, r in zip(batch, results):
        sx, sy = 512 / im['width'], 512 / im['height']  # write_submission takes boxes on a 512 x 512 input
        boxes = [[x1 * sx, y1 * sy, x2 * sx, y2 * sy] for x1, y1, x2, y2 in r.boxes.xyxy.tolist()]
        classes, scores = [int(c) for c in r.boxes.cls.tolist()], r.boxes.conf.tolist()
        total = defaultdict(float)
        for c, s in zip(classes, scores):
            total[c] += s
        preds[im['file_name']] = (max(total, key=total.get) if total else 4, boxes, classes, scores)
write_submission(images, preds, out=str(args.out.expanduser()))
print(f'{len(images)} images, {sum(len(v[1]) for v in preds.values())} detections, '
      f'{sum(1 for v in preds.values() if not v[1])} without detections -> {args.out}')
