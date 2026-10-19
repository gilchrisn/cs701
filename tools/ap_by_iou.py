"""AP at each COCO IoU threshold (0.50 ... 0.95), overall, per box-side bin (512 input px) and per class.
Shows where mAP@[.5:.95] is lost: a run whose AP50 is fine but whose AP falls fast with the threshold has a box-precision
problem. Writes <run>/ap_by_iou.json. Example: python tools/ap_by_iou.py ~/cs701-runs/2026-10-08_ref_s0"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import SIDE_BINS, coco_eval, to_512  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbench.paths import data_root  # noqa: E402


def ap(p, t, k=None, a=0):
    q = p[t, :, :, a, -1] if k is None else p[t, :, k, a, -1]
    q = q[q > -1]
    return float(q.mean()) if q.size else float('nan')


run = Path(sys.argv[1]).expanduser()
metrics = json.loads((run / 'metrics.json').read_text())
split = metrics.get('posthoc', {}).get('split', 'val')  # tools/infer.py --split train evaluates training images
gt = json.loads((data_root(metrics['args']['data']) / 'detection' / f'instances_{split}.json').read_text())
dets = json.loads((run / 'predictions_val.json').read_text())
inferred = {int(k) for k in json.loads((run / 'class_probs_val.json').read_text())}  # every image the model saw
gt = {**gt, 'images': [i for i in gt['images'] if i['id'] in inferred],
      'annotations': [a for a in gt['annotations'] if a['image_id'] in inferred]}
names = [c['name'] for c in sorted(gt['categories'], key=lambda c: c['id'])]
g512, d512 = to_512(gt, dets)
_, p, ev = coco_eval(g512, d512, SIDE_BINS)
thr = [round(float(t), 2) for t in ev.params.iouThrs]
res = {'split': split, 'n_images': len(gt['images']), 'iou_thresholds': thr,
       'by_side': {lab: [ap(p, t, a=i) for t in range(len(thr))] for i, (lab, _, _) in enumerate(SIDE_BINS)},
       'per_class': {names[k]: [ap(p, t, k=k) for t in range(len(thr))] for k in range(len(names))}}
(run / 'ap_by_iou.json').write_text(json.dumps(res, indent=2))
print('IoU threshold:', ' '.join(f'{t:5.2f}' for t in thr))
for group in ('by_side', 'per_class'):
    for lab, v in res[group].items():
        print(f'{lab:>13}:', ' '.join(f'{x:5.3f}' for x in v))
