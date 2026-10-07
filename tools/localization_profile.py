"""Where is localization lost? For every held-out ground-truth box: the IoU of the best same-class detection
(score >= 0.05), by box side at 512 input and by class. Writes <run>/localization_profile.json.
Example: python tools/localization_profile.py ~/cs701-runs/2026-10-08_ref_s0"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_iou

run = Path(sys.argv[1]).expanduser()
root = Path(json.loads((run / 'metrics.json').read_text())['args']['data'])
gt = json.loads((root / 'detection' / 'instances_val.json').read_text())
dets = json.loads((run / 'predictions_val.json').read_text())
names = {c['id']: c['name'] for c in gt['categories']}
size = {i['id']: (i['width'], i['height']) for i in gt['images']}
by_img = defaultdict(list)
for d in dets:
    by_img[d['image_id']].append(d)
xyxy = lambda b: [b[0], b[1], b[0] + b[2], b[1] + b[3]]
rows = []
for a in gt['annotations']:
    w, h = size[a['image_id']]
    side = np.sqrt(a['bbox'][2] * 512 / w * a['bbox'][3] * 512 / h)
    same = [d for d in by_img[a['image_id']] if d['category_id'] == a['category_id']]
    best = box_iou(torch.tensor([xyxy(a['bbox'])]), torch.tensor([xyxy(d['bbox']) for d in same])).max().item() if same else 0.0
    rows.append((names[a['category_id']], side, best))
bins = [('<8', 0, 8), ('8-16', 8, 16), ('16-32', 16, 32), ('>32', 32, 1e5)]
def summary(sel):
    b = np.array([r[2] for r in sel])
    return {'n': len(sel), 'median_best_iou': float(np.median(b)) if len(b) else None,
            'frac_best_iou_ge_0.5': float((b >= 0.5).mean()) if len(b) else None,
            'frac_best_iou_ge_0.75': float((b >= 0.75).mean()) if len(b) else None,
            'frac_best_iou_lt_0.1': float((b < 0.1).mean()) if len(b) else None}
res = {'by_side': {lab: summary([r for r in rows if lo <= r[1] < hi]) for lab, lo, hi in bins},
       'by_class': {}}
for c in sorted(set(r[0] for r in rows)):
    sel = [r for r in rows if r[0] == c]
    res['by_class'][c] = {**summary(sel), 'median_side_512': float(np.median([r[1] for r in sel]))}
(run / 'localization_profile.json').write_text(json.dumps(res, indent=1))
for k, v in res['by_side'].items():
    print(f"side {k:6s} n={v['n']:5d} median best IoU {v['median_best_iou']:.2f}  >=0.5 {v['frac_best_iou_ge_0.5']:.2f}  >=0.75 {v['frac_best_iou_ge_0.75']:.2f}  <0.1 {v['frac_best_iou_lt_0.1']:.2f}")
for k, v in res['by_class'].items():
    print(f"{k:13s} n={v['n']:5d} median side {v['median_side_512']:6.1f}  median best IoU {v['median_best_iou']:.2f}  >=0.5 {v['frac_best_iou_ge_0.5']:.2f}  >=0.75 {v['frac_best_iou_ge_0.75']:.2f}  <0.1 {v['frac_best_iou_lt_0.1']:.2f}")
