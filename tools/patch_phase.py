"""Does box precision depend on where an object sits on the ViT patch grid? For every held-out ground truth with side
< 32 px at 512 input: best same-class detection IoU (score >= 0.05, as tools/localization_profile.py) against the
offset of its centre from the nearest grid line, per axis, for grid period 16 (ViT-B/16 patches) and 4 (finest anchor
stride; control). 'near centre' = both axes >= period/4 from a grid line; 'near corner' = both axes < period/8.
Writes <run>/patch_phase.json. Example: python tools/patch_phase.py ~/cs701-runs/2026-10-09_cascade24_s0"""
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
size = {i['id']: (i['width'], i['height']) for i in gt['images']}
by_img = defaultdict(list)
for d in json.loads((run / 'predictions_val.json').read_text()):
    by_img[d['image_id']].append(d)
xyxy = lambda b: [b[0], b[1], b[0] + b[2], b[1] + b[3]]
rows = []  # (side at 512, centre x at 512, centre y at 512, best IoU)
for a in gt['annotations']:
    w, h = size[a['image_id']]
    sx, sy = 512 / w, 512 / h
    x, y, bw, bh = a['bbox']
    side = np.sqrt(bw * sx * bh * sy)
    if side >= 32:
        continue
    same = [d for d in by_img[a['image_id']] if d['category_id'] == a['category_id']]
    best = box_iou(torch.tensor([xyxy(a['bbox'])]), torch.tensor([xyxy(d['bbox']) for d in same])).max().item() if same else 0.0
    rows.append((side, (x + bw / 2) * sx, (y + bh / 2) * sy, best))
r = np.array(rows)
res = {'n_gt_side_lt_32': len(r)}
for period in (16, 4):
    off = lambda c: np.minimum(c % period, period - c % period)  # distance to the nearest grid line, 0..period/2
    dx, dy = off(r[:, 1]), off(r[:, 2])
    groups = {'near_centre': (dx >= period / 4) & (dy >= period / 4), 'near_corner': (dx < period / 8) & (dy < period / 8)}
    out = {}
    for name, sel in groups.items():
        found = sel & (r[:, 3] >= 0.1)
        out[name] = {'n': int(sel.sum()), 'n_found': int(found.sum()),
                     'mean_best_iou_found': float(r[found, 3].mean()) if found.any() else None,
                     'frac_ge_0.5': float((r[sel, 3] >= 0.5).mean()) if sel.any() else None,
                     'frac_ge_0.75': float((r[sel, 3] >= 0.75).mean()) if sel.any() else None}
    out['centre_minus_corner_mean_iou'] = out['near_centre']['mean_best_iou_found'] - out['near_corner']['mean_best_iou_found']
    res[f'period_{period}'] = out
(run / 'patch_phase.json').write_text(json.dumps(res, indent=1))
for period in (16, 4):
    o = res[f'period_{period}']
    print(f"period {period:2d}: " + '  '.join(f"{k} n={v['n']} found={v['n_found']} meanIoU={v['mean_best_iou_found']:.3f} >=0.5 {v['frac_ge_0.5']:.3f} >=0.75 {v['frac_ge_0.75']:.3f}"
                                         for k, v in o.items() if isinstance(v, dict)) + f"  centre-corner {o['centre_minus_corner_mean_iou']:+.4f}")
