"""How many anchors does each training ground-truth box get as an RPN positive (IoU >= 0.7), by box side at 512 input?
Uses the reference model's exact anchors (sarbench/model.py). With --nwd, the same count under max(IoU, NWD) (what --nwd trains with).
Writes <out>/anchor_coverage[_nwd].json. Example: python tools/anchor_coverage.py ~/cs701-data/holdout15_seed0 ~/cs701-runs/data_checks"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import box_iou

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbench.model import iou_or_nwd  # noqa: E402

root, out = Path(sys.argv[1]), Path(sys.argv[2])
use_nwd = '--nwd' in sys.argv
gen = AnchorGenerator(((8, 11), (16, 23), (32, 45), (64, 91), (128, 181)), ((0.5, 1.0, 2.0),) * 5)
feats = [torch.zeros(1, 1, 512 // s, 512 // s) for s in (4, 8, 16, 32, 64)]
anchors = gen(ImageList(torch.zeros(1, 1, 512, 512), [(512, 512)]), feats)[0]
coco = json.loads((root / 'detection' / 'instances_train.json').read_text())
size = {i['id']: (i['width'], i['height']) for i in coco['images']}
boxes, sides = [], []
for a in coco['annotations']:
    w, h = size[a['image_id']]
    x, y, bw, bh = a['bbox']
    sx, sy = 512 / w, 512 / h
    boxes.append([x * sx, y * sy, (x + bw) * sx, (y + bh) * sy])
    sides.append(np.sqrt(bw * sx * bh * sy))
boxes, sides = torch.tensor(boxes), np.array(sides)
sim = iou_or_nwd if use_nwd else box_iou
n_pos, best = [], []
for chunk in torch.split(boxes, 512):
    m = sim(chunk, anchors)
    n_pos.append((m >= 0.7).sum(1))
    best.append(m.max(1).values)
n_pos, best = torch.cat(n_pos).numpy(), torch.cat(best).numpy()
res = {}
for lab, lo, hi in [('<8', 0, 8), ('8-16', 8, 16), ('16-32', 16, 32), ('>32', 32, 1e5)]:
    sel = (sides >= lo) & (sides < hi)
    res[lab] = {'n_gt': int(sel.sum()), 'frac_with_no_positive': float((n_pos[sel] == 0).mean()),
                'median_positives': float(np.median(n_pos[sel])), 'median_best_similarity': float(np.median(best[sel]))}
name = 'anchor_coverage_nwd.json' if use_nwd else 'anchor_coverage.json'
(out / name).write_text(json.dumps(res, indent=1))
print('NWD' if use_nwd else 'IoU', json.dumps(res))
