"""AP per box-side bin (side at 512 x 512 input) split by original image size, plus the class mix of each image size.
Asks whether the 512 x 512 vs 256 x 256 mAP gap is only object size. Writes <run>/by_source.json.
Example: python tools/by_source.py ~/cs701-runs/2026-10-08_ref_s0"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import SIDE_BINS, ap_from_precision, coco_eval, to_512  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbench.paths import data_root  # noqa: E402

run = Path(sys.argv[1]).expanduser()
metrics = json.loads((run / 'metrics.json').read_text())
gt = json.loads((data_root(metrics['args']['data']) / 'detection' / 'instances_val.json').read_text())
dets = json.loads((run / 'predictions_val.json').read_text())
names = {c['id']: c['name'] for c in gt['categories']}
g512, d512 = to_512(gt, dets)
res = {}
for lab, wh in [('512x512', (512, 512)), ('256x256', (256, 256))]:
    ids = {i['id'] for i in gt['images'] if (i['width'], i['height']) == wh}
    g = {**g512, 'images': [i for i in g512['images'] if i['id'] in ids],
         'annotations': [x for x in g512['annotations'] if x['image_id'] in ids]}
    _, p, _ = coco_eval(g, [x for x in d512 if x['image_id'] in ids], SIDE_BINS)
    sides = np.sqrt([a['area'] for a in g['annotations']])
    res[lab] = {'n_images': len(ids), 'n_gt': len(g['annotations']),
                'by_side': {b: {'AP': ap_from_precision(p, a=k), 'n_gt': int(((sides >= lo) & (sides < hi)).sum())}
                            for k, (b, lo, hi) in enumerate(SIDE_BINS)},
                'gt_per_class': dict(Counter(names[a['category_id']] for a in g['annotations']))}
(run / 'by_source.json').write_text(json.dumps(res, indent=2))
for lab, r in res.items():
    print(lab, r['n_images'], 'images;', ', '.join(f"{b} {v['AP']:.3f} (n={v['n_gt']})" for b, v in r['by_side'].items()))
    print('   classes:', r['gt_per_class'])
