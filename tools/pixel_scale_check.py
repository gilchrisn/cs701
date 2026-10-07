"""Are the 8-bit pixels on a linear (power-law) scale or already log-scaled?
Speckle is multiplicative: I = R * n. On a linear or power-law display scale the local std grows in proportion to
the local mean (slope 1 of log std vs log mean); after a log mapping the noise is additive (slope 0).
Per image: 8 x 8 windows with mean in [20, 200] and no saturated pixel; windows binned by log mean; in each bin the
10th-percentile std (the most homogeneous windows); slope of log(std_p10) on log(mean) over bins with >= 20 windows.
Writes <out>/pixel_scale.json. Example: python tools/pixel_scale_check.py ~/cs701-data/train ~/cs701-runs/data_checks"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
names = json.loads((src.parent / 'classes.json').read_text()) if (src.parent / 'classes.json').exists() else None
rows = list(csv.DictReader(open(src / 'labels.csv')))
per_image = []
for r in rows:
    x = np.asarray(Image.open(src / r['file_name']).convert('L'), dtype=np.float64)
    h, w = (x.shape[0] // 8) * 8, (x.shape[1] // 8) * 8
    win = x[:h, :w].reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 64)
    m, s = win.mean(1), win.std(1)
    ok = (m >= 20) & (m <= 200) & (win.max(1) < 255)
    m, s = m[ok], s[ok]
    if len(m) < 100:
        continue
    lm = np.log(m)
    edges = np.arange(np.log(20), np.log(200) + 0.1, 0.1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (lm >= lo) & (lm < hi)
        if sel.sum() >= 20:
            p10 = np.percentile(s[sel], 10)
            if p10 > 0:
                xs.append(lm[sel].mean())
                ys.append(np.log(p10))
    if len(xs) < 5:
        continue
    slope = float(np.polyfit(xs, ys, 1)[0])
    cv = float(np.median(np.exp(np.array(ys) - np.array(xs))))  # std_p10 / mean per bin
    per_image.append({'image_id': int(r['image_id']), 'label': int(r['label']), 'size': f'{x.shape[1]}x{x.shape[0]}',
                      'slope': slope, 'cv_p10': cv})

groups = defaultdict(list)
for p in per_image:
    groups[('size', p['size'] if p['size'] in ('512x512', '256x256') else 'other')].append(p)
    groups[('class', p['label'])].append(p)
summary = {}
for (kind, key), ps in sorted(groups.items(), key=lambda kv: str(kv[0])):
    sl = np.array([p['slope'] for p in ps])
    summary[f'{kind}={key}'] = {'n': len(ps), 'slope_median': float(np.median(sl)), 'slope_q25': float(np.percentile(sl, 25)),
                                'slope_q75': float(np.percentile(sl, 75)),
                                'cv_p10_median': float(np.median([p['cv_p10'] for p in ps]))}
res = {'n_images_total': len(rows), 'n_images_used': len(per_image), 'summary': summary, 'per_image': per_image}
(out / 'pixel_scale.json').write_text(json.dumps(res, indent=1))
print(f"used {len(per_image)} of {len(rows)} images")
for k, v in summary.items():
    print(f"{k:12s} n={v['n']:5d} slope median {v['slope_median']:.2f} (IQR {v['slope_q25']:.2f}..{v['slope_q75']:.2f}) cv_p10 {v['cv_p10_median']:.2f}")
