"""Label fingerprints of the training set: how each (image source, class) group was labelled.
Prints, per group: boxes, share with all four coordinates whole numbers, share exactly square, most common box size.
Whole-pixel groups come from the SARDet-100K / HRSID side of the merge; fractional groups from rotated (RSAR) or optical-derived
(M4-SAR) labels converted to horizontal boxes.
Usage: python label_fingerprint.py <instances_train.json>"""
import json, sys
from collections import Counter, defaultdict
import numpy as np

js = json.load(open(sys.argv[1]))
im = {i['id']: i for i in js['images']}; names = {c['id']: c['name'] for c in js['categories']}
src = lambda i: 'png' if im[i]['file_name'].endswith('png') else ('jpg256' if im[i]['width'] == 256 else 'jpg512')
G = defaultdict(list)
for a in js['annotations']:
    G[(src(a['image_id']), names[a['category_id']])].append(a['bbox'])
print(f'{"source":7s} {"class":13s} {"boxes":>6s} {"whole":>6s} {"square":>7s}  most common (w, h)')
for k in sorted(G):
    b = np.array(G[k])
    if len(b) < 100: continue
    whole = np.mean(np.all(np.abs(b - np.round(b)) < 1e-6, 1)); square = np.mean(np.abs(b[:, 2] - b[:, 3]) < 1e-3)
    top = Counter((round(float(w), 3), round(float(h), 3)) for _, _, w, h in b).most_common(1)[0]
    print(f'{k[0]:7s} {k[1]:13s} {len(b):6d} {whole:6.2f} {square:7.2f}  {top[0]} x{top[1]}')
