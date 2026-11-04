"""T_conv: do flipped copies carry a fixed 1-px label offset (the dataset's flip convention) relative to the exact geometric flip?
(1) For held-out boxes in aligned copies with transform h-flip (1), v-flip (2), rot180 (3), identity (0): signed offset of each edge
    (held-out box minus geometrically mapped copy box), median and share exactly -1 / 0 / +1 px (within 0.05 px).
(2) Loss at stake (measured): over ALL held-out boxes in flipped-copy images (transforms 1-3), the marginal mAP gain at the baseline of
    snapping owners to the geometrically mapped copy label vs to the copy label shifted by the measured per-transform offset. The
    difference is the mAP a model that follows the dataset's flip convention (instead of the exact flip) could gain on these copies.
-> flip_convention.txt"""
import csv, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget as B
from loose_tests import copy_boxes
from loose_profile import gain_replace
pairs = {int(r['image_id']): int(r['transform']) for r in csv.DictReader(open('copies_pairs.csv')) if float(r['corr']) >= 0.6 and float(r['overlap']) >= 0.25}
CB = copy_boxes()
off = defaultdict(list); match = []
for img, m in CB.items():
    v = pairs.get(img)
    if v not in (0, 1, 2, 3): continue
    owner, owns, kinds, I = B.ASSIGN[img]
    for j, a in enumerate(B.GTS[img]):
        cand = [b for c, b in m if c == a['category_id']]
        if not cand: continue
        ious = B.iou_xywh(a['bbox'], cand)[0]
        if ious.max() < 0.3: continue
        g, b = a['bbox'], cand[int(ious.argmax())]
        d = [g[0] - b[0], g[1] - b[1], (g[0] + g[2]) - (b[0] + b[2]), (g[1] + g[3]) - (b[1] + b[3])]
        off[v].append(d)
        k = owner[j]
        if k >= 0 and kinds[k] == 'owner_same' and v in (1, 2, 3): match.append((img, k, b, v))
lines = ['signed edge offsets held-out minus geometric copy (x0, y0, x1, y1), median | share exactly -1 / 0 / +1 px per edge (measured)']
names = {0: 'identity', 1: 'h-flip', 2: 'v-flip', 3: 'rot180'}
med = {}
for v in sorted(off):
    x = np.array(off[v]); med[v] = np.median(x, 0)
    sh = lambda t: ' '.join(f'{np.mean(np.abs(x[:, e] - t) <= 0.05):.2f}' for e in range(4))
    lines.append(f'  {names[v]:8s} n {len(x):4d}: median {np.round(med[v], 2).tolist()} | -1: {sh(-1)} | 0: {sh(0)} | +1: {sh(1)}')
geo = {(img, k): b for img, k, b, v in match}
conv = {(img, k): [b[0] + med[v][0], b[1] + med[v][1], b[2] + med[v][2] - med[v][0], b[3] + med[v][3] - med[v][1]] for img, k, b, v in match}
exact = {(img, k): None for img, k, b, v in match}
g_geo, g_conv = gain_replace(geo), gain_replace(conv)
lines.append(f'owners in flipped-copy images with a copy label: {len(match)}; snapping them to the geometric copy label gains {g_geo:+.4f} mAP, '
             f'to the convention-shifted copy label {g_conv:+.4f}; difference {g_conv - g_geo:+.4f} (measured)')
open('flip_convention.txt', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
