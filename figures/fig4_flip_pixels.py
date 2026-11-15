"""Fig. 4 of the report: one box edge of a horizontally flipped training copy, in the dataset's labels and with Ultralytics' flip.
Reads the held-out split (--data) and analysis/results/copies_pairs.csv. The assert checks the dataset's box is 1 px left of the
exactly flipped box in both edges.
Usage: python figures/fig4_flip_pixels.py [--data data/holdout15_seed0] [--out figures/out]"""
import numpy as np

import _style as S
from _style import COL

cli = S.args(__doc__)
OUT = S.out_dir(cli.out)
HO = cli.data
h, t = 4840, 8057
row = [r for r in S.read_pairs() if int(r['image_id']) == h and int(r['copy_id']) == t][0]
assert row['transform'] == '1' and row['dx'] == '0' and row['dy'] == '0'
TR = S.load_coco(f'{HO}/detection/instances_train.json'); VA = S.load_coco(f'{HO}/detection/instances_val.json')
ft, fv = TR[1][t]['file_name'], VA[1][h]['file_name']
A = S.gray(f'{HO}/{ft}')
W = A.shape[1]
Mi = A[:, ::-1]
Bh = S.gray(f'{HO}/{fv}')
assert np.abs(Mi - Bh).mean() < 0.5
tb, vb = TR[2][t], VA[2][h]
best = None
for a in tb:
    x, y, w, hh = a['bbox']
    if not 8 <= max(w, hh) <= 24:
        continue
    lib = [W - x - w, y, w, hh]
    m = [b for b in vb if b['category_id'] == a['category_id'] and S.iou(lib, b['bbox']) > 0.3]
    if not m:
        continue
    ds = max(m, key=lambda b: S.iou(lib, b['bbox']))['bbox']
    dist = min([abs(ds[0] - b['bbox'][0]) + abs(ds[1] - b['bbox'][1]) for b in vb if b['bbox'] != ds] or [999])
    if best is None or dist > best[0]:
        best = (dist, a['bbox'], lib, ds)
_, cb, lib, ds = best
dxs = [ds[0] - lib[0], ds[0] + ds[2] - lib[0] - lib[2]]
assert all(abs(d + 1) < 0.06 for d in dxs), dxs
v = S.iou(lib, ds)
n = 18
cx, cy = cb[0] + cb[2] / 2, cb[1] + cb[3] / 2
ax0 = int(np.clip(round(cx - n / 2), 0, W - n)); ay0 = int(np.clip(round(cy - n / 2), 0, W - n))
mx0 = W - ax0 - n
c1, c2 = A[ay0:ay0 + n, ax0:ax0 + n], Mi[ay0:ay0 + n, mx0:mx0 + n]
lo, hi = np.percentile(np.r_[c1.ravel(), c2.ravel()], (1, 99.7))
f = S.P(COL, 1.8, OUT)
Pw, gp = 0.92, 0.2
x00 = (COL - 3 * Pw - 2 * gp) / 2
xs = [x00 + k * (Pw + gp) for k in range(3)]
py = 0.62
ttl = ['original', 'mirrored', 'boxes']
o = f.ov()

def pane(k, arr, x0, y0):
    a = f.ax(xs[k], py, Pw, Pw)
    S.show(a, arr, x0, y0, x0 + n, y0 + n, stretch=False, vmin=lo, vmax=hi); S.thin(a)
    S.grid(a, x0, y0, x0 + n, y0 + n, 1, col='#9AA0A6', lw=0.25, alpha=0.6, z=3)
    f.title(xs[k], py + Pw + 0.1, '(%s)' % 'abc'[k], ttl[k])
    return a
a1 = pane(0, A, ax0, ay0)
S.bx(a1, cb, S.TRUE_C, 1.2)
pane(1, Mi, mx0, ay0)
a3 = pane(2, Mi, mx0, ay0)
S.bx(a3, ds, S.TRUE_C, 1.2, z=6)
S.bx(a3, lib, S.LIB_C, 1.2, ls=(0, (3, 1.5)), z=7)
ly = 0.38
o.plot([x00, x00 + 0.22], [ly, ly], color=S.TRUE_C, lw=1.2); o.text(x00 + 0.28, ly, "dataset's box", va='center', fontsize=7)
xl = x00 + 1.35
o.plot([xl, xl + 0.22], [ly, ly], color=S.LIB_C, lw=1.2, ls=(0, (3, 1.5))); o.text(xl + 0.28, ly, 'Ultralytics box', va='center', fontsize=7)
o.text(x00, 0.14, 'grid square = 1 px', fontsize=7, color=S.MUTED, va='center')
o.text(xs[2] + Pw, 0.14, f'IoU of the two boxes: {v:.2f}', fontsize=7, va='center', ha='right')
f.save('fig4_flip_pixels')


