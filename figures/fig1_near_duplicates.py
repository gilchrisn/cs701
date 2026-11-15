"""Fig. 1 of the report: (a) a held-out image next to its flipped training copy and their difference, (b) the share of images
in each overlap group in the held-out, validation and test sets.
Reads: the held-out split (--data), analysis/results/copies_pairs.csv, tools/holdout15_seed0_scene_groups.json (held-out group counts),
analysis/results/test_groups.txt (validation and test group counts).
Usage: python figures/fig1_near_duplicates.py [--data data/holdout15_seed0] [--out figures/out]"""
import json
import os
import re

import numpy as np

import _style as S

cli = S.args(__doc__)
OUT = S.out_dir(cli.out)
HO = cli.data
h, t = 3678, 9147
row = [r for r in S.read_pairs() if int(r['image_id']) == h and int(r['copy_id']) == t][0]
assert row['transform'] == '1' and row['dx'] == '0' and row['dy'] == '0' and float(row['corr']) > 0.95
ft = S.load_coco(f'{HO}/detection/instances_train.json')[1][t]['file_name']
fv = S.load_coco(f'{HO}/detection/instances_val.json')[1][h]['file_name']
A, B = S.gray(f'{HO}/{ft}'), S.gray(f'{HO}/{fv}')
M = A[:, ::-1]
D = np.abs(M - B)
mad = D.mean()

# group counts (no match in train, overlapping crop, near-duplicate)
g = json.load(open(os.path.join(S.REPO, 'tools', 'holdout15_seed0_scene_groups.json')))
held = tuple(len(g[k]) for k in ('G0', 'G1', 'G2'))
cnt = {}
for line in open(os.path.join(S.RES, 'test_groups.txt')):
    m = re.match(r'(val|test) n=\d+: G0 (\d+).*G1 (\d+).*G2 (\d+)', line)
    if m:
        cnt[m.group(1)] = tuple(int(x) for x in m.groups()[1:])

f = S.P(S.FULL, 1.45, OUT)
o = f.ov()
Pw, gp, y0 = 0.95, 0.09, 0.08
for k, (arr, ttl) in enumerate([(A, 'train image'), (M, 'flipped'), (B, 'held-out image'), (D, 'difference')]):
    xx = 0.05 + k * (Pw + gp)
    ax = f.ax(xx, y0, Pw, Pw)
    S.show(ax, arr, 0, 0, 512, 512, stretch=False, vmin=0, vmax=255); S.thin(ax)
    o.text(xx, y0 + Pw + 0.05, ttl, fontsize=7, color=S.MUTED, va='bottom')
    if k == 3:
        ax.text(24, 488, f'mean difference\n{mad:.2f} of 255', color='white', ha='left', va='bottom', fontsize=7, linespacing=1.15)
top = 1.33
f.title(0.05, top, '(a)', 'held-out image and its training copy')
bx0, bw = 4.9, 1.85
ax = f.ax(bx0, 0.36, bw, 0.55)
splits = [('held-out', held), ('val', cnt['val']), ('test', cnt['test'])]
for i, (n, c) in enumerate(splits):
    tot = sum(c)
    left = 0
    for gi, gk in enumerate(('G0', 'G1', 'G2')):
        v = 100 * c[gi] / tot
        ax.barh(i, v, left=left, color=S.G_COL[gk], height=0.7, ec='white', lw=0.5)
        ax.text(left + v / 2, i, f'{v:.0f}', ha='center', va='center', fontsize=7, color=S.INK)
        left += v
ax.set_yticks(range(3)); ax.set_yticklabels([n for n, _ in splits]); ax.invert_yaxis()
ax.set_xlim(0, 100); ax.set_ylim(2.5, -0.5); ax.set_xticks([0, 50, 100]); ax.set_xlabel('share of images (%)', labelpad=1)
ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
c = splits[0][1]
tot = sum(c)
cum = np.r_[0, np.cumsum(c)] / tot * 100
names = ['no match\nin train', 'overlapping\ncrop', 'near-\nduplicate']
for gi in range(3):
    mid = (cum[gi] + cum[gi + 1]) / 2
    ha = 'center'
    if gi == 2:
        mid, ha = 100, 'right'
    ax.text(mid, -0.62, names[gi], ha=ha, va='bottom', fontsize=7, color=S.INK, linespacing=1.1, clip_on=False)
f.title(bx0 - 0.4, top, '(b)', 'overlap with train, by split')
f.save('fig1_near_duplicates')
