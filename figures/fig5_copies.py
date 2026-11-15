#!/usr/bin/env python
"""Fig. 5 of the report: copies in detail. (a) an overlapping crop and its training image, (b) a near-duplicate rotated by 180
degrees, (c) per class, the percentage of held-out images in each overlap group, with the final model's held-out AP.
Reads: the held-out split (--data), analysis/results/copies_pairs.csv, tools/holdout15_seed0_scene_groups.json,
analysis/results/chain_metrics.json (per-class AP of the final 144-epoch recipe).
Usage: python figures/fig5_copies.py [--data data/holdout15_seed0] [--out figures/out]"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RES = os.path.join(REPO, 'analysis', 'results')
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--data', default=os.path.join(REPO, 'data', 'holdout15_seed0'))
ap.add_argument('--out', default=os.path.join(HERE, 'out'))
cli = ap.parse_args()
D = cli.data
OUT = cli.out
os.makedirs(OUT, exist_ok=True)
INK, INK2, FRAME = '#1F2328', '#5F6670', '#9AA0A6'
GCOL = {'G0': '#E69F00', 'G1': '#009E73', 'G2': '#CC79A7'}
BOX = '#00D084'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'text.color': INK, 'axes.edgecolor': FRAME,
                     'xtick.color': INK2, 'ytick.color': INK2, 'xtick.labelsize': 6.6, 'ytick.labelsize': 7,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.5, 'pdf.fonttype': 42})

ho = json.load(open(f'{D}/detection/instances_val.json'))
tr = json.load(open(f'{D}/detection/instances_train.json'))
cls = {c['id']: c['name'].replace('_', ' ') for c in ho['categories']}


def index(js):
    im, bx = {i['id']: i for i in js['images']}, {}
    for a in js['annotations']:
        bx.setdefault(a['image_id'], []).append(a)
    return im, bx


hoi, hob = index(ho)
tri, trb = index(tr)
pairs = [r for r in csv.DictReader(open(f'{RES}/copies_pairs.csv'))
         if float(r['corr']) >= 0.6 and float(r['overlap']) >= 0.25]
klass = lambda r: cls[hob[int(r['image_id'])][0]['category_id']]
crop = min((r for r in pairs if r['group'] == 'G1' and r['transform'] == '0' and klass(r) == 'aircraft'
            and 0.5 <= float(r['overlap']) <= 0.75), key=lambda r: int(r['image_id']))
rot = min((r for r in pairs if r['group'] == 'G2' and r['transform'] == '3' and r['dx'] == r['dy'] == '0' and float(r['corr']) >= 0.95
           and klass(r) in ('airport', 'harbor')), key=lambda r: int(r['image_id']))

fig = plt.figure(figsize=(6.9, 1.62))
S, gap, y0 = 1.06, 0.06, 0.08


def ax_at(x, w=S, h=S, y=y0):
    return fig.add_axes([x / 6.9, y / 1.62, w / 6.9, h / 1.62])


def show(ax, info, boxes, sub):
    g = np.asarray(Image.open(f"{D}/{info['file_name']}").convert('L'))
    ax.imshow(g, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    for a in boxes:
        x, y, w, h = a['bbox']
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=BOX, lw=1.0))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(FRAME); s.set_linewidth(0.5)
    ax.text(0, 1.03, sub, transform=ax.transAxes, ha='left', va='bottom', fontsize=6.6, color=INK2)
    return g.shape


def title(x, letter, text):
    fig.text(x / 6.9, 1.5 / 1.62, letter, fontweight='bold', fontsize=7.5, va='bottom')
    fig.text((x + 0.2) / 6.9, 1.5 / 1.62, text, fontsize=7.5, va='bottom')


# (a) overlapping crop: the held-out image and its training image, with the held-out frame drawn on the training image
xa = 0.05
hi, ti = hoi[int(crop['image_id'])], tri[int(crop['copy_id'])]
a1 = ax_at(xa)
show(a1, hi, hob[hi['id']], 'held-out image')
a2 = ax_at(xa + S + gap)
show(a2, ti, trb[ti['id']], 'training image')
dx, dy = int(crop['dx']), int(crop['dy'])   # held-out pixel (x, y) shows training pixel (x - dx, y - dy): identity transform, scale 1
x0t, y0t = max(0, -dx), max(0, -dy)          # the shared region, in training coordinates
x1t, y1t = min(ti['width'], hi['width'] - dx), min(ti['height'], hi['height'] - dy)
for ax, ox, oy in ((a2, 0, 0), (a1, dx, dy)):  # outline the shared region in both images
    ax.add_patch(Rectangle((x0t + ox + 2, y0t + oy + 2), x1t - x0t - 4, y1t - y0t - 4, fill=False, ec='white', lw=0.8, ls=(0, (3, 2))))
title(xa, '(a)', 'overlapping crop')

# (b) near-duplicate rotated by 180 degrees
xb = xa + 2 * S + gap + 0.22
hi, ti = hoi[int(rot['image_id'])], tri[int(rot['copy_id'])]
show(ax_at(xb), hi, hob[hi['id']], 'held-out image')
show(ax_at(xb + S + gap), ti, trb[ti['id']], 'training image')
title(xb, '(b)', 'near-duplicate, rotated 180°')

# (c) per class: percentage of held-out images in each overlap group, and the final model's held-out AP
grp = {i: g for g, ids in json.load(open(f'{REPO}/tools/holdout15_seed0_scene_groups.json')).items() for i in ids}
imcls = {a['image_id']: cls[a['category_id']] for a in ho['annotations']}
tab = {}
for i, c in imcls.items():
    tab.setdefault(c, {'G0': 0, 'G1': 0, 'G2': 0})[grp[i]] += 1
ap = json.load(open(f'{RES}/chain_metrics.json'))['epochs_144_final_recipe_85pct']['per_class']
order = sorted(tab, key=lambda c: tab[c]['G0'] / sum(tab[c].values()))
xc = xb + 2 * S + gap + 0.95
ac = fig.add_axes([xc / 6.9, 0.3 / 1.62, (6.9 - xc - 0.42) / 6.9, 0.98 / 1.62])
for k, c in enumerate(order):
    n, left = sum(tab[c].values()), 0
    for g in ('G2', 'G1', 'G0'):
        v = 100 * tab[c][g] / n
        ac.barh(k, v, left=left, color=GCOL[g], height=0.72, ec='white', lw=0.4)
        left += v
    ac.text(103, k, f'{ap[c]:.2f}', va='center', ha='left', fontsize=6.6, color=INK)
ac.text(103, -0.55, 'AP', va='bottom', ha='left', fontsize=6.6, color=INK2)
ac.set_yticks(range(len(order))); ac.set_yticklabels(order); ac.tick_params(axis='y', length=0)
ac.set_xlim(0, 100); ac.set_xticks([0, 50, 100]); ac.set_ylim(len(order) - 0.5, -0.5)
ac.spines['left'].set_visible(False)
ac.set_xlabel('held-out images (%)', fontsize=6.6, labelpad=1)
title(xc - 0.62, '(c)', 'overlap by class')
for k, (g, lab) in enumerate((('G2', 'near-duplicate'), ('G1', 'crop'), ('G0', 'no match'))):
    lx = xc - 0.62 + (0, 0.84, 1.24)[k]
    fig.patches.append(Rectangle((lx / 6.9, 1.4 / 1.62), 0.07 / 6.9, 0.07 / 1.62, transform=fig.transFigure, fc=GCOL[g], ec='none'))
    fig.text((lx + 0.1) / 6.9, 1.435 / 1.62, lab, fontsize=6.2, va='center', color=INK2)

fig.savefig(f'{OUT}/fig5_copies.pdf')
fig.savefig(f'{OUT}/fig5_copies.png', dpi=200)
print('crop pair', crop['image_id'], crop['copy_id'], 'dx', crop['dx'], 'dy', crop['dy'], 'overlap', crop['overlap'])
print('rot pair', rot['image_id'], rot['copy_id'], 'corr', rot['corr'])
print('built', f'{OUT}/fig5_copies.pdf')
