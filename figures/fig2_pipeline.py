#!/usr/bin/env python
"""Fig. 2 of the report: pipeline diagram (A: finding copies of training images, B: training and prediction). Reads no data.
Usage: python figures/fig2_pipeline.py [--out figures/out]"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'))
OUT = ap.parse_args().out
os.makedirs(OUT, exist_ok=True)
INK, INK2 = '#1F2328', '#5F6670'
LIB_FC, LIB_EC = '#F3F4F6', '#B4BAC2'      # Ultralytics YOLO26l-P2, unchanged
OURS_FC, OURS_EC = '#FFF1C1', '#C9A227'    # our changes and additions
W, H = 6.9, 1.98

plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK, 'pdf.fonttype': 42})
fig = plt.figure(figsize=(W, H))
o = fig.add_axes([0, 0, 1, 1])
o.set_xlim(0, W); o.set_ylim(0, H); o.axis('off')


def box(x, y, w, h, title, body, ours=False):
    fc, ec = (OURS_FC, OURS_EC) if ours else (LIB_FC, LIB_EC)
    o.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.05', fc=fc, ec=ec, lw=0.6))
    o.text(x + w / 2, y + h - 0.07, title, ha='center', va='top', fontsize=7.4, fontweight='bold')
    o.text(x + w / 2, y + h - 0.25, body, ha='center', va='top', fontsize=6.9, color=INK2, linespacing=1.25)


def arrow(x0, y0, x1, y1):
    o.annotate('', xy=(x1, y1), xytext=(x0, y0),
               arrowprops=dict(arrowstyle='-|>', color=INK2, lw=0.7, mutation_scale=6, shrinkA=0, shrinkB=0))


def heading(x, y, letter, text):
    o.text(x, y, letter, fontsize=7.5, fontweight='bold', va='bottom')
    o.text(x + 0.16, y, text, fontsize=7.5, va='bottom')


# ---------------- row A: finding copies of training images (all ours)
yA, hA, wA, gA = 1.08, 0.62, (6.8 - 2 * 0.32) / 3, 0.32
heading(0.05, yA + hA + 0.06, 'A', 'Train\u2013test overlap')
xa = [0.05, 0.05 + wA + gA, 0.05 + 2 * (wA + gA)]
box(xa[0], yA, wA, hA, 'Match to the training set', 'each held-out, val and test image\nagainst every training image,\nalso after flipping it', ours=True)
box(xa[1], yA, wA, hA, 'Overlap groups', 'near-duplicate, overlapping crop,\nno match in train;\nmAP is reported per group', ours=True)
box(xa[2], yA, wA, hA, 'Compare boxes of copies', 'boxes in flipped copies are\n1 px apart; the dataset\nuses x′ = W − 1 − x', ours=True)
for i in range(2):
    arrow(xa[i] + wA, yA + hA / 2, xa[i + 1], yA + hA / 2)

# ---------------- row B: the detector
yB, hB = 0.04, 0.74
heading(0.05, yB + hB + 0.06, 'B', 'Training and prediction')
ws = [1.0, 1.35, 1.85, 1.25, 1.0]
names = [('Input', 'grey SAR image,\n512 px', False),
         ('Augmentation', 'flips follow the dataset\'s\nconvention; no mosaic,\nno scale jitter', True),
         ('YOLO26l-P2 detector', 'COCO-pretrained; strides 4 to 32;\none-stage head', False),
         ('Image class', 'class with the largest\ntotal box score', True),
         ('Output', 'image class\nand boxes', False)]
gap = (W - 0.10 - sum(ws)) / (len(ws) - 1)
xs = [0.05]
for wd in ws[:-1]:
    xs.append(xs[-1] + wd + gap)
for x, wd, (t, b, ours) in zip(xs, ws, names):
    box(x, yB, wd, hB, t, b, ours)
# KL loss: our change to the detector's loss
kx, kw = xs[2] + 0.45, ws[2] - 0.9
o.add_patch(FancyBboxPatch((kx, yB + 0.06), kw, 0.19, boxstyle='round,pad=0,rounding_size=0.04', fc=OURS_FC, ec=OURS_EC, lw=0.6))
o.text(kx + kw / 2, yB + 0.155, 'KL box loss', ha='center', va='center', fontsize=6.8)
for i in range(len(ws) - 1):
    arrow(xs[i] + ws[i], yB + hB / 2, xs[i + 1], yB + hB / 2)

# ---------------- connector: the flip formula found in A is used in B's augmentation (elbow, no diagonal)
cx0 = xa[2] + wA / 2
cx1 = xs[1] + ws[1] / 2
ymid = yB + hB + 0.2
o.plot([cx0, cx0, cx1], [yA, ymid, ymid], color=INK2, lw=0.7, ls=(0, (3, 2)))
arrow(cx1, ymid, cx1, yB + hB)
o.text((cx0 + cx1) / 2 + 0.6, ymid, ' flip convention, used in training ', fontsize=6.6, color=INK2, ha='center', va='center',
       bbox=dict(fc='white', ec='none', pad=0.5))

# ---------------- legend (top right)
lx, ly = 4.75, H - 0.12
for dx, (fc, ec, lab) in zip((0, 0.62), ((OURS_FC, OURS_EC, 'ours'), (LIB_FC, LIB_EC, 'Ultralytics, unchanged'))):
    o.add_patch(FancyBboxPatch((lx + dx, ly - 0.05), 0.16, 0.1, boxstyle='round,pad=0,rounding_size=0.02', fc=fc, ec=ec, lw=0.6))
    o.text(lx + dx + 0.22, ly, lab, va='center', fontsize=6.6, color=INK2)

fig.savefig(f'{OUT}/fig2_pipeline.pdf')
fig.savefig(f'{OUT}/fig2_pipeline.png', dpi=200)
print('built', f'{OUT}/fig2_pipeline.pdf')
