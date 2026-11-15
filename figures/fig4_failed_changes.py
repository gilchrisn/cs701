"""Fig. 4 of the report: change in held-out mAP of eleven changes against their control runs (grey band: run-to-run variation).
Reads figures/data/failed_changes.csv (held-out mAP of each change and of its control; origin of every number in that file) and
figures/data/seed_pairs.csv (the eight seed pairs whose root mean square gives the grey band).
Usage: python figures/fig4_failed_changes.py [--out figures/out]"""
import csv
import math
import os

import matplotlib.transforms as mtrans

import _style as S
from _style import COL, INK, MUTED

OUT = S.out_dir(S.args(__doc__).out)
d = {r['change']: (float(r['mAP_change']) - float(r['mAP_control'])) for r in csv.DictReader(open(os.path.join(S.HERE, 'data', 'failed_changes.csv')))}
pairs = list(csv.DictReader(open(os.path.join(S.HERE, 'data', 'seed_pairs.csv'))))
band = math.sqrt(sum((float(r['mAP_seed0']) - float(r['mAP_seed1'])) ** 2 for r in pairs) / len(pairs)) * 100
band = round(band, 2)   # 0.55 points, as in the caption
groups = [('Backbone', ['YOLO26x, 8 layers frozen', 'ConvNeXt-B', 'DINOv2 feature fusion', 'SPD-Conv']),
          ('Detection head', ['D-FINE detector', 'Refinement stage']),
          ('Box loss', ['NWD loss', 'DFL']),
          ('Input and augmentation', ['Oversampling bridge images', 'SAR-derived input channels', 'Copy-paste'])]
f = S.P(COL, 2.05, OUT)
axx, axw, axy, axh = 1.55, 1.65, 0.4, 1.4
ax = f.ax(axx, axy, axw, axh)
ax.axvspan(-band, band, color='#E4E7EB', zorder=0, lw=0)
y = 0.0
yt, yl, heads = [], [], []
for g, items in groups:
    heads.append((y, g))
    y -= 1.0
    for nm in items:
        wrapped = nm
        two = '\n' in wrapped
        if two:
            y -= 0.25
        v = d[nm] * 100
        ax.hlines(y, 0, v, color='#C9CDD2', lw=0.8, zorder=1)
        ax.plot([v], [y], 'o', color=INK, mec='none', ms=4, zorder=3)
        yt.append(y); yl.append(wrapped)
        y -= 1.0 + (0.25 if two else 0)
    y -= 0.45
ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6.5, linespacing=1.05)
ax.axvline(0, color=INK, lw=0.6, zorder=2)
ax.set_xlim(-4.6, 1.2); ax.set_ylim(y + 0.55, 0.7)
ax.set_xticks([-4, -3, -2, -1, 0, 1])
ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
ax.set_xlabel('change in held-out mAP\nvs control (points)', labelpad=2)
tr = mtrans.blended_transform_factory(f.fig.transFigure, ax.transData)
for yy, g in heads:
    ax.text(0.05 / COL, yy, g, transform=tr, ha='left', va='center', fontsize=7, color=MUTED, clip_on=False)
ax.text(0, 0.62, 'run-to-run\nvariation', ha='center', va='bottom', fontsize=6.5, color=MUTED, clip_on=False, linespacing=1.0)
f.save('fig4_failed_changes')
