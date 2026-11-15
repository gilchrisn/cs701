#!/usr/bin/env python
"""Fig. 3 of the report: held-out mAP after each step, by object size and by overlap group.
Reads only analysis/results/chain_metrics.json (python analysis/chain_metrics.py writes it; every step scored at score >= 0.05).
Usage: python figures/fig3_journey.py [--out figures/out]"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--out', default=os.path.join(HERE, 'out'))
ap.add_argument('--metrics', default=os.path.join(HERE, '..', 'analysis', 'results', 'chain_metrics.json'))
args = ap.parse_args()
OUT = args.out
os.makedirs(OUT, exist_ok=True)
CM = json.load(open(args.metrics))
INK, INK2, GRID = '#1F2328', '#59616B', '#D8DCE0'
GCOL = {'G0': '#E69F00', 'G1': '#009E73', 'G2': '#CC79A7'}
GNAME = {'G0': 'no match in train', 'G1': 'overlapping crop', 'G2': 'near-duplicate'}
SIZES = [('<8', '< 8 px'), ('8-16', '8–16 px'), ('16-32', '16–32 px'), ('>32', '> 32 px')]
SCOL = ['#9ECAE1', '#4A98C9', '#1D63A6', '#08306B']   # one hue, light = small objects, dark = large

STEPS = [('course_vit_lora_24ep', 'course\nmodel'), ('cascade_24ep', 'Cascade\nR-CNN'),
         ('cascade_best_48+24ep', '+ longer\ntraining'), ('yolo26lp2_96ep_mosaic_on', 'YOLO26l-P2'),
         ('mosaic_scale_off', 'mosaic and\nscale jitter off'), ('kl_box_loss', '+ KL\nbox loss'),
         ('flip_convention_plus_vflip', '+ flip\nconvention'), ('epochs_144_final_recipe_85pct', '+ 144\nepochs\n(submitted)'),
         ('exact_copy_ft48_post_competition', 'brightness,\nshift jitter\noff*')]
X = list(range(len(STEPS)))

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.titlesize': 7.5, 'xtick.labelsize': 6.6, 'ytick.labelsize': 6.6,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5,
    'ytick.major.width': 0.5, 'xtick.major.size': 2, 'ytick.major.size': 2, 'pdf.fonttype': 42})

fig, axes = plt.subplots(2, 1, figsize=(6.9, 2.72), sharex=True)
fig.subplots_adjust(left=0.075, right=0.86, bottom=0.165, top=0.925, hspace=0.38)


def panel(ax, series, title, overall_line=True, phase_note=True):
    if overall_line:
        overall = [CM[k]['mAP'] for k, _ in STEPS]
        ax.plot(X, overall, color=INK, lw=1.0, ls=(0, (3, 2)), zorder=4)
        ax.text(X[-1] + 0.2, overall[-1], 'all objects', color=INK, fontsize=7, va='center')
    for vals, col, name, lw, mk in series:
        ax.plot(X, vals, color=col, lw=lw, marker=mk, ms=1.8 + lw, zorder=3)
        ax.text(X[-1] + 0.2, vals[-1], name, color=INK, fontsize=7, va='center')
    ax.axvspan(len(X) - 1.5, len(X) - 0.5, color='#F1F2F4', zorder=0, lw=0)
    if phase_note:
        ax.text(len(X) - 1, 0.81, 'after Phase 1', ha='center', va='top', fontsize=6.6, color=INK2)
    ax.set_xticks(X)
    ax.set_xticklabels([n for _, n in STEPS], linespacing=1.1)
    ax.set_xlim(-0.3, len(X) - 0.5)
    ax.set_ylim(0, 0.85); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.yaxis.grid(True, color='#E5E7EB', lw=0.4); ax.set_axisbelow(True)
    ax.set_title(r'$\bf{' + title[:3] + '}$' + title[3:], loc='left', pad=3)
    ax.set_ylabel('held-out mAP')


size_series = [([CM[k]['by_side'][s] for k, _ in STEPS], c, n, 1.1, m) for (s, n), c, m in zip(SIZES, SCOL, 'os^D')]
group_series = [([CM[k]['G2'] for k, _ in STEPS], GCOL['G2'], GNAME['G2'], 1.9, 'o'),
                ([CM[k]['G1'] for k, _ in STEPS], GCOL['G1'], GNAME['G1'], 1.0, 'o'),
                ([CM[k]['G0'] for k, _ in STEPS], GCOL['G0'], GNAME['G0'], 1.9, 'o')]
panel(axes[0], size_series, '(a) by object size (box side)', overall_line=False)
panel(axes[1], group_series, '(b) by overlap with the training set', phase_note=False)

# keep end labels from overlapping: push labels up where two series end close together
for ax in axes:
    texts = [t for t in ax.texts if t.get_position()[0] > len(X) - 1]
    texts.sort(key=lambda t: t.get_position()[1])
    ends = [t.get_position()[1] for t in texts]
    for a, b in zip(texts, texts[1:]):
        if b.get_position()[1] - a.get_position()[1] < 0.095:
            b.set_y(a.get_position()[1] + 0.095)
    for t, y0 in zip(texts, ends):   # short leader line where a label had to move away from its line end
        t.set_x(X[-1] + 0.32)
        ax.plot([X[-1] + 0.06, X[-1] + 0.28], [y0, t.get_position()[1]], color=INK2, lw=0.4, clip_on=False)

fig.savefig(f'{OUT}/fig3_journey.pdf')
fig.savefig(f'{OUT}/fig3_journey.png', dpi=170)
print('built', f'{OUT}/fig3_journey.pdf')
