"""Shared look and helpers for the report figures (figures/fig*.py): fonts, colours, a figure class that places axes in inches,
image-crop and box helpers. Copied from the plotting code we used for the report; nothing here reads data."""
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
COL, FULL = 3.3, 6.9          # one-column and two-column width of the report, in inches
INK, MUTED, FRAME, GRIDC = '#1F2328', '#5F6368', '#9AA0A6', '#E5E7EB'
INK2 = '#5F6670'
G_COL = {'G0': '#E69F00', 'G1': '#009E73', 'G2': '#CC79A7'}   # no match in train / overlapping crop / near-duplicate
TRUE_C, LIB_C = '#009E73', '#D55E00'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 7,
    'axes.edgecolor': FRAME, 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.5,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2, 'ytick.major.size': 2,
    'pdf.fonttype': 42, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
})


def out_dir(path=None):
    d = path or os.path.join(HERE, 'out')
    os.makedirs(d, exist_ok=True)
    return d


def args(desc, extra=None):
    """common command line: --data (held-out split folder) and --out (where the .pdf/.png go)"""
    import argparse
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument('--data', default=os.path.join(REPO, 'data', 'holdout15_seed0'),
                    help='held-out split folder made by tools/make_holdout.py (default: data/holdout15_seed0 in the repository)')
    ap.add_argument('--out', default=None, help='output folder (default figures/out)')
    for a, kw in (extra or []):
        ap.add_argument(a, **kw)
    return ap.parse_args()


class P:
    """a figure whose axes are placed in inches from the lower left corner"""
    def __init__(self, w, h, out):
        self.W, self.H, self.out = w, h, out
        self.fig = plt.figure(figsize=(w, h))
        self._ov = None

    def ax(self, x, y, w, h, **kw):
        return self.fig.add_axes([x / self.W, y / self.H, w / self.W, h / self.H], **kw)

    def ov(self):
        if self._ov is None:
            a = self.fig.add_axes([0, 0, 1, 1], zorder=20)
            a.set_xlim(0, self.W); a.set_ylim(0, self.H); a.axis('off'); a.patch.set_alpha(0)
            self._ov = a
        return self._ov

    def title(self, x, y, letter, s):
        """panel title: bold '(a)', then regular text, 7.5 pt"""
        o = self.ov()
        t = o.text(x, y, letter, fontsize=7.5, fontweight='bold', ha='left', va='baseline')
        self.fig.canvas.draw()
        bb = t.get_window_extent(self.fig.canvas.get_renderer())
        w = bb.width / self.fig.dpi
        o.text(x + w + 0.045, y, s, fontsize=7.5, ha='left', va='baseline')

    def save(self, name):
        self.fig.savefig(f'{self.out}/{name}.pdf')
        self.fig.savefig(f'{self.out}/{name}.png', dpi=200)
        plt.close(self.fig)
        print('built', f'{self.out}/{name}.pdf')


def thin(ax):
    for s in ax.spines.values():
        s.set_visible(True); s.set_edgecolor(FRAME); s.set_linewidth(0.5)


def load_coco(path):
    d = json.load(open(path))
    info = {i['id']: i for i in d['images']}
    by = {}
    for a in d['annotations']:
        by.setdefault(a['image_id'], []).append(a)
    return d, info, by


def gray(path):
    return np.asarray(Image.open(path).convert('L'), dtype=float)


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / (a[2] * a[3] + b[2] * b[3] - i)


def bx(ax, b, col, lw=2.2, ls='-', z=5):
    ax.add_patch(Rectangle((b[0], b[1]), b[2], b[3], fill=False, ec=col, lw=lw, ls=ls, zorder=z))


def show(ax, g, x0, y0, x1, y1, stretch=True, vmin=0, vmax=255, frame=INK2):
    """nearest-neighbour crop of a grey image in image-pixel coordinates (pixel i covers [i, i+1])."""
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    c = g[y0:y1, x0:x1]
    if stretch:
        vmin, vmax = np.percentile(c, 1), max(np.percentile(c, 99.7), np.percentile(c, 1) + 8)
    ax.imshow(c, cmap='gray', vmin=vmin, vmax=vmax, extent=(x0, x1, y1, y0), interpolation='nearest')
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_edgecolor(frame); s.set_linewidth(1.2)


def grid(ax, x0, y0, x1, y1, step, col='white', lw=0.9, alpha=0.75, z=4):
    for t in np.arange(np.ceil(x0 / step) * step, x1 + 1e-9, step):
        ax.plot([t, t], [y0, y1], color=col, lw=lw, alpha=alpha, zorder=z)
    for t in np.arange(np.ceil(y0 / step) * step, y1 + 1e-9, step):
        ax.plot([x0, x1], [t, t], color=col, lw=lw, alpha=alpha, zorder=z)


def read_pairs():
    import csv
    return list(csv.DictReader(open(os.path.join(RES, 'copies_pairs.csv'))))
