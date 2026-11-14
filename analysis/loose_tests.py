"""Mechanism tests on the top 'loose' cells (owners = budget.py). All gains are marginal mAP gains at the baseline (measured), each
divided by the gain of snapping the same owners to their exact true box ('exact').
T1 copy-label oracle (G1/G2 cells): owners whose image is aligned to a training copy (copies_pairs.csv: corr >= 0.6, overlap >= 25%) and
   whose true box has a same-class mapped copy box at IoU >= 0.1 are snapped to that copy box instead. share_label = 1 - copy/exact
   = the part of these owners' loss that even reproducing the training label exactly cannot remove (label inconsistency).
   coverage = exact gain of these owners / exact gain of the whole cell.
T2 systematic bias: images split by id parity; per (class, source) the median relative errors of owners with IoU >= 0.3 (dx/w, dy/h,
   log w ratio, log h ratio) fitted on one half and applied to the cell's owners in the other half (both directions, summed).
   share_bias = corrected gain / exact gain (what a per-class box correction learned from labelled data recovers).
T3 one-pixel precision: owners replaced by their true box with every edge moved by an independent uniform [-1, 1] px error (5 draws,
   mean). share_1px = 1 - gain/exact = the part of the loss that remains even at +-1 px precision.
-> loose_tests.txt"""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget as B  # noqa: E402
import copies as CP  # noqa: E402
from loose_profile import CELLS, gain_replace, owners  # noqa: E402

HERE = os.environ.get('COPIES_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'))  # holds copies_pairs.csv
DATA = os.path.join(os.path.expanduser(os.environ.get('DATA_ROOT', 'data')), 'holdout15_seed0')


def copy_boxes():
    import csv
    import json
    train = json.load(open(f'{DATA}/detection/instances_train.json'))
    info = {i['id']: i for i in train['images']}
    tb = defaultdict(list)
    for a in train['annotations']:
        tb[a['image_id']].append(a)
    out = {}
    for r in csv.DictReader(open(f'{HERE}/copies_pairs.csv')):
        if float(r['corr']) < 0.6 or float(r['overlap']) < 0.25:
            continue
        img, tid, v, dy, dx, s = int(r['image_id']), int(r['copy_id']), int(r['transform']), int(r['dy']), int(r['dx']), float(r['scale'])
        ti = info[tid]
        m = []
        for a in tb[tid]:
            x, y, w, h = CP.box_t(a['bbox'], ti['width'], ti['height'], v)
            m.append((a['category_id'], [x * s + dx, y * s + dy, w * s, h * s]))
        out[img] = m
    return out


def rel_err(p, g):
    pcx, pcy, gcx, gcy = p[0] + p[2] / 2, p[1] + p[3] / 2, g[0] + g[2] / 2, g[1] + g[3] / 2
    return np.array([(pcx - gcx) / g[2], (pcy - gcy) / g[3], np.log(p[2] / g[2]), np.log(p[3] / g[3])])


def correct(p, e):
    cx, cy = p[0] + p[2] / 2, p[1] + p[3] / 2
    w, h = p[2] / np.exp(e[2]), p[3] / np.exp(e[3])
    cx, cy = cx - e[0] * w, cy - e[1] * h
    return [cx - w / 2, cy - h / 2, w, h]


if __name__ == '__main__':
    CB = copy_boxes()
    # T2 bias fit: per (class, source, half) median relative error of owners with IoU >= 0.3, any group
    errs = defaultdict(list)
    for img in B.IMGS:
        owner, owns, kinds, I = B.ASSIGN[img]
        for j, a in enumerate(B.GTS[img]):
            k = owner[j]
            if k >= 0 and kinds[k] == 'owner_same' and I[k, j] >= 0.3:
                errs[(a['category_id'], B.SOURCE[img], img % 2)].append(rel_err(B.DTS[img][k]['bbox'], a['bbox']))
    med = {key: np.median(v, 0) for key, v in errs.items() if len(v) >= 10}
    rng = np.random.default_rng(0)
    lines = []
    for cell in CELLS:
        o = owners(*cell)
        c = {v: k for k, v in B.NAME.items()}[cell[0]]
        exact = gain_replace({(x[1], x[3]): B.GTS[x[1]][x[2]]['bbox'] for x in o})
        res = [f"{' '.join(cell)}: exact gain {exact:.4f}"]
        # T1
        if cell[1] in ('G1', 'G2'):
            sel, cb = [], {}
            for x in o:
                g = B.GTS[x[1]][x[2]]['bbox']
                cand = [b for cc, b in CB.get(x[1], []) if cc == c]
                if cand:
                    ious = B.iou_xywh(g, cand)[0]
                    if ious.max() >= 0.1:
                        sel.append(x)
                        cb[(x[1], x[3])] = cand[int(ious.argmax())]
            ex_s = gain_replace({(x[1], x[3]): B.GTS[x[1]][x[2]]['bbox'] for x in sel})
            cp_s = gain_replace(cb)
            res.append(f"T1 copy label: owners with a copy box {len(sel)}/{len(o)}, coverage {ex_s / exact:.2f}, copy gain {cp_s:.4f} vs "
                       f"exact {ex_s:.4f} -> share_label {1 - cp_s / ex_s:.2f}")
        # T2
        corr_gain, exact_half = 0.0, 0.0
        for half in (0, 1):
            fit = med.get((c, cell[2], 1 - half))
            sel = [x for x in o if x[1] % 2 == half]
            if fit is None or not sel:
                continue
            corr_gain += gain_replace({(x[1], x[3]): correct(B.DTS[x[1]][x[3]]['bbox'], fit) for x in sel})
            exact_half += gain_replace({(x[1], x[3]): B.GTS[x[1]][x[2]]['bbox'] for x in sel})
        fits = [np.round(med[(c, cell[2], h)], 3).tolist() for h in (0, 1) if (c, cell[2], h) in med]
        res.append(f"T2 bias correction: fitted (dx/w, dy/h, log w, log h) per half {fits}; corrected gain {corr_gain:.4f} vs exact "
                   f"{exact_half:.4f} -> share_bias {corr_gain / exact_half if exact_half else float('nan'):.2f}")
        # T3
        g1 = []
        for _ in range(5):
            rep = {}
            for x in o:
                gx, gy, gw, gh = B.GTS[x[1]][x[2]]['bbox']
                e = rng.uniform(-1, 1, 4)
                x0, y0, x1, y1 = gx + e[0], gy + e[1], gx + gw + e[2], gy + gh + e[3]
                rep[(x[1], x[3])] = [x0, y0, max(x1 - x0, 0.5), max(y1 - y0, 0.5)]
            g1.append(gain_replace(rep))
        res.append(f"T3 +-1 px: gain {np.mean(g1):.4f} (sd {np.std(g1):.4f}) -> share_1px {1 - np.mean(g1) / exact:.2f}")
        lines.append('\n  '.join(res))
        print(lines[-1], flush=True)
    open(f'{HERE}/loose_tests.txt', 'w').write('\n'.join(lines) + '\n')
