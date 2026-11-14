"""Profile of the top 'loose' cells: owners' IoU with their true box, box size, and the marginal mAP gain (at the baseline) of snapping
only owners with IoU < 0.5 vs only owners with IoU >= 0.5 (measured). -> loose_profile.txt"""
import os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget as B

CELLS = [('ship', 'G0', 'jpg256'), ('wind_turbine', 'G2', 'jpg512'), ('car', 'G0', 'jpg512'), ('aircraft', 'G1', 'jpg512'),
         ('bridge', 'G0', 'jpg512'), ('oil_tank', 'G2', 'jpg512'), ('wind_turbine', 'G1', 'jpg512'), ('harbor', 'G2', 'jpg512'),
         ('car', 'G1', 'jpg512'), ('airport', 'G2', 'jpg512'), ('airport', 'G0', 'jpg512'), ('playground', 'G0', 'jpg512'), ('aircraft', 'G0', 'png')]
ID = {v: k for k, v in B.NAME.items()}


def owners(cname, grp, src):
    out = []
    for n, img in enumerate(B.IMGS):
        if B.GROUP[img] != grp or B.SOURCE[img] != src:
            continue
        owner, owns, kinds, I = B.ASSIGN[img]
        for j, a in enumerate(B.GTS[img]):
            k = owner[j]
            if a['category_id'] == ID[cname] and k >= 0:
                out.append((n, img, j, k, float(I[k, j]), float(np.sqrt(a['bbox'][2] * a['bbox'][3]))))
    return out


def gain_replace(repl):
    """repl: {(img, det index): new box}; marginal mAP gain at the baseline (classes as predicted / relabel not applied)."""
    g = B.Game(); base = g.value(); by = defaultdict(dict)
    for (img, k), b in repl.items():
        by[img][k] = b
    changed = set()
    for n, img in enumerate(B.IMGS):
        if img not in by:
            continue
        dets = [(x['category_id'], by[img].get(k, x['bbox']), x['score']) for k, x in enumerate(B.DTS[img])]
        for c in B.CATS:
            g.cur[c][n] = B.match([d for d in dets if d[0] == c], [a['bbox'] for a in B.GTS[img] if a['category_id'] == c]); changed.add(c)
    for c in changed:
        g.aps[c] = B.ap(g.cur[c], B.NPOS[c])
    return g.value() - base


if __name__ == '__main__':
    lines = []
    for cell in CELLS:
        o = owners(*cell)
        iou = np.array([x[4] for x in o]); side = np.array([x[5] for x in o])
        snap = lambda sel: gain_replace({(x[1], x[3]): B.GTS[x[1]][x[2]]['bbox'] for x in sel})
        g_all, g_lo, g_hi = snap(o), snap([x for x in o if x[4] < 0.5]), snap([x for x in o if x[4] >= 0.5])
        lines.append(f"{' '.join(cell)}: owners {len(o)}; IoU median {np.median(iou):.3f}, <0.5 {np.mean(iou < 0.5):.2f}, 0.5-0.75 "
                     f"{np.mean((iou >= 0.5) & (iou < 0.75)):.2f}, 0.75-0.9 {np.mean((iou >= 0.75) & (iou < 0.9)):.2f}, >=0.9 {np.mean(iou >= 0.9):.2f}; "
                     f"side median {np.median(side):.1f} px | snap gain all {g_all:.4f}: IoU<0.5 owners {g_lo:.4f} ({g_lo / g_all:.2f}), "
                     f"IoU>=0.5 owners {g_hi:.4f} ({g_hi / g_all:.2f}) (measured)")
    open('loose_profile.txt', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
