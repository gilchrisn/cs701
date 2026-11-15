"""Statistics about held-out copies of training images (Sec. 5 of the report).

Inputs: results/copies_pairs.csv (alignment of each held-out near-duplicate / overlapping-crop image to its training copy; see README),
tools/holdout15_seed0_scene_groups.json (overlap group of every held-out image) and the held-out labels (for the per-class shares).
An image counts as 'aligned' if correlation >= 0.6 and overlap >= 25%. A 'whole copy' is aligned with no shift (dx = dy = 0) and
overlap >= 0.95. Transform codes: 0 identity, 1 horizontal flip, 2 vertical flip, 3 rotation by 180 degrees.

Usage: python analysis/copy_stats.py --data data/holdout15_seed0      -> prints the numbers and writes analysis/results/copy_stats.txt"""
import argparse
import csv
import json
import os
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = os.path.join(HERE, '..', 'tools', 'holdout15_seed0_scene_groups.json')
PAIRS = os.path.join(HERE, 'results', 'copies_pairs.csv')
NAME = {'G0': 'no match in train', 'G1': 'overlapping crop', 'G2': 'near-duplicate'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/holdout15_seed0', help='held-out split folder (made by tools/make_holdout.py)')
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'copy_stats.txt'))
    args = ap.parse_args()
    groups = json.load(open(GROUPS))
    pairs = list(csv.DictReader(open(PAIRS)))
    for r in pairs:
        for k in ('corr', 'overlap'):
            r[k] = float(r[k])
        for k in ('dx', 'dy', 'transform'):
            r[k] = int(r[k])
    aligned = [r for r in pairs if r['corr'] >= 0.6 and r['overlap'] >= 0.25]
    shifted = lambda r: r['dx'] != 0 or r['dy'] != 0
    L = []
    n_copy = len(groups['G1']) + len(groups['G2'])
    L.append(f'held-out copies (near-duplicate + overlapping crop images): {n_copy}; aligned (corr >= 0.6, overlap >= 25%): {len(aligned)}')
    for g, tag in (('G2', 'near-duplicates'), ('G1', 'crops')):
        a = [r for r in aligned if r['group'] == g]
        whole = [r for r in a if not shifted(r) and r['overlap'] >= 0.95]
        sh = [r for r in a if shifted(r)]
        L.append(f'{tag}: {len(groups[g])} in the group, {len(a)} aligned, median correlation {np.median([r["corr"] for r in a]):.4f}')
        L.append(f'  whole copies (no shift, overlap >= 0.95): {len(whole)}; by transform: '
                 + ', '.join(f'{k} {v}' for k, v in sorted(Counter(r['transform'] for r in whole).items())))
        L.append(f'  shifted: {len(sh)}'
                 + (f'; median larger shift component {np.median([max(abs(r["dx"]), abs(r["dy"])) for r in sh]):.0f} px, '
                    f'median overlap {np.median([r["overlap"] for r in sh]):.4f}' if sh else ''))
    L.append(f'all aligned copies: median correlation {np.median([r["corr"] for r in aligned]):.4f}')
    # per-class shares of the three groups on the held-out split
    held = json.load(open(os.path.join(args.data, 'detection', 'instances_val.json')))
    cname = {c['id']: c['name'].replace('_', ' ') for c in held['categories']}
    grp = {i: g for g, ids in groups.items() for i in ids}
    imcls = {a['image_id']: cname[a['category_id']] for a in held['annotations']}
    tab = {}
    for i, c in imcls.items():
        tab.setdefault(c, Counter())[grp[i]] += 1
    L.append('share of held-out images per class in each overlap group (%): class | n images | no match | overlapping crop | near-duplicate')
    for c in sorted(tab, key=lambda c: tab[c]['G0'] / sum(tab[c].values())):
        n = sum(tab[c].values())
        L.append(f'  {c:12s} {n:4d} | ' + ' | '.join(f'{100 * tab[c][g] / n:5.1f}' for g in ('G0', 'G1', 'G2')))
    n = sum(sum(t.values()) for t in tab.values())
    L.append(f'  all          {n:4d} | ' + ' | '.join(f'{100 * len(groups[g]) / n:5.1f}' for g in ('G0', 'G1', 'G2'))
             + f'   (counts {len(groups["G0"])} / {len(groups["G1"])} / {len(groups["G2"])})')
    open(args.out, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
