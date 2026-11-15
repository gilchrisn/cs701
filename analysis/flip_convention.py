"""Do flipped copies carry a fixed 1-px label offset (the dataset's flip formula) relative to the exact geometric flip?

For every held-out image aligned to a training copy (results/copies_pairs.csv, kept if correlation >= 0.6 and overlap >= 25%), the
copy's true boxes are mapped into the held-out frame with the exact geometric transform (h-flip: x -> W - x). Each held-out true box
is matched to the mapped copy box of its class with the highest IoU (kept if IoU >= 0.3), and the signed offset of each edge
(held-out box minus mapped copy box) is recorded. Output, per transform: the median offset of each edge (x0, y0, x1, y1) and the share
of edges that are exactly -1 / 0 / +1 px off (within 0.05 px). A share near 0.8 at -1 for the left and right edges of h-flipped copies
means the dataset flips boxes with x' = W - 1 - x.

Usage: python analysis/flip_convention.py --data data/holdout15_seed0      -> analysis/results/flip_convention.txt
Needs only the held-out split (detection/instances_train.json and instances_val.json) and the stored copies_pairs.csv."""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
NAMES = {0: 'identity', 1: 'h-flip', 2: 'v-flip', 3: 'rot180'}


def pt_t(x, y, W, H, v):  # where continuous point (x, y) of a W x H image lands under dihedral transform v
    return [(x, y), (W - x, y), (x, H - y), (W - x, H - y), (y, x), (y, W - x), (H - y, x), (H - y, W - x)][v]


def box_t(b, W, H, v):  # xywh -> xywh under transform v
    x, y, w, h = b
    p = [pt_t(px, py, W, H, v) for px, py in ((x, y), (x + w, y + h))]
    x0, x1 = sorted((p[0][0], p[1][0]))
    y0, y1 = sorted((p[0][1], p[1][1]))
    return [x0, y0, x1 - x0, y1 - y0]


def iou_xywh(a, b):  # one box a against boxes b, xywh, as pycocotools (no +1)
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0, None], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1, None], b[None, :, 1]), 0, None)
    inter = iw * ih
    return (inter / (a[:, 2:3] * a[:, 3:4] + (b[:, 2] * b[:, 3])[None] - inter))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/holdout15_seed0', help='held-out split folder (made by tools/make_holdout.py)')
    ap.add_argument('--out', default=os.path.join(RES, 'flip_convention.txt'))
    args = ap.parse_args()

    train = json.load(open(os.path.join(args.data, 'detection', 'instances_train.json')))
    held = json.load(open(os.path.join(args.data, 'detection', 'instances_val.json')))
    info = {i['id']: i for i in train['images']}
    train_boxes, held_boxes = defaultdict(list), defaultdict(list)
    for a in train['annotations']:
        train_boxes[a['image_id']].append(a)
    for a in held['annotations']:
        held_boxes[a['image_id']].append(a)

    off = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(RES, 'copies_pairs.csv'))):
        if float(r['corr']) < 0.6 or float(r['overlap']) < 0.25:
            continue
        img, tid, v = int(r['image_id']), int(r['copy_id']), int(r['transform'])
        if v not in NAMES:
            continue
        dy, dx, s = int(r['dy']), int(r['dx']), float(r['scale'])
        ti = info[tid]
        mapped = []
        for a in train_boxes[tid]:
            x, y, w, h = box_t(a['bbox'], ti['width'], ti['height'], v)
            mapped.append((a['category_id'], [x * s + dx, y * s + dy, w * s, h * s]))
        for a in held_boxes[img]:
            cand = [b for c, b in mapped if c == a['category_id']]
            if not cand:
                continue
            ious = iou_xywh(a['bbox'], cand)
            if ious.max() < 0.3:
                continue
            g, b = a['bbox'], cand[int(ious.argmax())]
            off[v].append([g[0] - b[0], g[1] - b[1], (g[0] + g[2]) - (b[0] + b[2]), (g[1] + g[3]) - (b[1] + b[3])])

    lines = ['signed edge offsets held-out minus geometric copy (x0, y0, x1, y1), median | share exactly -1 / 0 / +1 px per edge (measured)']
    for v in sorted(off):
        x = np.array(off[v])
        sh = lambda t: ' '.join(f'{np.mean(np.abs(x[:, e] - t) <= 0.05):.2f}' for e in range(4))
        lines.append(f'  {NAMES[v]:8s} n {len(x):4d}: median {np.round(np.median(x, 0), 2).tolist()} | -1: {sh(-1)} | 0: {sh(0)} | +1: {sh(1)}')
    open(args.out, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
