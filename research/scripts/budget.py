"""Error budget of the best run (YOLO26l-P2, 96 ep, mosaic + scale off; 2026-10-28_yolo26lp2_96ep_nomosaic_noscale_s0, score >= 0.001)
on the held-out split: the whole gap 1 - mAP split over cells (class x error type x scene group x source), exactly additive.

Ownership (per image, TIDE-like, t_b = 0.1): pass 1, detections in score order take the unowned true box of THEIR class with the
highest IoU >= 0.1; pass 2, remaining detections in score order take an unowned true box of another class with IoU >= 0.1.
Error types and their oracle fixes:
  wrong_class  owner of another class            -> relabelled to the true class
  loose        owner (any class)                 -> box replaced by the true box
  false_alarm  detection owning nothing          -> removed  (sub-kinds kept for analysis: duplicate = IoU >= 0.1 with an owned
                                                    true box of its class; background = the rest)
  missed       true box without an owner         -> a detection with the true box, true class, score 2.0 is added
All four fixes together give AP 1 in every class. A cell = (true class, or the detection's class for false alarms) x type x scene group
(G0 new / G1 shifted copy / G2 near-identical copy, runs/holdout_3group_split_2026-10-30.json) x source (jpg512 incl. one 486x283
JPG / jpg256 / png = 256 and 224 px PNG). The loss of a cell is its Shapley value for the game v(S) = mAP with the fixes of the cells
in S applied: its marginal mAP gain averaged over random orders (Monte Carlo, antithetic pairs). In every order the gains sum to
v(all) - v(none) = 1 - mAP, so the budget covers 100% of the gap; only the split carries Monte Carlo error (SE reported).
AP is computed exactly as pycocotools COCOeval (bbox, area all, maxDets 100, 10 IoU thresholds, 101 recall points); checked
against COCOeval on the baseline and on modified prediction sets (check.txt).
Usage: python budget.py <n_permutation_pairs> <n_processes>"""
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

R = os.path.expanduser('~/cs701/runs')
OUT = os.environ.get('BUDGET_OUT', f'{R}/error_budget_2026-10-30')  # env overrides added 2026-10-31 for budget 2; defaults unchanged
GT = os.path.expanduser('~/cs701/data/holdout15_seed0/detection/instances_val.json')
PRED = os.environ.get('BUDGET_PRED', f'{R}/2026-10-28_yolo26lp2_96ep_nomosaic_noscale_s0/eval_last_conf0.001/predictions_val.json')
SPLIT = f'{R}/holdout_3group_split_2026-10-30.json'
TYPES = ('wrong_class', 'loose', 'false_alarm', 'missed')
THR = np.linspace(0.5, 0.95, 10)
REC = np.linspace(0, 1, 101)

gt = json.load(open(GT))
CATS = sorted(c['id'] for c in gt['categories'])
NAME = {c['id']: c['name'] for c in gt['categories']}
IMGS = sorted(i['id'] for i in gt['images'])
GROUP = {i: g for g, ids in json.load(open(SPLIT)).items() for i in ids}
SOURCE = {}
for im in gt['images']:
    ext = im['file_name'].rsplit('.', 1)[1]
    SOURCE[im['id']] = 'png' if ext == 'png' else ('jpg256' if im['width'] == 256 else 'jpg512')
GTS = defaultdict(list)
for a in gt['annotations']:
    GTS[a['image_id']].append(a)
DTS = defaultdict(list)
for d in json.load(open(PRED)):
    DTS[d['image_id']].append(d)


def iou_xywh(a, b):  # a (n,4), b (m,4) xywh -> (n,m), as pycocotools (no +1)
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0, None], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1, None], b[None, :, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:3] * a[:, 3:4] + (b[:, 2] * b[:, 3])[None] - inter)


def assign(img):
    """Owner of each true box and type of each detection for one image."""
    d, g = DTS[img], GTS[img]
    order = sorted(range(len(d)), key=lambda k: -d[k]['score'])
    I = iou_xywh([x['bbox'] for x in d], [x['bbox'] for x in g]) if d and g else np.zeros((len(d), len(g)))
    owner = [-1] * len(g)
    owns = [-1] * len(d)
    for same in (True, False):
        for k in order:
            if owns[k] >= 0:
                continue
            best, bj = 0.1, -1
            for j, a in enumerate(g):
                if owner[j] >= 0 or (a['category_id'] == d[k]['category_id']) != same:
                    continue
                if I[k, j] >= best:
                    best, bj = I[k, j], j
            if bj >= 0:
                owner[bj], owns[k] = k, bj
    kinds = []
    for k, x in enumerate(d):
        if owns[k] >= 0:
            kinds.append('owner_same' if g[owns[k]]['category_id'] == x['category_id'] else 'owner_wrong')
        else:
            dup = any(I[k, j] >= 0.1 and a['category_id'] == x['category_id'] for j, a in enumerate(g))
            kinds.append('fa_duplicate' if dup else 'fa_background')
    return owner, owns, kinds, I


ASSIGN = {i: assign(i) for i in IMGS}


def cell(img, cls, typ):
    return (cls, typ, GROUP[img], SOURCE[img])


def image_dets(img, active):
    """Modified detection list of one image under the set of active cells."""
    d, g = DTS[img], GTS[img]
    owner, owns, kinds, _ = ASSIGN[img]
    out = []
    for k, x in enumerate(d):
        if kinds[k].startswith('fa'):
            if cell(img, x['category_id'], 'false_alarm') in active:
                continue
            out.append((x['category_id'], x['bbox'], x['score']))
        else:
            a = g[owns[k]]
            c = a['category_id'] if kinds[k] == 'owner_same' or cell(img, a['category_id'], 'wrong_class') in active else x['category_id']
            b = a['bbox'] if cell(img, a['category_id'], 'loose') in active else x['bbox']
            out.append((c, b, x['score']))
    for j, a in enumerate(g):
        if owner[j] < 0 and cell(img, a['category_id'], 'missed') in active:
            out.append((a['category_id'], a['bbox'], 2.0))
    return out


def match(dets, gts):
    """COCOeval.evaluateImg for one image and class (area all, maxDets 100): scores and (n, 10) TP flags."""
    idx = np.argsort([-s for _, _, s in dets], kind='mergesort')[:100]
    dets = [dets[i] for i in idx]
    if not dets:
        return np.zeros(0), np.zeros((0, 10), bool)
    ious = iou_xywh([b for _, b, _ in dets], gts) if gts else np.zeros((len(dets), 0))
    tp = np.zeros((len(dets), 10), bool)
    for t, thr in enumerate(THR):
        used = np.zeros(len(gts), bool)
        for di in range(len(dets)):
            iou, m = min(thr, 1 - 1e-10), -1
            for gi in range(len(gts)):
                if used[gi] or ious[di, gi] < iou:
                    continue
                iou, m = ious[di, gi], gi
            if m >= 0:
                used[m], tp[di, t] = True, True
    return np.array([s for _, _, s in dets]), tp


def ap(per_image, npos):
    """COCOeval.accumulate for one class: per_image = list (image order) of (scores, tp)."""
    s = np.concatenate([p[0] for p in per_image])
    tp = np.concatenate([p[1] for p in per_image])
    o = np.argsort(-s, kind='mergesort')
    tp = tp[o]
    out = []
    for t in range(10):
        tps, fps = np.cumsum(tp[:, t]), np.cumsum(~tp[:, t])
        rc, pr = tps / npos, tps / (fps + tps + np.spacing(1))
        pr = np.maximum.accumulate(pr[::-1])[::-1]
        q = np.zeros(len(REC))
        inds = np.searchsorted(rc, REC, side='left')
        ok = inds < len(pr)
        q[ok] = pr[inds[ok]]
        out.append(q.mean())
    return float(np.mean(out))


CACHE = {}
NPOS = {c: sum(a['category_id'] == c for a in gt['annotations']) for c in CATS}


class Game:
    """Incremental v(S): per (image, class) matched arrays, recomputed only where a toggled cell acts."""

    def __init__(self, base=None):
        self.active = set()
        if base is not None:  # start from a copy of an empty-set game
            self.cur = {c: list(v) for c, v in base.cur.items()}
            self.aps = dict(base.aps)
            return
        self.cur = {c: [None] * len(IMGS) for c in CATS}
        for n, img in enumerate(IMGS):
            self._update(n, img)
        self.aps = {c: ap(self.cur[c], NPOS[c]) for c in CATS}

    def _update(self, n, img):
        dets = image_dets(img, self.active)
        byc = defaultdict(list)
        for c, b, s in dets:
            byc[c].append((c, b, s))
        changed = set()
        for c in CATS:
            key = (img, c, tuple(sorted((d[1][0], d[1][1], d[1][2], d[1][3], d[2]) for d in byc[c])))
            if key not in CACHE:
                CACHE[key] = match(byc[c], [a['bbox'] for a in GTS[img] if a['category_id'] == c])
            if self.cur[c][n] is not CACHE[key]:
                self.cur[c][n] = CACHE[key]
                changed.add(c)
        return changed

    def toggle(self, p, images):
        self.active.add(p)
        changed = set()
        for n, img in images:
            changed |= self._update(n, img)
        for c in changed:
            self.aps[c] = ap(self.cur[c], NPOS[c])
        return self.value()

    def value(self):
        return float(np.mean([self.aps[c] for c in CATS]))


def players():
    P = defaultdict(set)
    for n, img in enumerate(IMGS):
        owner, owns, kinds, _ = ASSIGN[img]
        d, g = DTS[img], GTS[img]
        for k, x in enumerate(d):
            if kinds[k].startswith('fa'):
                P[cell(img, x['category_id'], 'false_alarm')].add((n, img))
            else:
                a = g[owns[k]]
                if kinds[k] == 'owner_wrong':
                    P[cell(img, a['category_id'], 'wrong_class')].add((n, img))
                if [a['bbox'][q] for q in range(4)] != [x['bbox'][q] for q in range(4)]:
                    P[cell(img, a['category_id'], 'loose')].add((n, img))
        for j, a in enumerate(g):
            if owner[j] < 0:
                P[cell(img, a['category_id'], 'missed')].add((n, img))
    return {p: sorted(v) for p, v in P.items()}


def run_pairs(args):
    seed, npairs = args
    rng = np.random.default_rng(seed)
    P = players()
    keys = sorted(P)
    acc = np.zeros(len(keys))
    acc2 = np.zeros(len(keys))
    count = 0
    base = Game()
    for _ in range(npairs):
        perm = rng.permutation(len(keys))
        for order in (perm, perm[::-1]):  # antithetic pair
            game = Game(base)
            v = game.value()
            marg = np.zeros(len(keys))
            for i in order:
                v2 = game.toggle(keys[i], P[keys[i]])
                marg[i] = v2 - v
                v = v2
            acc += marg
            acc2 += marg ** 2
            count += 1
    return acc, acc2, count, keys


if __name__ == '__main__':
    npairs, nproc = int(sys.argv[1]), int(sys.argv[2])
    with Pool(nproc) as pool:
        res = pool.map(run_pairs, [(s, npairs) for s in range(nproc)])
    keys = res[0][3]
    acc = sum(r[0] for r in res)
    acc2 = sum(r[1] for r in res)
    n = sum(r[2] for r in res)
    mean = acc / n
    se = np.sqrt(np.maximum(acc2 / n - mean ** 2, 0) / n)
    base = Game().value()
    rows = sorted(zip(keys, mean, se), key=lambda r: -r[1])
    with open(f'{OUT}/budget_cells.csv', 'w') as f:
        f.write('class,type,group,source,loss_map,se,share_of_gap\n')
        for (c, t, g, s), m, e in rows:
            f.write(f'{NAME[c]},{t},{g},{s},{m:.6f},{e:.6f},{m / (1 - base):.6f}\n')
    print(f'baseline mAP {base:.4f}; gap {1 - base:.4f}; sum of cell losses {mean.sum():.4f}; permutations {n}; cells {len(keys)}')
