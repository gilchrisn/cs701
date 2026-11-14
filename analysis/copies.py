"""Label consistency between copies of the same scene: for every held-out image of scene group G1 / G2 (holdout_3group_split_2026-10-30),
find its training copy, align it, map the copy's true boxes into the held-out frame and compare them with the held-out true boxes.
Candidates per held-out image: the training images named by the earlier same-scene search (scratch agentC: robust.pkl 'res' = ORB+RANSAC
best match, robust_mirror.pkl 'orbflip' = the same on the h-flipped query, twins.npz 'am_ho' = best gradient-thumbnail cosine).
Alignment: each of the 8 rotations/flips of the candidate, resized to the held-out size, translated by the phase-correlation peak;
score = Pearson correlation of the overlapping pixels; best (candidate, transform) kept if correlation >= 0.6 and overlap >= 25%.
Boxes map exactly (continuous pixel edges: h-flip x -> W - x, etc.). Per held-out true box inside the overlap (>= 50% of its area):
label_iou = best IoU with a mapped copy box of the same class (0 if none). Per mapped copy box inside the overlap with no held-out box
of its class at IoU >= 0.1: an 'extra' label (the copy labels something the held-out image does not).
Self-test first: random images and boxes pushed through all 8 transforms and back must come back exact (copies_selftest.txt).
Usage: python copies.py -> copies_boxes.csv, copies_pairs.csv, copies_summary.txt"""
import csv
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget as B  # noqa: E402

# Paths (changed 2026-11-15): DATA_ROOT holds holdout15_seed0/ (default ./data); COPIES_CANDIDATES is the folder with the candidate lists robust.pkl,
# robust_mirror.pkl, twins.npz of the earlier same-scene search (NOT in this repo: the folder was a scratch folder and is gone; the stored
# output of that run is analysis/results/copies_pairs.csv); COPIES_OUT is where outputs go (default analysis/results).
HERE = os.environ.get('COPIES_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'))
DATA = os.path.join(os.path.expanduser(os.environ.get('DATA_ROOT', 'data')), 'holdout15_seed0')
AC = os.path.expanduser(os.environ.get('COPIES_CANDIDATES', 'candidates'))


def img_t(a, v):  # 8 dihedral transforms of an (H, W) array
    return [a, a[:, ::-1], a[::-1], a[::-1, ::-1], a.T, np.rot90(a), np.rot90(a, 3), a[::-1, ::-1].T][v]


def pt_t(x, y, W, H, v):  # where continuous point (x, y) of a W x H image lands under img_t(., v)
    return [(x, y), (W - x, y), (x, H - y), (W - x, H - y), (y, x), (y, W - x), (H - y, x), (H - y, W - x)][v]


def box_t(b, W, H, v):  # xywh -> xywh
    x, y, w, h = b
    p = [pt_t(px, py, W, H, v) for px, py in ((x, y), (x + w, y + h))]
    x0, x1 = sorted((p[0][0], p[1][0]))
    y0, y1 = sorted((p[0][1], p[1][1]))
    return [x0, y0, x1 - x0, y1 - y0]


def selftest():
    rng = np.random.default_rng(0)
    lines = []
    for v in range(8):
        a = rng.random((40, 60))
        x, y, w, h = 7, 11, 13, 5
        m = np.zeros_like(a)
        m[y:y + h, x:x + w] = 1
        tb = box_t([x, y, w, h], 60, 40, v)
        tm = img_t(m, v)
        ys, xs = np.nonzero(tm)
        ok = [xs.min(), ys.min(), xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min()] == [round(q) for q in tb]
        lines.append(f'transform {v}: box maps onto the moved pixels: {ok}')
        assert ok
    return lines


def load(path, size=None):
    im = Image.open(path).convert('L')
    if size is not None and im.size != size:
        im = im.resize(size, Image.BILINEAR)
    return np.asarray(im, dtype=np.float64)


def phase_shift(a, b):
    """(dy, dx) such that b shifted by (dy, dx) best matches a (circular phase correlation)."""
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    A, Bf = np.fft.fft2((a - a.mean()) * win), np.fft.fft2((b - b.mean()) * win)
    r = A * np.conj(Bf)
    r /= np.abs(r) + 1e-9
    c = np.fft.ifft2(r).real
    dy, dx = np.unravel_index(np.argmax(c), c.shape)
    H, W = a.shape
    return (dy - H if dy > H // 2 else dy), (dx - W if dx > W // 2 else dx)


def overlap_corr(a, b, dy, dx):
    H, W = a.shape
    ya, yb = max(0, dy), max(0, -dy)
    xa, xb = max(0, dx), max(0, -dx)
    h, w = H - abs(dy), W - abs(dx)
    if h < 8 or w < 8:
        return -1.0, 0.0
    pa, pb = a[ya:ya + h, xa:xa + w].ravel(), b[yb:yb + h, xb:xb + w].ravel()
    if pa.std() < 1e-6 or pb.std() < 1e-6:
        return -1.0, 0.0
    return float(np.corrcoef(pa, pb)[0, 1]), h * w / (H * W)


def main():
    lines = selftest()
    rob = pickle.load(open(f'{AC}/robust.pkl', 'rb'))
    mir = pickle.load(open(f'{AC}/robust_mirror.pkl', 'rb'))
    tw = np.load(f'{AC}/twins.npz', allow_pickle=True)
    paths, ids, split = rob['paths'], rob['ids'], rob['split']
    tr_paths, tr_ids = tw['tr_paths'], tw['tr_ids']
    ho_index = {int(i): k for k, (i, s) in enumerate(zip(ids, split)) if s == 'holdout' or s == 'ho' or s == 'held'}
    if not ho_index:  # split labels other than 'train'/'val' mark held-out
        ho_index = {int(i): k for k, (i, s) in enumerate(zip(ids, split)) if s not in ('train', 'val')}
    ho_ids = list(tw['ho_ids'])
    train = json.load(open(f'{DATA}/detection/instances_train.json'))
    tr_info = {i['id']: i for i in train['images']}
    tr_boxes = defaultdict(list)
    for a in train['annotations']:
        tr_boxes[a['image_id']].append(a)
    path_to_trid = {os.path.basename(p): int(i) for p, i in zip(paths, ids) if True}
    ho_info = {i['id']: i for i in B.gt['images']}
    rows, pairs = [], []
    for img in B.IMGS:
        if B.GROUP[img] not in ('G1', 'G2'):
            continue
        info = ho_info[img]
        if info['width'] != info['height']:
            continue
        k = ho_index.get(img)
        cands = set()
        if k is not None and k in rob['res'] and rob['res'][k][1] >= 0:
            cands.add(int(rob['res'][k][1]))
        if k is not None and k in mir['orbflip'] and mir['orbflip'][k][2] >= 0:
            cands.add(int(mir['orbflip'][k][2]))
        cand_tr = set()
        for c in cands:  # robust indices point into the combined list; keep training images
            if split[c] == 'train':
                cand_tr.add(int(ids[c]))
        if img in ho_ids:
            cand_tr.add(int(tr_ids[int(tw['am_ho'][ho_ids.index(img)])]))
        A = load(f"{DATA}/{info['file_name']}")
        Wa = info['width']
        best = None
        for tid in cand_tr:
            ti = tr_info[tid]
            if ti['width'] != ti['height']:
                continue
            Bimg = load(f"{DATA}/{ti['file_name']}", (Wa, Wa))
            for v in range(8):
                Bv = img_t(Bimg, v)
                dy, dx = phase_shift(A, Bv)
                r, ov = overlap_corr(A, Bv, dy, dx)
                if best is None or r > best[0]:
                    best = (r, ov, tid, v, dy, dx, Wa / ti['width'])
        if best is None:
            continue
        r, ov, tid, v, dy, dx, s = best
        pairs.append([img, B.GROUP[img], B.SOURCE[img], tid, v, dy, dx, round(s, 3), round(r, 3), round(ov, 3)])
        if r < 0.6 or ov < 0.25:
            continue
        ti = tr_info[tid]
        mapped = []
        for a in tr_boxes[tid]:
            x, y, w, h = box_t(a['bbox'], ti['width'], ti['height'], v)
            mapped.append((a['category_id'], [x * s + dx, y * s + dy, w * s, h * s]))
        H = W = Wa
        X0, Y0, X1, Y1 = max(0, dx), max(0, dy), min(W, W + dx), min(H, H + dy)

        def inside(b):
            ix = max(0, min(b[0] + b[2], X1) - max(b[0], X0))
            iy = max(0, min(b[1] + b[3], Y1) - max(b[1], Y0))
            return ix * iy >= 0.5 * b[2] * b[3]

        for j, a in enumerate(B.GTS[img]):
            if not inside(a['bbox']):
                continue
            same = [b for c, b in mapped if c == a['category_id']]
            li = float(B.iou_xywh(a['bbox'], same).max()) if same else 0.0
            rows.append(['gt', img, j, B.NAME[a['category_id']], B.GROUP[img], B.SOURCE[img], tid, v, round(r, 3), round(li, 4),
                         round(float(np.sqrt(a['bbox'][2] * a['bbox'][3])), 1)])
        for c, b in mapped:
            if not inside(b):
                continue
            same = [a['bbox'] for a in B.GTS[img] if a['category_id'] == c]
            li = float(B.iou_xywh(b, same).max()) if same else 0.0
            if li < 0.1:
                rows.append(['extra', img, -1, B.NAME[c], B.GROUP[img], B.SOURCE[img], tid, v, round(r, 3), round(li, 4),
                             round(float(np.sqrt(b[2] * b[3])), 1)])
    with open(f'{HERE}/copies_boxes.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['kind', 'image_id', 'gt_index', 'class', 'group', 'source', 'copy_id', 'transform', 'corr', 'label_iou', 'side'])
        w.writerows(rows)
    with open(f'{HERE}/copies_pairs.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_id', 'group', 'source', 'copy_id', 'transform', 'dy', 'dx', 'scale', 'corr', 'overlap'])
        w.writerows(pairs)
    acc = [p for p in pairs if p[8] >= 0.6 and p[9] >= 0.25]
    lines.append(f'held-out G1/G2 images {sum(B.GROUP[i] in ("G1", "G2") for i in B.IMGS)}; aligned to a training copy (corr >= 0.6, '
                 f'overlap >= 25%): {len(acc)} (G2 {sum(p[1] == "G2" for p in acc)}, G1 {sum(p[1] == "G1" for p in acc)})')
    by = defaultdict(list)
    ex = defaultdict(int)
    for r_ in rows:
        if r_[0] == 'gt':
            by[(r_[3], r_[4])].append(r_[9])
        else:
            ex[(r_[3], r_[4])] += 1
    lines.append('per class x group: n held-out boxes | no copy label (IoU < 0.1) | copy label IoU median | share >= 0.5 / 0.75 / 0.9 '
                 '| extra copy labels')
    for key in sorted(by):
        x = np.array(by[key])
        lines.append(f'  {key[0]:12s} {key[1]}  n {len(x):4d} | none {np.mean(x < 0.1):.2f} | median {np.median(x):.3f} | '
                     f'{np.mean(x >= 0.5):.2f} / {np.mean(x >= 0.75):.2f} / {np.mean(x >= 0.9):.2f} | extra {ex[key]}')
    open(f'{HERE}/copies_summary.txt', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
