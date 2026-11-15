"""Label consistency between held-out images and their training copies, and the 'copy ceiling' (Sec. 5 of the report).

Part 1 (label consistency). Every aligned held-out image (results/copies_pairs.csv: correlation >= 0.6, overlap >= 25%, groups
near-duplicate G2 and overlapping crop G1) is paired with its training copy. The copy's boxes are mapped into the held-out frame by
the stored transform (box_t of flip_convention.py: continuous pixel edges), then x * scale + dx, y * scale + dy. For every held-out box
whose area lies at least 50% inside the overlap, label IoU = best IoU with a mapped copy box of the same class (0 if there is none).
Printed: per class and group, the number of boxes and the median label IoU.

Part 2 (copy ceiling). On the held-out near-duplicates that are unshifted, unrotated copies (group G2, transform 0, dx = dy = 0,
overlap >= 0.95; 276 images), the training copy's own boxes (score 1) are scored as predictions with COCO mAP, next to the
model predictions on the same images. Label transfer is not allowed at test time; this is only a ceiling for 'memorise the copies better'.

Usage: python analysis/copy_labels.py --data data/holdout15_seed0 --preds runs/final85_eval/predictions_val.json --preds-ft <file>
  --preds     COCO predictions of the 144-epoch 85% model on the held-out images (tools/score_yolo_holdout.py writes it; for the
              0.6975 in the report use --conf 0.00001 when scoring; the default --conf 0.001 gives 0.6971)
  --preds-ft  optional, predictions of the exact-copy fine-tune (0.7674 in the report)"""
import argparse
import contextlib
import csv
import io
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flip_convention import box_t, iou_xywh  # noqa: E402

PAIRS = os.path.join(HERE, 'results', 'copies_pairs.csv')


def load_pairs():
    return [r for r in csv.DictReader(open(PAIRS)) if float(r['corr']) >= 0.6 and float(r['overlap']) >= 0.25]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/holdout15_seed0')
    ap.add_argument('--preds', help='predictions_val.json of the 144-epoch 85% model')
    ap.add_argument('--preds-ft', help='predictions_val.json of the exact-copy fine-tune')
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'copy_labels.txt'))
    args = ap.parse_args()
    D = os.path.join(args.data, 'detection')
    held = json.load(open(os.path.join(D, 'instances_val.json')))
    train = json.load(open(os.path.join(D, 'instances_train.json')))
    cname = {c['id']: c['name'].replace('_', ' ') for c in held['categories']}
    tinfo = {i['id']: i for i in train['images']}
    hinfo = {i['id']: i for i in held['images']}
    tbox, hbox = defaultdict(list), defaultdict(list)
    for a in train['annotations']:
        tbox[a['image_id']].append(a)
    for a in held['annotations']:
        hbox[a['image_id']].append(a)

    pairs = load_pairs()
    mapped = {}   # held-out image -> mapped copy boxes (category, xywh)
    li = defaultdict(list)
    for r in pairs:
        img, tid, v = int(r['image_id']), int(r['copy_id']), int(r['transform'])
        dy, dx, s = int(r['dy']), int(r['dx']), float(r['scale'])
        ti = tinfo[tid]
        W = H = hinfo[img]['width']
        m = []
        for a in tbox[tid]:
            x, y, w, h = box_t(a['bbox'], ti['width'], ti['height'], v)
            m.append((a['category_id'], [x * s + dx, y * s + dy, w * s, h * s]))
        mapped[img] = m
        X0, Y0, X1, Y1 = max(0, dx), max(0, dy), min(W, W + dx), min(H, H + dy)
        for a in hbox[img]:
            b = a['bbox']
            ix = max(0, min(b[0] + b[2], X1) - max(b[0], X0))
            iy = max(0, min(b[1] + b[3], Y1) - max(b[1], Y0))
            if ix * iy < 0.5 * b[2] * b[3]:
                continue
            same = [bb for c, bb in m if c == a['category_id']]
            li[(cname[a['category_id']], r['group'])].append(float(iou_xywh(b, same).max()) if same else 0.0)
    L = ['label IoU between held-out boxes and mapped training-copy boxes: class | group | n boxes | median label IoU']
    for k in sorted(li):
        L.append(f'  {k[0]:12s} {k[1]}  n {len(li[k]):4d} | median {np.median(li[k]):.3f}')

    # ---- copy ceiling
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    q = lambda: contextlib.redirect_stdout(io.StringIO())
    with q():
        gt = COCO(os.path.join(D, 'instances_val.json'))
    whole = [r for r in pairs if r['group'] == 'G2' and r['transform'] == '0' and r['dx'] == '0' and r['dy'] == '0'
             and float(r['overlap']) >= 0.95]
    ids = sorted(int(r['image_id']) for r in whole)
    idset = set(ids)
    copy = [{'image_id': int(r['image_id']), 'category_id': a['category_id'], 'bbox': a['bbox'], 'score': 1.0}
            for r in whole for a in tbox[int(r['copy_id'])]]

    def mAP(dets):
        with q():
            e = COCOeval(gt, gt.loadRes(dets), 'bbox')
            e.params.imgIds = ids
            e.evaluate(); e.accumulate(); e.summarize()
        return e.stats[0]
    L.append(f'copy ceiling on {len(ids)} unshifted, unrotated near-duplicate copies (held-out COCO mAP):')
    L.append(f'  training copy boxes as predictions: {mAP(copy):.4f}')
    for name, path in (('144-epoch 85% model', args.preds), ('exact-copy fine-tune', args.preds_ft)):
        if path:
            L.append(f'  {name}: {mAP([d for d in json.load(open(path)) if d["image_id"] in idset]):.4f}')
    open(args.out, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
