"""Tables 1 and 3 of the report: our stored leaderboard submissions scored on the validation and test labels that the course released
on 13 Nov 2026: macro-F1, COCO mAP (AP50, AP75), Delta m, and mAP per overlap group (no match in train / overlapping crop / near-duplicate).

Labels: download val_instances.json, val_labels.csv, test_instances.json and test_labels.csv from
https://huggingface.co/datasets/doem1997/cs701-sar-course-data into one folder (--labels).
Groups: analysis/results/test_groups.json (made by analysis/overlap_groups.py). Submissions: analysis/results/submissions/*.zip, the
classification.csv + detection.json we uploaded (or would have uploaded) for each step.
Usage: python analysis/score_released.py --labels labels/   -> prints the tables, writes analysis/results/released_scores.txt"""
import argparse
import contextlib
import csv
import io
import json
import os
import zipfile

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--labels', required=True, help='folder with {val,test}_instances.json and {val,test}_labels.csv')
ap.add_argument('--groups', default=os.path.join(HERE, 'results', 'test_groups.json'))
ap.add_argument('--subs', default=os.path.join(HERE, 'results', 'submissions'))
ap.add_argument('--out', default=os.path.join(HERE, 'results', 'released_scores.txt'))
args = ap.parse_args()

REF = {'val': (93.31, 26.13), 'test': (93.99, 25.41)}   # course reference macro-F1 and mAP (Delta m = 0)
ROWS = [  # (row name, split, training images, submission zip)
    ('course model (ViT+LoRA, 24 ep)', 'val', '85%', 'val_course_model_85pct.zip'),
    ('Cascade R-CNN, 48+24 ep', 'val', '85%', 'val_cascade_48+24ep_85pct.zip'),
    ('YOLO26l-P2, mosaic on, 96 ep', 'val', '85%', 'val_yolo_mosaic_on_96ep_85pct.zip'),
    ('YOLO26l-P2, mosaic off, 96 ep', 'val', '85%', 'val_yolo_mosaic_off_96ep_85pct.zip'),
    ('+ KL loss + flip convention, 96 ep', 'val', 'all', 'val_kl_flip_96ep_alldata.zip'),
    ('+ 144 ep (submitted)', 'val', 'all', 'val_submitted_144ep_alldata.zip'),
    ('+ 144 ep (submitted)', 'test', 'all', 'test_submitted_144ep_alldata.zip'),
    ('fine-tune, brightness and shift off (not submitted)', 'test', 'all', 'test_finetune_24ep_not_submitted.zip'),
]
PAIRS = [  # Table 3: (change, split, row before, row after)
    ('detector switch', 'val', 1, 2), ('mosaic and scale jitter off', 'val', 2, 3),
    ('96 -> 144 epochs', 'val', 4, 5), ('brightness and shift off (fine-tune)', 'test', 6, 7)]

G = json.load(open(args.groups))
groups = {'val': {}, 'test': {}}
for s, i, g in zip(G['split'], G['id'], G['group']):
    groups[s][i] = g
gts, labels = {}, {}
for s in ('val', 'test'):
    with contextlib.redirect_stdout(io.StringIO()):
        gts[s] = COCO(os.path.join(args.labels, f'{s}_instances.json'))
    labels[s] = {int(r['image_id']): int(r['label']) for r in csv.DictReader(open(os.path.join(args.labels, f'{s}_labels.csv')))}


def coco(gt, dets, ids):
    dets = [d for d in dets if d['image_id'] in ids]
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, gt.loadRes(dets), 'bbox')
        E.params.imgIds = sorted(ids)
        E.evaluate(); E.accumulate(); E.summarize()
    return 100 * E.stats[0], 100 * E.stats[1], 100 * E.stats[2]


def macro_f1(true, pred):
    f = []
    for c in range(9):
        tp = sum(1 for i in true if true[i] == c and pred.get(i) == c)
        fp = sum(1 for i in true if true[i] != c and pred.get(i) == c)
        fn = sum(1 for i in true if true[i] == c and pred.get(i) != c)
        f.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return 100 * float(np.mean(f))


lines, res = [], []
for name, s, tr, zname in ROWS:
    z = zipfile.ZipFile(os.path.join(args.subs, zname))
    det = json.loads(z.read('detection.json'))
    cls = {int(r['image_id']): int(r['label']) for r in csv.DictReader(io.StringIO(z.read('classification.csv').decode()))}
    ids = set(labels[s])
    m, m50, m75 = coco(gts[s], det, ids)
    f1 = macro_f1(labels[s], cls)
    dm = 50 * ((f1 - REF[s][0]) / REF[s][0] + (m - REF[s][1]) / REF[s][1])
    per = {g: coco(gts[s], det, {i for i in ids if groups[s].get(i) == g})[0] for g in ('G0', 'G1', 'G2')}
    res.append(dict(all=m, **per))
    lines.append(f"{s:4s} {tr:3s} {name:52s} F1 {f1:5.2f}  mAP {m:5.2f}  AP50 {m50:5.2f}  AP75 {m75:5.2f}  dm {dm:6.2f} | "
                 f"no match {per['G0']:5.2f}  crop {per['G1']:5.2f}  near-dup {per['G2']:5.2f}")
lines.append('')
lines.append('change in mAP (points): all / no match / crop / near-duplicate')
for name, s, a, b in PAIRS:
    d = {k: res[b][k] - res[a][k] for k in ('all', 'G0', 'G1', 'G2')}
    lines.append(f"{name:40s} {s:4s} {d['all']:+5.1f} {d['G0']:+5.1f} {d['G1']:+5.1f} {d['G2']:+5.1f}")
for s in ('val', 'test'):
    lines.append(f"{s} images per group: " + ', '.join(f"{g} {sum(1 for i in labels[s] if groups[s].get(i) == g)}" for g in ('G0', 'G1', 'G2')))
open(args.out, 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
