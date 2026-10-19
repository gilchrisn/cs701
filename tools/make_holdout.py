"""Write the folder layout train.py expects, from the Hugging Face train release, with part of train held out.
The held-out images are written as both 'val' and 'test' (val/test labels are never released).
Example: python tools/make_holdout.py --src ~/cs701-data/train --out ~/cs701-data/holdout15_seed0"""
import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--src', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--frac', type=float, default=0.15, help='fraction of every class held out')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--train-all', action='store_true',
               help='final model: train on ALL images; the held-out images stay as val/test for monitoring only '
                    '(they are then also in train, so their scores are training-set scores)')
args = p.parse_args()

rows = list(csv.DictReader(open(args.src / 'labels.csv')))
coco = json.loads((args.src / 'instances.json').read_text())
by_class = defaultdict(list)
for r in rows:
    by_class[r['label']].append(r)
rng = random.Random(args.seed)
held = set()
for label, rs in sorted(by_class.items()):
    ids = sorted(r['image_id'] for r in rs)
    rng.shuffle(ids)
    held.update(ids[:round(args.frac * len(ids))])  # stratified: the same fraction of every class

(args.out / 'classification').mkdir(parents=True, exist_ok=True)
(args.out / 'detection').mkdir(parents=True, exist_ok=True)
if not (args.out / 'images').exists():
    os.symlink(os.path.relpath((args.src / 'images').resolve(), args.out.resolve()), args.out / 'images')  # relative: survives moving the data folder
splits = {'train': [r for r in rows if args.train_all or r['image_id'] not in held],
          'val': [r for r in rows if r['image_id'] in held]}
splits['test'] = splits['val']
for split, rs in splits.items():
    with open(args.out / 'classification' / f'{split}.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image', 'label'])
        w.writerows([r['file_name'], r['label']] for r in rs)
    ids = {int(r['image_id']) for r in rs}
    sub = {'images': [i for i in coco['images'] if i['id'] in ids],
           'annotations': [a for a in coco['annotations'] if a['image_id'] in ids],
           'categories': coco['categories']}
    (args.out / 'detection' / f'instances_{split}.json').write_text(json.dumps(sub))
    print(split, len(rs), 'images', len(sub['annotations']), 'boxes')
(args.out / 'SPLIT.json').write_text(json.dumps({'src': str(args.src), 'frac': args.frac, 'seed': args.seed, 'train_all': args.train_all,
                                                  'held_out_image_ids': sorted(int(i) for i in held)}))
