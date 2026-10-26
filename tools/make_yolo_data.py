"""Our split as an Ultralytics YOLO dataset (detection only): <out>/images/{train,val}/<file> (relative links to the source
images), <out>/labels/{train,val}/<stem>.txt with one 'class cx cy w h' line per box (class = category_id - 1, coordinates
normalised by the original image size), <out>/data.yaml. Images without boxes get an empty label file.
Example: python tools/make_yolo_data.py ~/cs701/data/holdout15_seed0 ~/cs701/data/yolo_holdout15_seed0"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

src, out = Path(sys.argv[1]).expanduser().resolve(), Path(sys.argv[2]).expanduser()
names = None
for split in ('train', 'val'):
    coco = json.loads((src / 'detection' / f'instances_{split}.json').read_text())
    names = {c['id'] - 1: c['name'] for c in coco['categories']}
    boxes = defaultdict(list)
    for a in coco['annotations']:
        boxes[a['image_id']].append(a)
    (out / 'images' / split).mkdir(parents=True, exist_ok=True)
    (out / 'labels' / split).mkdir(parents=True, exist_ok=True)
    n_boxes = 0
    for im in coco['images']:
        f = (src / im['file_name']).resolve()
        link = out / 'images' / split / f.name
        if not link.is_symlink():
            os.symlink(os.path.relpath(f, link.parent), link)
        W, H = im['width'], im['height']
        lines = [f"{a['category_id'] - 1} {(a['bbox'][0] + a['bbox'][2] / 2) / W:.6f} {(a['bbox'][1] + a['bbox'][3] / 2) / H:.6f} "
                 f"{a['bbox'][2] / W:.6f} {a['bbox'][3] / H:.6f}" for a in boxes[im['id']]]
        (out / 'labels' / split / f'{f.stem}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
        n_boxes += len(lines)
    print(split, len(coco['images']), 'images', n_boxes, 'boxes')
(out / 'data.yaml').write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n" +
                               ''.join(f'  {k}: {v}\n' for k, v in sorted(names.items())))
