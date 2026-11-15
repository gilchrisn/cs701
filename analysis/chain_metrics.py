"""Every step of the report's chain of models scored the same way on the held-out split: detections with score >= 0.05, at most 100 per
image (the course models only stored >= 0.05, so this is the one cut-off all stages share). Writes chain_metrics.json (default analysis/results/, --out)."""
import argparse
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from analyze import SIDE_BINS, ap_from_precision, coco_eval, to_512

# --runs: folder holding the run folders named in ROWS
# (each run's predictions_val.json, written by tools/score_yolo_holdout.py or train.py); --data: folder holding holdout15_seed0/.
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--data', type=Path, default=Path('data'))
ap.add_argument('--runs', type=Path, default=Path('runs'))
ap.add_argument('--out', type=Path, default=Path(__file__).parent / 'results' / 'chain_metrics.json')
ap.add_argument('--only', nargs='*', help='row names to compute (default: all)')
ap.add_argument('--predictions', nargs='*', default=[], metavar='ROW=FILE', help='read this predictions file for a row instead of the one under --runs')
args = ap.parse_args()
R = args.runs.expanduser()
gt = json.loads((args.data.expanduser() / 'holdout15_seed0/detection/instances_val.json').read_text())
names = [c['name'].replace('_', ' ') for c in sorted(gt['categories'], key=lambda c: c['id'])]
split = json.loads((Path(__file__).resolve().parents[1] / 'tools' / 'holdout15_seed0_scene_groups.json').read_text())
W = {'G0': 587 / 1426, 'G1': 463 / 1426, 'G2': 376 / 1426}
ROWS = {  # run folder under --runs (as named by the README commands) -> its predictions_val.json
 'course_vit_lora_24ep': 'course24/predictions_val.json',
 'cascade_24ep': 'cascade24/predictions_val.json',
 'cascade_best_48+24ep': 'cascade48_ft24/predictions_val.json',
 'yolo26lp2_96ep_mosaic_on': 'yolo_mosaic_on_eval/predictions_val.json',
 'mosaic_scale_off': 'yolo_mosaic_off_eval/predictions_val.json',
 'kl_box_loss': 'yolo_kl_eval/predictions_val.json',
 'flip_convention_plus_vflip': 'yolo_kl_flip_eval/predictions_val.json',
 'epochs_144_final_recipe_85pct': 'final85_eval/predictions_val.json',
 'exact_copy_ft24_post_competition': 'ft24_eval/predictions_val.json',
 'exact_copy_ft48_post_competition': 'ft48_eval/predictions_val.json',
}
def subset(ids):
    s = set(ids)
    return {**gt, 'images': [i for i in gt['images'] if i['id'] in s], 'annotations': [a for a in gt['annotations'] if a['image_id'] in s]}
res = {}
override = dict(x.split('=', 1) for x in args.predictions)
for k, rel in ROWS.items():
    if args.only and k not in args.only:
        continue
    by = {}
    for d in json.loads((Path(override[k]) if k in override else R / rel).read_text()):
        if d['score'] >= 0.05: by.setdefault(d['image_id'], []).append(d)
    dets = [d for v in by.values() for d in sorted(v, key=lambda d: -d['score'])[:100]]
    stats, prec, _ = coco_eval(gt, dets)
    g512, d512 = to_512(gt, dets)
    _, p512, _ = coco_eval(g512, d512, SIDE_BINS)
    r = {'file': rel, 'n_det': len(dets), 'mAP': float(stats[0]), 'AP50': float(stats[1]), 'AP75': float(stats[2]),
         'per_class': {n: float(ap_from_precision(prec, k=c)) for c, n in enumerate(names)},
         'by_side': {lab: float(ap_from_precision(p512, a=i)) for i, (lab, _, _) in enumerate(SIDE_BINS) if lab != 'all'}}
    for g, ids in split.items():
        s = set(ids); r[g] = float(coco_eval(subset(ids), [d for d in dets if d['image_id'] in s])[0][0])
    r['val_weighted'] = sum(W[g] * r[g] for g in W)
    res[k] = r
    print(f"{k:36s} mAP {r['mAP']:.4f} AP50 {r['AP50']:.3f} AP75 {r['AP75']:.3f} | side " + ' '.join(f'{v:.3f}' for v in r['by_side'].values()) +
          f" | new {r['G0']:.3f} shift {r['G1']:.3f} ident {r['G2']:.3f} vw {r['val_weighted']:.4f} | bridge {r['per_class']['bridge']:.3f} playg {r['per_class']['playground']:.3f}", flush=True)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(res, indent=1))
