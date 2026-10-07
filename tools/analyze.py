"""Error analysis of one run on the held-out split, from its saved files (predictions_val.json, class_probs_val.json).
Writes <run>/analysis.json and prints a summary:
  - COCO mAP, AP50, AP75; AP by box side at 512 input (side = sqrt(w h) after resizing to 512 x 512); per-class AP
  - classification: accuracy, macro-F1, confusion matrix (rows = truth, columns = prediction)
  - oracles: COCO mAP after fixing one error type (TIDE-style, but at mAP@[.5:.95]):
      cls   every detection relabelled to the image's TRUE class, then NMS 0.5 (duplicates created by relabelling)
      bkg   detections that overlap no ground-truth box (max IoU < 0.1, any class) removed
      loc   a detection with a same-class ground truth at IoU >= 0.1 is snapped onto it; lower-scored detections
            snapped onto the same ground truth are removed
      dupe  a detection whose best same-class match (IoU >= 0.5) is already taken by a higher-scored one is removed
      miss  ground-truth boxes that no detection of any class overlaps (IoU >= 0.1) are removed from the ground truth
  - post-hoc fixes that need no labels: relabel to the PREDICTED image class (+ NMS 0.5); rescoring s' = s * p_cls(c)
  - TIDE (Bolya et al., ECCV 2020) dAP per error type, at its default IoU 0.5 (AP50, not mAP)
Example: python tools/analyze.py ~/cs701-runs/2026-10-08_ref_s0"""
import contextlib
import copy
import csv
import io
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_iou, nms

SIDE_BINS = [('all', 0, 1e5), ('<8', 0, 8), ('8-16', 8, 16), ('16-32', 16, 32), ('>32', 32, 1e5)]


def xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def coco_eval(gt, dets, area_bins=None):
    """COCOeval on dicts; returns (stats, precision array, eval object). area_bins: [(label, lo_side, hi_side)]."""
    if not dets:
        return np.zeros(12), None, None
    with contextlib.redirect_stdout(io.StringIO()):
        g = COCO()
        g.dataset = copy.deepcopy(gt)
        g.createIndex()
        d = g.loadRes(copy.deepcopy(dets))
        ev = COCOeval(g, d, 'bbox')
        if area_bins:
            ev.params.areaRng = [[lo ** 2, hi ** 2] for _, lo, hi in area_bins]
            ev.params.areaRngLbl = [lab for lab, _, _ in area_bins]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev.stats, ev.eval['precision'], ev


def ap_from_precision(p, k=None, a=0):
    """Mean precision over IoU thresholds and recall points (and classes, unless k is given) at area index a."""
    q = p[:, :, :, a, -1] if k is None else p[:, :, k, a, -1]
    q = q[q > -1]
    return float(q.mean()) if q.size else float('nan')


def to_512(gt, dets):
    """Copies of gt and dets with boxes in 512 x 512 input pixels (IoU is unchanged by the per-axis scaling)."""
    size = {i['id']: (i['width'], i['height']) for i in gt['images']}

    def scale(b, iid):
        w, h = size[iid]
        sx, sy = 512 / w, 512 / h
        return [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]

    g = copy.deepcopy(gt)
    for a in g['annotations']:
        a['bbox'] = scale(a['bbox'], a['image_id'])
        a['area'] = a['bbox'][2] * a['bbox'][3]
    d = [{**x, 'bbox': scale(x['bbox'], x['image_id'])} for x in dets]
    return g, d


def by_image(items):
    out = defaultdict(list)
    for x in items:
        out[x['image_id']].append(x)
    return out


def nms_per_image(dets, iou=0.5):
    """Class-agnostic NMS within every image (used after relabelling all of an image's detections to one class)."""
    out = []
    for iid, ds in by_image(dets).items():
        boxes = torch.tensor([xyxy(x['bbox']) for x in ds], dtype=torch.float32)
        keep = nms(boxes, torch.tensor([x['score'] for x in ds]), iou).tolist()
        out += [ds[k] for k in keep]
    return out


def iou_matrix(ds, gs):
    if not ds or not gs:
        return torch.zeros(len(ds), len(gs))
    return box_iou(torch.tensor([xyxy(x['bbox']) for x in ds], dtype=torch.float32),
                   torch.tensor([xyxy(g['bbox']) for g in gs], dtype=torch.float32))


def oracle_bkg(gt, dets):
    gts = by_image(gt['annotations'])
    out = []
    for iid, ds in by_image(dets).items():
        m = iou_matrix(ds, gts[iid])
        out += [x for j, x in enumerate(ds) if m.shape[1] and m[j].max() >= 0.1]
    return out


def oracle_loc(gt, dets):
    gts = by_image(gt['annotations'])
    out = []
    for iid, ds in by_image(dets).items():
        ds = sorted(ds, key=lambda x: -x['score'])
        gs = gts[iid]
        m = iou_matrix(ds, gs)
        taken = set()
        for j, x in enumerate(ds):
            same = [k for k, g in enumerate(gs) if g['category_id'] == x['category_id']]
            if same:
                k = max(same, key=lambda k: m[j, k].item())
                if m[j, k] >= 0.1:
                    if k in taken:
                        continue  # a higher-scored detection already snapped onto this ground truth
                    taken.add(k)
                    out.append({**x, 'bbox': list(gs[k]['bbox'])})
                    continue
            out.append(x)
    return out


def oracle_dupe(gt, dets):
    gts = by_image(gt['annotations'])
    out = []
    for iid, ds in by_image(dets).items():
        ds = sorted(ds, key=lambda x: -x['score'])
        gs = gts[iid]
        m = iou_matrix(ds, gs)
        taken = set()
        for j, x in enumerate(ds):
            cand = [k for k, g in enumerate(gs) if g['category_id'] == x['category_id'] and m[j, k] >= 0.5]
            free = [k for k in cand if k not in taken]
            if free:
                taken.add(max(free, key=lambda k: m[j, k].item()))
            elif cand:
                continue  # duplicate: every same-class match is already taken
            out.append(x)
    return out


def oracle_miss(gt, dets):
    ds_by = by_image(dets)
    g = copy.deepcopy(gt)
    keep = []
    for iid, gs in by_image(gt['annotations']).items():
        m = iou_matrix(ds_by.get(iid, []), gs)
        for k, a in enumerate(gs):
            if m.shape[0] and m[:, k].max() >= 0.1:
                keep.append(a)
    g['annotations'] = keep
    return g, len(gt['annotations']) - len(keep)


def tide_dap(gt, dets):
    try:
        from tidecv import TIDE, datasets
    except ImportError:
        return {'error': 'tidecv not installed'}
    try:
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            gp, dp = Path(tmp) / 'gt.json', Path(tmp) / 'dt.json'
            poly = lambda b: [[b[0], b[1], b[0] + b[2], b[1], b[0] + b[2], b[1] + b[3], b[0], b[1] + b[3]]]
            g = {**gt, 'annotations': [{**x, 'segmentation': poly(x['bbox'])} for x in gt['annotations']]}
            gp.write_text(json.dumps(g))  # tidecv reads 'segmentation' even in box mode
            dp.write_text(json.dumps([{**x, 'segmentation': poly(x['bbox'])} for x in dets]))
            tide = TIDE()
            tide.evaluate(datasets.COCO(str(gp)), datasets.COCOResult(str(dp)), mode=TIDE.BOX)
            errs = tide.get_all_errors()
        main = next(iter(errs['main'].values()))
        special = next(iter(errs['special'].values()))
        return {'main': {k: float(v) for k, v in main.items()}, 'special': {k: float(v) for k, v in special.items()}}
    except Exception as e:  # tidecv is old; report rather than crash the analysis
        return {'error': repr(e)}


def macro_f1(cm):
    f1s = []
    for c in range(len(cm)):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(len(cm))) - tp
        fn = sum(cm[c]) - tp
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s)), f1s


def main(run):
    run = Path(run).expanduser()
    metrics = json.loads((run / 'metrics.json').read_text())
    root = Path(metrics['args']['data'])
    gt = json.loads((root / 'detection' / 'instances_val.json').read_text())
    dets = json.loads((run / 'predictions_val.json').read_text())
    probs = {int(k): v for k, v in json.loads((run / 'class_probs_val.json').read_text()).items()}
    names = [c['name'] for c in sorted(gt['categories'], key=lambda c: c['id'])]
    file_to_id = {i['file_name']: i['id'] for i in gt['images']}
    label = {file_to_id[r['image']]: int(r['label']) for r in csv.DictReader(open(root / 'classification' / 'val.csv'))}
    res = {'run': str(run), 'n_images': len(gt['images']), 'n_gt': len(gt['annotations']), 'n_dets': len(dets)}

    # Base metrics, by size, per class
    stats, prec, _ = coco_eval(gt, dets)
    res['mAP'], res['AP50'], res['AP75'] = float(stats[0]), float(stats[1]), float(stats[2])
    g512, d512 = to_512(gt, dets)
    _, p512, _ = coco_eval(g512, d512, SIDE_BINS)
    sides = np.sqrt([a['area'] for a in g512['annotations']])
    res['by_side'] = {lab: {'AP': ap_from_precision(p512, a=i), 'n_gt': int(((sides >= lo) & (sides < hi)).sum())}
                      for i, (lab, lo, hi) in enumerate(SIDE_BINS)}
    res['per_class_AP'] = {names[k]: ap_from_precision(prec, k=k) for k in range(len(names))}
    res['per_class_by_side'] = {names[k]: {lab: ap_from_precision(p512, k=k, a=i) for i, (lab, _, _) in
                                           enumerate(SIDE_BINS)} for k in range(len(names))}

    # Classification
    cm = [[0] * len(names) for _ in names]
    for iid, y in label.items():
        cm[y][int(np.argmax(probs[iid]))] += 1
    f1, f1s = macro_f1(cm)
    res['classification'] = {'accuracy': sum(cm[c][c] for c in range(len(cm))) / len(label), 'macro_f1': f1,
                             'per_class_f1': dict(zip(names, f1s)), 'confusion': cm}

    # Oracles (mAP after fixing one error type) and label-free post-hoc fixes
    true_relabel = nms_per_image([{**x, 'category_id': label[x['image_id']] + 1} for x in dets])
    pred_relabel = nms_per_image([{**x, 'category_id': int(np.argmax(probs[x['image_id']])) + 1} for x in dets])
    rescored = [{**x, 'score': x['score'] * probs[x['image_id']][x['category_id'] - 1]} for x in dets]
    g_miss, n_missed = oracle_miss(gt, dets)
    variants = {'cls': (gt, true_relabel), 'bkg': (gt, oracle_bkg(gt, dets)), 'loc': (gt, oracle_loc(gt, dets)),
                'dupe': (gt, oracle_dupe(gt, dets)), 'miss': (g_miss, dets),
                'posthoc_relabel_pred': (gt, pred_relabel), 'posthoc_rescore': (gt, rescored)}
    res['oracle_and_posthoc_mAP'] = {}
    for name, (g, d) in variants.items():
        s, _, _ = coco_eval(g, d)
        res['oracle_and_posthoc_mAP'][name] = {'mAP': float(s[0]), 'AP50': float(s[1]), 'dmAP': float(s[0] - stats[0])}
    res['n_gt_missed_iou0.1'] = n_missed
    res['tide_ap50'] = tide_dap(gt, dets)
    (run / 'analysis.json').write_text(json.dumps(res, indent=2))

    # Summary
    print(f"{run.name}: mAP {res['mAP']:.4f} AP50 {res['AP50']:.4f} AP75 {res['AP75']:.4f} | "
          f"acc {res['classification']['accuracy']:.4f} macro-F1 {f1:.4f}")
    print('AP by box side at 512 input: ' + ', '.join(f"{k} {v['AP']:.3f} (n={v['n_gt']})" for k, v in res['by_side'].items()))
    print('per-class AP: ' + ', '.join(f'{k} {v:.3f}' for k, v in res['per_class_AP'].items()))
    print('mAP change if fixed / post-hoc: ' + ', '.join(f"{k} {v['dmAP']:+.4f}" for k, v in res['oracle_and_posthoc_mAP'].items()))
    print(f'ground truths missed entirely (no detection at IoU >= 0.1): {n_missed} of {res["n_gt"]}')
    print('TIDE (AP50-based) dAP:', json.dumps(res['tide_ap50']))


if __name__ == '__main__':
    main(sys.argv[1])
