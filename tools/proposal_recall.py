"""Where box precision is lost: per ground-truth box, the best IoU over the RPN's test-time proposals (top 1000 after
NMS) and over the final detections of the same class, binned by box side at 512 input. Also, as in Cascade R-CNN's
Fig. 1c (arXiv 1712.00726): every proposal with IoU >= 0.3 to some ground truth is assigned to it, the RoI head
regresses it for that ground truth's class, and output IoU is compared with input IoU per input-IoU bin and box size
(output > input = the regressor improves boxes); the head's softmax probability of the true class per bin shows whether
well-placed proposals are scored high enough (below 0.05 they are dropped before NMS). For --cascade runs this
section measures stage 1 only; the detection-IoU section measures the final (stage-2) boxes. Writes <run>/proposal_recall.json.
Example: python tools/proposal_recall.py ~/cs701-runs/2026-10-08_ref_s0 --device cuda"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbench.adapters import add_adapters  # noqa: E402
from sarbench.backbones import build_backbone  # noqa: E402
from sarbench.data import SARMultiTask, collate  # noqa: E402
from sarbench.model import MultiTaskModel  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('run', type=Path)
p.add_argument('--device', default='cuda')
p.add_argument('--batch-size', type=int, default=4)
p.add_argument('--limit', type=int)
args = p.parse_args()
run = args.run.expanduser()
train_args = json.loads((run / 'metrics.json').read_text())['args']
backbone = build_backbone(train_args['backbone'], pretrained=False)
if train_args['adapt'] != 'full':
    add_adapters(backbone, train_args['adapt'])
model = MultiTaskModel(backbone, task_routing=train_args['adapt'] == 'moelora',
                       spatial_prior=train_args.get('spatial_prior', False), nwd=train_args.get('nwd', False),
                       nwd_rpn=train_args.get('nwd_rpn', False),
                       dynamic_rcnn=train_args.get('dynamic_rcnn', False), cascade=train_args.get('cascade', False),
                       iou_score=train_args.get('iou_score', False),
                       kl_loss=train_args.get('kl_loss', False))
model.load_state_dict(torch.load(run / 'model.pt', map_location='cpu'))
model.to(args.device).eval()
captured = {}
model.rpn.register_forward_hook(lambda mod, inp, out: captured.__setitem__('proposals', out[0]))
model.roi_heads.register_forward_pre_hook(lambda mod, inp: captured.update(features=inp[0], sizes=inp[2]))

data = SARMultiTask(Path(train_args['data']), 'val', limit=args.limit, log_input=train_args.get('log_input', False))
rows = []  # (side at 512 input, best proposal IoU, best same-class detection IoU)
regress_rows = []  # (side of the assigned ground truth, proposal IoU, IoU after regression, true-class probability)
heads = model.roi_heads
with torch.no_grad():
    for images, _, targets in DataLoader(data, args.batch_size, num_workers=4, collate_fn=collate):
        with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == 'cuda'):
            _, outputs, _ = model(images.to(args.device))
        for i, (t, props, out) in enumerate(zip(targets, captured['proposals'], outputs)):
            gt = t['boxes'].float()  # XYXY on the 512 x 512 input
            if not len(gt):
                continue
            side = ((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])).sqrt()
            iou_all = box_iou(gt, props.float().cpu())
            best_prop = iou_all.max(dim=1).values
            in_iou, owner = iou_all.max(dim=0)  # each proposal's best ground truth
            keep = in_iou >= 0.3
            if keep.any():
                chosen = props[keep.to(props.device)].float()
                feats = {k: v[i:i + 1].float() for k, v in captured['features'].items()}
                logits, deltas = heads.box_predictor(heads.box_head(heads.box_roi_pool(feats, [chosen],
                                                                                   [captured['sizes'][i]])))
                cls = t['labels'][owner[keep]].to(chosen.device)
                ar = torch.arange(len(chosen))
                out_boxes = heads.box_coder.decode(deltas, [chosen]).reshape(len(chosen), -1, 4)[ar, cls].float().cpu()
                out_iou = box_iou(out_boxes, gt).gather(1, owner[keep][:, None]).squeeze(1)
                prob = logits.float().softmax(dim=1)[ar, cls].cpu()
                regress_rows += list(zip(side[owner[keep]].tolist(), in_iou[keep].tolist(), out_iou.tolist(),
                                         prob.tolist()))
            det_boxes, det_labels = out['boxes'].float().cpu(), out['labels'].cpu()
            iou = box_iou(gt, det_boxes) if len(det_boxes) else torch.zeros(len(gt), 0)
            same = t['labels'][:, None] == det_labels[None]
            best_det = (iou * same).max(dim=1).values if iou.shape[1] else torch.zeros(len(gt))
            rows += list(zip(side.tolist(), best_prop.tolist(), best_det.tolist()))
rows = np.array(rows)
bins = [('all', 0, 1e5), ('<8', 0, 8), ('8-16', 8, 16), ('16-32', 16, 32), ('>32', 32, 1e5)]
res = {}
for lab, lo, hi in bins:
    r = rows[(rows[:, 0] >= lo) & (rows[:, 0] < hi)]
    res[lab] = {'n_gt': len(r),
                'proposal_iou_median': float(np.median(r[:, 1])), 'detection_iou_median': float(np.median(r[:, 2])),
                **{f'proposal_iou>={t}': float((r[:, 1] >= t).mean()) for t in (0.5, 0.7)},
                **{f'detection_iou>={t}': float((r[:, 2] >= t).mean()) for t in (0.5, 0.7)},
                'detection_minus_proposal_median': float(np.median(r[:, 2] - r[:, 1]))}

rr = np.array(regress_rows)
res['regression_by_input_iou'] = {}
for lab, lo, hi in bins:
    for a, b in [(0.3, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]:
        r = rr[(rr[:, 0] >= lo) & (rr[:, 0] < hi) & (rr[:, 1] >= a) & (rr[:, 1] < b)]
        if len(r):
            res['regression_by_input_iou'][f'{lab} in[{a},{min(b, 1.0)})'] = {
                'n': len(r), 'input_iou_mean': float(r[:, 1].mean()), 'output_iou_mean': float(r[:, 2].mean()),
                'true_class_prob_mean': float(r[:, 3].mean()), 'true_class_prob<0.05': float((r[:, 3] < 0.05).mean())}
(run / 'proposal_recall.json').write_text(json.dumps(res, indent=2))
for lab, v in res.items():
    if lab == 'regression_by_input_iou':
        continue
    print(f"{lab:>6} n={v['n_gt']:5d}  proposal IoU median {v['proposal_iou_median']:.2f} (>=.5 {v['proposal_iou>=0.5']:.0%}, "
          f">=.7 {v['proposal_iou>=0.7']:.0%})  detection IoU median {v['detection_iou_median']:.2f} "
          f"(>=.5 {v['detection_iou>=0.5']:.0%}, >=.7 {v['detection_iou>=0.7']:.0%})  "
          f"det - prop median {v['detection_minus_proposal_median']:+.2f}")

print('RoI head on proposals, by box side and input IoU: n, mean input IoU -> mean output IoU, true-class prob')
for k, v in res['regression_by_input_iou'].items():
    print(f"  {k:>20} n={v['n']:6d}  {v['input_iou_mean']:.3f} -> {v['output_iou_mean']:.3f}  "
          f"prob {v['true_class_prob_mean']:.2f} (< 0.05: {v['true_class_prob<0.05']:.0%})")
