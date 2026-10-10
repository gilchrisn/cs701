"""Re-run a trained run's model on the held-out split with different post-processing (no retraining), and save the
predictions where tools/analyze.py can read them: <run>/<tag>/{predictions_val.json, class_probs_val.json, metrics.json}.
Example: python tools/infer.py ~/cs701-runs/2026-10-08_ref_s0 --tag score0.001 --score-thresh 0.001 --device cpu"""
import argparse
import json
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.ops import boxes as box_ops

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbench.adapters import add_adapters  # noqa: E402
from sarbench.backbones import build_backbone  # noqa: E402
from sarbench.data import SARMultiTask, collate, to_original_xywh  # noqa: E402
from sarbench.model import MultiTaskModel  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('run', type=Path)
p.add_argument('--tag', required=True)
p.add_argument('--score-thresh', type=float, default=0.05)
p.add_argument('--nms-thresh', type=float, default=0.5)
p.add_argument('--detections', type=int, default=100)
p.add_argument('--device', default='cpu')
p.add_argument('--threads', type=int, default=8)
p.add_argument('--batch-size', type=int, default=4)
p.add_argument('--limit', type=int, help='first N images only (quick checks)')
p.add_argument('--split', default='val', help="'train' re-infers training images (no augmentation) to measure fit")
p.add_argument('--cascade-test-boxes', choices=['stage1', 'stage2'], default='stage2')
p.add_argument('--cascade-test-logits', choices=['stage1', 'stage2', 'mean'], default='mean')
p.add_argument('--no-iou-score', action='store_true', help='--iou-score runs: class probability only (ablation)')
p.add_argument('--no-var-vote', action='store_true', help='--kl-loss runs: stage-2 boxes without voting (ablation)')
p.add_argument('--sigma-t', type=float, help='--kl-loss runs: variance-voting sigma_t (default 0.02)')
p.add_argument('--box-vote', type=float, metavar='IOU',
               help='after NMS, replace each kept box by the score-weighted mean of the same-class pre-NMS candidates '
                    'with IoU >= IOU to it (box voting, Gidaris & Komodakis arXiv 1505.01749); scores unchanged')
p.add_argument('--var-vote', type=float, metavar='SIGMA_T',
               help='after NMS, replace each kept box by the mean of the same-class pre-NMS candidates with IoU > 0, '
                    'weighted by exp(-(1-IoU)^2/SIGMA_T) (KL-Loss variance voting, arXiv 1809.08545, with all box '
                    'variances equal); scores unchanged')
args = p.parse_args()


def postprocess_with_voting(self, class_logits, box_regression, proposals, image_shapes):
    """torchvision RoIHeads.postprocess_detections (0.2x) with one added step after NMS: box or variance voting."""
    num_classes = class_logits.shape[-1]
    pred_boxes = self.box_coder.decode(box_regression, proposals).split([len(b) for b in proposals], 0)
    pred_scores = F.softmax(class_logits, -1).split([len(b) for b in proposals], 0)
    all_boxes, all_scores, all_labels = [], [], []
    for boxes, scores, image_shape in zip(pred_boxes, pred_scores, image_shapes):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
        labels = torch.arange(num_classes, device=boxes.device).view(1, -1).expand_as(scores)
        boxes, scores, labels = boxes[:, 1:].reshape(-1, 4), scores[:, 1:].reshape(-1), labels[:, 1:].reshape(-1)
        inds = torch.where(scores > self.score_thresh)[0]
        boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)[: self.detections_per_img]
        iou, same = box_ops.box_iou(boxes[keep], boxes), labels[keep, None] == labels[None, :]
        if args.box_vote is not None:
            w = ((iou >= args.box_vote) & same) * scores
        else:
            w = ((iou > 0) & same) * torch.exp(-(1 - iou) ** 2 / args.var_vote)
        voted = (w @ boxes) / w.sum(dim=1, keepdim=True)  # each kept box is its own neighbour, so the sum is > 0
        all_boxes.append(voted)
        all_scores.append(scores[keep])
        all_labels.append(labels[keep])
    return all_boxes, all_scores, all_labels

torch.set_num_threads(args.threads)
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
                       kl_loss=train_args.get('kl_loss', False),
                       balanced_l1=train_args.get('balanced_l1', False))
model.load_state_dict(torch.load(run / 'model.pt', map_location='cpu'))
model.roi_heads.score_thresh = args.score_thresh
model.roi_heads.nms_thresh = args.nms_thresh
model.roi_heads.detections_per_img = args.detections
if train_args.get('cascade', False):
    model.roi_heads.test_boxes, model.roi_heads.test_logits = args.cascade_test_boxes, args.cascade_test_logits
    model.roi_heads.test_iou_score = not args.no_iou_score
    model.roi_heads.test_var_vote = not args.no_var_vote
    if args.sigma_t is not None:
        model.roi_heads.sigma_t = args.sigma_t
assert args.box_vote is None or args.var_vote is None, 'choose one voting rule'
assert not (train_args.get('kl_loss') and (args.box_vote is not None or args.var_vote is not None)), \
    '--kl-loss runs vote inside the model (_var_vote); use --sigma-t / --no-var-vote instead'
if args.box_vote is not None or args.var_vote is not None:
    model.roi_heads.postprocess_detections = types.MethodType(postprocess_with_voting, model.roi_heads)
model.to(args.device).eval()

data = Path(train_args['data'])
loader = DataLoader(SARMultiTask(data, args.split, limit=args.limit, log_input=train_args.get('log_input', False)), args.batch_size,
                    num_workers=4, collate_fn=collate)
detections, probs = [], {}
with torch.no_grad():
    for images, _, targets in loader:
        with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == 'cuda'):
            logits, outputs, _ = model(images.to(args.device))
        for target, out, pr in zip(targets, outputs, logits.float().softmax(dim=1).tolist()):
            probs[target['image_id']] = pr
            boxes = to_original_xywh(out['boxes'].cpu(), target['orig_size']).tolist()
            detections += [{'image_id': target['image_id'], 'category_id': c, 'bbox': b, 'score': s}
                           for b, c, s in zip(boxes, out['labels'].tolist(), out['scores'].tolist())]
out_dir = run / args.tag
out_dir.mkdir(exist_ok=True)
(out_dir / 'predictions_val.json').write_text(json.dumps(detections))
(out_dir / 'class_probs_val.json').write_text(json.dumps(probs))
(out_dir / 'metrics.json').write_text(json.dumps({'args': train_args, 'posthoc': vars(args) | {'run': str(run)}},
                                                 default=str, indent=2))
print(f'{len(detections)} detections written to {out_dir}')
