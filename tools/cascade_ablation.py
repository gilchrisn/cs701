"""Which part of a Cascade run (--cascade) makes its detections: re-infer the held-out split once and, from the same
backbone features and RPN proposals, post-process with boxes from {stage 1, stage 2} x class scores from {stage 1,
stage 2, mean}. Writes <run>/ablate_<boxes>_<logits>/{predictions_val.json, class_probs_val.json, metrics.json} for
tools/analyze.py. Example: python tools/cascade_ablation.py ~/cs701-runs/2026-10-08_cascade_s0 --device cpu"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbench.adapters import add_adapters  # noqa: E402
from sarbench.backbones import build_backbone  # noqa: E402
from sarbench.data import SARMultiTask, collate, to_original_xywh  # noqa: E402
from sarbench.model import MultiTaskModel  # noqa: E402
from sarbench.paths import data_root  # noqa: E402

CONFIGS = [(b, s) for b in ('stage1', 'stage2') for s in ('stage1', 'stage2', 'mean')]  # + 'mean_noiou' for --iou-score

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('run', type=Path)
p.add_argument('--device', default='cpu')
p.add_argument('--threads', type=int, default=6)
p.add_argument('--batch-size', type=int, default=4)
p.add_argument('--limit', type=int, help='first N images only (quick checks)')
args = p.parse_args()
torch.set_num_threads(args.threads)
run = args.run.expanduser()
train_args = json.loads((run / 'metrics.json').read_text())['args']
assert train_args.get('cascade'), 'not a --cascade run'
if train_args.get('iou_score'):
    CONFIGS.append(('stage2', 'mean_noiou'))  # stage-1-box configs never get the IoU factor (model.py)

backbone = build_backbone(train_args['backbone'], pretrained=False)
if train_args['adapt'] != 'full':
    add_adapters(backbone, train_args['adapt'])
model = MultiTaskModel(backbone, task_routing=train_args['adapt'] == 'moelora', cascade=True,
                       nwd_rpn=train_args.get('nwd_rpn', False), iou_score=train_args.get('iou_score', False),
                       kl_loss=train_args.get('kl_loss', False),
                       balanced_l1=train_args.get('balanced_l1', False),
                       roi_box_weight=train_args.get('roi_box_weight', 1.0))
model.load_state_dict(torch.load(run / 'model.pt', map_location='cpu'))
model.to(args.device).eval()
captured = {}
model.roi_heads.register_forward_pre_hook(lambda mod, inp: captured.update(f=inp[0], p=inp[1], s=inp[2]))

loader = DataLoader(SARMultiTask(data_root(train_args['data']), 'val', limit=args.limit), args.batch_size, num_workers=2,
                    collate_fn=collate)
detections, probs = {c: [] for c in CONFIGS}, {}
with torch.no_grad():
    for images, _, targets in loader:
        logits, default_out, _ = model(images.to(args.device))
        for target, pr in zip(targets, logits.float().softmax(dim=1).tolist()):
            probs[target['image_id']] = pr
        for boxes_from, scores_from in CONFIGS:
            model.roi_heads.test_boxes, model.roi_heads.test_logits = boxes_from, scores_from.replace('_noiou', '')
            model.roi_heads.test_iou_score = not scores_from.endswith('_noiou')
            outputs, _ = model.roi_heads(captured['f'], captured['p'], captured['s'])
            if (boxes_from, scores_from) == ('stage2', 'mean'):  # the method's own path must match the full forward
                assert all(torch.equal(a['boxes'], b['boxes']) for a, b in zip(outputs, default_out))
            for target, out in zip(targets, outputs):
                boxes = to_original_xywh(out['boxes'].cpu(), target['orig_size']).tolist()
                detections[boxes_from, scores_from] += [
                    {'image_id': target['image_id'], 'category_id': c, 'bbox': b, 'score': s}
                    for b, c, s in zip(boxes, out['labels'].tolist(), out['scores'].tolist())]
        model.roi_heads.test_boxes, model.roi_heads.test_logits, model.roi_heads.test_iou_score = 'stage2', 'mean', True
for (boxes_from, scores_from), dets in detections.items():
    out_dir = run / f'ablate_{boxes_from}_{scores_from}'
    out_dir.mkdir(exist_ok=True)
    (out_dir / 'predictions_val.json').write_text(json.dumps(dets))
    (out_dir / 'class_probs_val.json').write_text(json.dumps(probs))
    posthoc = vars(args) | {'run': str(run), 'split': 'val', 'boxes_from': boxes_from, 'scores_from': scores_from}
    (out_dir / 'metrics.json').write_text(json.dumps({'args': train_args, 'posthoc': posthoc}, default=str, indent=2))
    print(f'{boxes_from} boxes, {scores_from} scores: {len(dets)} detections -> {out_dir}')
