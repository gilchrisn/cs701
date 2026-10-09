"""Post-hoc NMS IoU-threshold sweep: re-infer the held-out split once and, from the same backbone features and RPN
proposals, re-run the RoI heads with each NMS threshold. Writes <run>/nms<t>/{predictions_val.json,
class_probs_val.json, metrics.json} for tools/analyze.py. Example:
python tools/nms_sweep.py ~/cs701-runs/2026-10-08_cascade_s0 --thresholds 0.3 0.4 0.5 0.6 0.7"""
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

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('run', type=Path)
p.add_argument('--thresholds', type=float, nargs='+', default=[0.3, 0.4, 0.5, 0.6, 0.7])
p.add_argument('--device', default='cpu')
p.add_argument('--threads', type=int, default=6)
p.add_argument('--batch-size', type=int, default=4)
p.add_argument('--limit', type=int, help='first N images only (quick checks)')
args = p.parse_args()
torch.set_num_threads(args.threads)
run = args.run.expanduser()
train_args = json.loads((run / 'metrics.json').read_text())['args']
backbone = build_backbone(train_args['backbone'], pretrained=False)
if train_args['adapt'] != 'full':
    add_adapters(backbone, train_args['adapt'])
model = MultiTaskModel(backbone, task_routing=train_args['adapt'] == 'moelora',
                       spatial_prior=train_args.get('spatial_prior', False), nwd=train_args.get('nwd', False),
                       nwd_rpn=train_args.get('nwd_rpn', False), dynamic_rcnn=train_args.get('dynamic_rcnn', False),
                       cascade=train_args.get('cascade', False), iou_score=train_args.get('iou_score', False),
                       kl_loss=train_args.get('kl_loss', False))
model.load_state_dict(torch.load(run / 'model.pt', map_location='cpu'))
model.to(args.device).eval()
default_nms = model.roi_heads.nms_thresh
captured = {}
model.roi_heads.register_forward_pre_hook(lambda mod, inp: captured.update(f=inp[0], p=inp[1], s=inp[2]))
loader = DataLoader(SARMultiTask(Path(train_args['data']), 'val', limit=args.limit), args.batch_size, num_workers=2,
                    collate_fn=collate)
detections, probs = {t: [] for t in args.thresholds}, {}
with torch.no_grad():
    for images, _, targets in loader:
        logits, default_out, _ = model(images.to(args.device))
        for target, pr in zip(targets, logits.float().softmax(dim=1).tolist()):
            probs[target['image_id']] = pr
        for t in args.thresholds:
            model.roi_heads.nms_thresh = t
            outputs, _ = model.roi_heads(captured['f'], captured['p'], captured['s'])
            if t == default_nms:  # the default threshold must reproduce the full forward
                assert all(torch.equal(a['boxes'], b['boxes']) for a, b in zip(outputs, default_out))
            for target, out in zip(targets, outputs):
                boxes = to_original_xywh(out['boxes'].cpu(), target['orig_size']).tolist()
                detections[t] += [{'image_id': target['image_id'], 'category_id': c, 'bbox': b, 'score': s}
                                  for b, c, s in zip(boxes, out['labels'].tolist(), out['scores'].tolist())]
        model.roi_heads.nms_thresh = default_nms
for t, dets in detections.items():
    out_dir = run / f'nms{t}'
    out_dir.mkdir(exist_ok=True)
    (out_dir / 'predictions_val.json').write_text(json.dumps(dets))
    (out_dir / 'class_probs_val.json').write_text(json.dumps(probs))
    posthoc = vars(args) | {'run': str(run), 'split': 'val', 'nms_thresh': t}
    (out_dir / 'metrics.json').write_text(json.dumps({'args': train_args, 'posthoc': posthoc}, default=str, indent=2))
    print(f'NMS {t}: {len(dets)} detections -> {out_dir}')
