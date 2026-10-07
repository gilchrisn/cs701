"""Re-run a trained run's model on the held-out split with different post-processing (no retraining), and save the
predictions where tools/analyze.py can read them: <run>/<tag>/{predictions_val.json, class_probs_val.json, metrics.json}.
Example: python tools/infer.py ~/cs701-runs/2026-10-08_ref_s0 --tag score0.001 --score-thresh 0.001 --device cpu"""
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
p.add_argument('--tag', required=True)
p.add_argument('--score-thresh', type=float, default=0.05)
p.add_argument('--nms-thresh', type=float, default=0.5)
p.add_argument('--detections', type=int, default=100)
p.add_argument('--device', default='cpu')
p.add_argument('--threads', type=int, default=8)
p.add_argument('--batch-size', type=int, default=4)
p.add_argument('--limit', type=int, help='first N images only (quick checks)')
p.add_argument('--split', default='val', help="'train' re-infers training images (no augmentation) to measure fit")
args = p.parse_args()
torch.set_num_threads(args.threads)
run = args.run.expanduser()
train_args = json.loads((run / 'metrics.json').read_text())['args']

backbone = build_backbone(train_args['backbone'], pretrained=False)
if train_args['adapt'] != 'full':
    add_adapters(backbone, train_args['adapt'])
model = MultiTaskModel(backbone, task_routing=train_args['adapt'] == 'moelora',
                       spatial_prior=train_args.get('spatial_prior', False), nwd=train_args.get('nwd', False),
                       dynamic_rcnn=train_args.get('dynamic_rcnn', False), cascade=train_args.get('cascade', False))
model.load_state_dict(torch.load(run / 'model.pt', map_location='cpu'))
model.roi_heads.score_thresh = args.score_thresh
model.roi_heads.nms_thresh = args.nms_thresh
model.roi_heads.detections_per_img = args.detections
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
