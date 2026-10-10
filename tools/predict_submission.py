"""Write a Codabench submission zip (classification.csv + detection.json, via make_submission.write_submission) for
the images listed in an images.json (ids, file names, sizes; released with val/test). Preprocessing is the same as
training (gray, resized to 512 x 512, normalised). Example, after the val release:
python tools/predict_submission.py ~/cs701-runs/2026-10-10_submission_cascade24_full \
    --images-json ~/cs701-data/val/images.json --root ~/cs701-data/val --out submission_val.zip"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from make_submission import write_submission  # noqa: E402
from sarbench.adapters import add_adapters  # noqa: E402
from sarbench.backbones import build_backbone  # noqa: E402
from sarbench.data import MEAN, STD  # noqa: E402
from sarbench.model import MultiTaskModel  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('run', type=Path)
p.add_argument('--images-json', type=Path, required=True)
p.add_argument('--root', type=Path, required=True, help='folder the file names in images.json are relative to')
p.add_argument('--out', type=Path, required=True)
p.add_argument('--device', default='cuda')
p.add_argument('--threads', type=int, default=8)
p.add_argument('--batch-size', type=int, default=4)
args = p.parse_args()
torch.set_num_threads(args.threads)
run = args.run.expanduser()
train_args = json.loads((run / 'metrics.json').read_text())['args']
assert not train_args.get('log_input'), 'log-input preprocessing not implemented here'
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
images = json.loads(args.images_json.expanduser().read_text())
images = images['images'] if isinstance(images, dict) else images
transform = v2.Compose([v2.ToImage(), v2.Resize((512, 512)), v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize([MEAN], [STD])])


class Images(Dataset):
    def __len__(self):
        return len(images)

    def __getitem__(self, i):
        return transform(Image.open(args.root.expanduser() / images[i]['file_name']).convert('L')), i


preds = {}
with torch.no_grad():
    for batch, idx in DataLoader(Images(), args.batch_size, num_workers=4):
        with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == 'cuda'):
            logits, outputs, _ = model(batch.to(args.device))
        for i, lg, out in zip(idx.tolist(), logits.float(), outputs):
            preds[images[i]['file_name']] = (int(lg.argmax()), out['boxes'].float().cpu().tolist(),
                                             (out['labels'].cpu() - 1).tolist(),  # model: 1..9; writer adds 1
                                             out['scores'].float().cpu().tolist())
write_submission(images, preds, out=str(args.out.expanduser()))
print(f'{len(images)} images, {sum(len(v[1]) for v in preds.values())} detections -> {args.out}')
