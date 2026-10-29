"""Train or fine-tune an Ultralytics YOLO detector (512 x 512 unless --imgsz) on a dataset written by tools/make_yolo_data.py.
Weights go to <out>/weights/{best,last}.pt, per-epoch held-out metrics to <out>/results.csv.
Our best model (96 epochs from COCO weights):
    python train_yolo.py --data data/yolo_holdout15_seed0/data.yaml --model yolo26l-p2.yaml --init yolo26l.pt --out runs/yolo26lp2_96ep
Fine-tune saved weights; any other Ultralytics training argument goes at the end as key=value (lr0 is ignored unless an
optimizer is named: the default optimizer=auto picks its own learning rate):
    python train_yolo.py --data data/yolo_holdout15_seed0/data.yaml --model yolo26l-p2_96ep.pt --epochs 24 --out runs/ft \
        optimizer=AdamW lr0=0.0002 freeze=10"""
import argparse
import ast
from pathlib import Path

from ultralytics import YOLO

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--data', type=Path, required=True, help='data.yaml from tools/make_yolo_data.py')
p.add_argument('--model', default='yolo26l-p2.yaml', help='architecture (.yaml), or weights (.pt) to start from')
p.add_argument('--init', help='weights loaded into the --model architecture (every layer whose shape matches)')
p.add_argument('--epochs', type=int, default=96)
p.add_argument('--batch', type=int, default=16)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--device', default='0')
p.add_argument('--imgsz', type=int, default=512, help='training input size (course limit for submissions: 512); prediction tools reuse it from the weights')
p.add_argument('--kl', action='store_true', help='KL box loss (sarbench/yolo_kl.py); strip the weights before predicting')
p.add_argument('--out', type=Path, required=True)
args, rest = p.parse_known_args()


def value(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


extra = {k: value(v) for k, v in (kv.split('=', 1) for kv in rest)}
if args.kl:
    from sarbench import yolo_kl
    yolo_kl.apply()
model = YOLO(args.model)
if args.init:
    model = model.load(args.init)
out = args.out.expanduser().resolve()
model.train(data=str(args.data.expanduser().resolve()), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, seed=args.seed, workers=4,
            device=args.device, project=str(out.parent), name=out.name, exist_ok=True, plots=False, **extra)
