"""Training recipe of the report's YOLO26l-P2 models (Ultralytics 8.4.182): yolo26l-p2.yaml initialised from the COCO weights yolo26l.pt,
512 px, mosaic 0, scale 0, seed 0, workers 4, save_period 8. Variants are joined with '+':
  kl        KL box loss (kl_mod, He et al. 2019); score the stripped weights (python yolo/kl_mod.py strip <ckpt>)
  flipconv  flips map boxes with the dataset convention x' = W - 1 - x (flipconv_mod); vertical flip 0.5 on top of horizontal 0.5
  exact     no HSV and no translation jitter (the post-deadline fine-tune)
  none      no change
Final model: kl+flipconv, 144 epochs. Post-deadline fine-tune: kl+flipconv+exact with INIT_FROM=<stripped 144-epoch checkpoint>, 48 epochs.
Epoch counts other than 96 pin optimizer=MuSGD lr0=0.01 momentum=0.9 (Ultralytics 'auto' picks MuSGD only above 10,000 steps).
Run from the folder holding yolo26l.pt. Usage: python train_variants.py <variant> <name> <data.yaml> <batch> <project> [epochs] [fraction] [seed]
(fraction is for smoke tests only). Reduced on 2026-11-15 from research/scripts/train_variants_2026-10-29.py: the other variants were dropped."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultralytics import YOLO  # noqa: E402

variant_arg, name, data, batch, project = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
epochs = int(sys.argv[6]) if len(sys.argv) > 6 else 96
fraction = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0  # smoke tests only
seed = int(sys.argv[8]) if len(sys.argv) > 8 else 0
extra = {}
variants = variant_arg.split('+')  # components joined by '+', e.g. kl+flipconv
for variant in variants:
    if variant == 'kl':
        import kl_mod
        kl_mod.apply()
    elif variant == 'flipconv':
        import flipconv_mod
        flipconv_mod.patch()
        extra['flipud'] = 0.5
    elif variant == 'exact':  # no photometric or translation jitter, flips only
        extra.update(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, translate=0.0)
    elif variant != 'none':
        raise SystemExit(f'unknown variant {variant}')
if epochs != 96:  # Ultralytics 'auto' gives MuSGD lr 0.01 momentum 0.9 only above 10,000 steps (96 ep); pin it for shorter or longer runs
    extra.update(optimizer='MuSGD', lr0=0.01, momentum=0.9)
print('VARIANT', variant_arg, 'extra', extra, flush=True)
cfg, w = 'yolo26l-p2.yaml', 'yolo26l.pt'
if os.environ.get('INIT_FROM'):  # fine-tune from a trained checkpoint (weights transferred by name and shape); low lr, 1 warm-up epoch
    # INIT_FROM must be a KL-STRIPPED checkpoint (4 box outputs): kl_mod widens the head to 8 outputs after loading, so an 8-output
    # checkpoint would not transfer its box layers
    w = os.environ['INIT_FROM']; lr0 = float(os.environ.get('LR0', '0.002'))
    extra.update(optimizer='MuSGD', lr0=lr0, momentum=0.9, warmup_epochs=1, warmup_bias_lr=lr0)
    print('INIT_FROM', w, 'lr0', extra['lr0'], flush=True)
imgsz = extra.pop('imgsz', 512)
kw = dict(mosaic=0.0, scale=0.0)  # our best recipe
kw.update(extra)
(YOLO(w) if os.environ.get('INIT_FROM') else YOLO(cfg).load(w)).train(data=data, imgsz=imgsz, epochs=epochs, batch=batch, seed=seed, workers=4, device=0,
                        project=project, name=name, exist_ok=True, plots=False, save_period=8, fraction=fraction, **kw)
