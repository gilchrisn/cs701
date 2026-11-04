"""D48-D52: full runs of the D38 recipe (YOLO26l-P2 = yolo26l-p2.yaml + yolo26l.pt COCO weights, 512 px, mosaic 0, scale 0, seed 0,
optimizer auto, workers 4) with one change; save_period 8 (checkpoints for weight averaging; saving does not change training).
  box2x     box loss weight 15 (default 7.5)                                                        D48
  dihedral  flipud 0.5 + Transpose 0.5 (aug_mod) on top of fliplr 0.5: the 8 rotations/flips equal   D49
  strip     strip-conv branch on the stride-4 / stride-8 head inputs (strip_mod), zero-initialised  D50
  kl        KL box loss (kl_mod, He et al. 2019); score <ckpt>_stripped.pt (python kl_mod.py strip <ckpt>)       D55
  x         YOLO26x-P2 (yolo26x-p2.yaml + yolo26x.pt), first 8 layers frozen (40 M trainable cap), as D24b       D56
  fourier   DNOD MSFM Fourier branch into the Detect classification inputs (fourier_mod)                    D59
  nohflip   no horizontal flip (fliplr 0)                                                               D62
  vflip     vertical flip 0.5 on top of fliplr 0.5 (no transpose): identity, h-flip, v-flip, 180 deg   D68
  refine    second box-refinement stage trained jointly (refine_mod)                                    D72
  dfl       distribution focal loss on (reg_max 16, yolo26l-p2dfl.yaml; box head's last layer starts untrained)          D74
  flipconv  flips map boxes with the dataset convention (W - 1 - x), vertical flip 0.5 on (flipconv_mod)   D73
  img768    train (and score) at 768 px input instead of 512                                             D70
  erasing   random erasing p 0.5 (aug_mod.RandomErasing)                                                 D63
  mosaic    mosaic 1.0 (scale stays 0)                                                                   D64
  scale25   scale jitter 0.25 (mosaic stays 0)                                                           D65
  trans20   translate 0.2 (default 0.1)                                                                  D66
  nwd       NWD in the label assigner: overlap = max(CIoU, NWD), C 35.7 px (as the D33 fine-tune)             D67
  none      no change                                                                                D52 (all images), D54 (control)
Run from the folder holding yolo26l.pt. Usage: python train_variants_2026-10-29.py <variant> <name> <data.yaml> <batch> <project> [epochs] [fraction] [seed]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

variant_arg, name, data, batch, project = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
epochs = int(sys.argv[6]) if len(sys.argv) > 6 else 96
fraction = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0  # smoke tests only
seed = int(sys.argv[8]) if len(sys.argv) > 8 else 0
extra = {}
variants = variant_arg.split('+')  # components joined by '+', e.g. kl+nohflip
for variant in variants:
    if variant == 'box2x':
        extra['box'] = 15.0
    elif variant == 'dihedral':
        from aug_mod import patch_transpose
        patch_transpose(0.5)
        extra['flipud'] = 0.5
    elif variant == 'strip':
        from strip_mod import add_strips
        from ultralytics.models.yolo.detect.train import DetectionTrainer
        orig = DetectionTrainer.get_model

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = orig(self, cfg, weights, verbose)
            print('STRIP wrapped (layer, channels):', add_strips(model), flush=True)
            return model

        DetectionTrainer.get_model = get_model
    elif variant == 'kl':
        import kl_mod
        kl_mod.apply()
    elif variant == 'fourier':
        from ultralytics.models.yolo.detect.train import DetectionTrainer as _DT
        import fourier_mod
        _orig_gm = _DT.get_model

        def _gm(self, cfg=None, weights=None, verbose=True):
            model = _orig_gm(self, cfg, weights, verbose)
            print('FOURIER branch added on Detect inputs, channels', fourier_mod.add_fourier(model), flush=True)
            return model

        _DT.get_model = _gm
    elif variant == 'nohflip':
        extra['fliplr'] = 0.0
    elif variant == 'dfl':
        pass  # model config chosen below
    elif variant == 'flipconv':
        import flipconv_mod
        flipconv_mod.patch()
        extra['flipud'] = 0.5
    elif variant == 'refine':
        import refine_mod
        refine_mod.apply()
    elif variant == 'vflip':
        extra['flipud'] = 0.5
    elif variant == 'img768':
        extra['imgsz'] = 768
    elif variant == 'erasing':
        from aug_mod import patch_erasing
        patch_erasing(0.5)
    elif variant == 'mosaic':
        extra['mosaic'] = 1.0  # Ultralytics default; close_mosaic (last 10 epochs off) stays default
    elif variant == 'imgcls':  # 2026-11-01 (QUESTIONS.md 'YOLO CLASSIFICATION HEAD'): image-classification head inside the detector
        import imgcls_mod
        imgcls_mod.apply()
    elif variant == 'exact':  # 2026-11-02 (QUESTIONS.md 'EXACT-COPY FINE-TUNE'): no photometric or translation jitter, flips only
        extra.update(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, translate=0.0)
    elif variant == 'mosaichalf':  # 2026-10-31 (QUESTIONS.md 'MOSAIC FIRST HALF'): mosaic on for the first half, off for the second half
        extra['mosaic'] = 1.0
        extra['close_mosaic'] = epochs // 2
    elif variant == 'scale25':
        extra['scale'] = 0.25
    elif variant == 'trans20':
        extra['translate'] = 0.2
    elif variant == 'nwd':  # NWD in the label assigner, exactly as the D33 fine-tune (ft_variants_2026-10-27.py): overlap = max(CIoU, NWD)
        from ultralytics.utils import tal as _T
        from ultralytics.utils.metrics import bbox_iou as _biou

        def _iou_calc(self, gt_bboxes, pd_bboxes, C=35.7):  # C = mean box side of our training boxes (Wang et al. 2021, arXiv 2110.13389)
            ciou = _biou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)
            g, p = gt_bboxes.float(), pd_bboxes.float()
            d = torch.stack([(g[..., 0] + g[..., 2] - p[..., 0] - p[..., 2]) / 2, (g[..., 1] + g[..., 3] - p[..., 1] - p[..., 3]) / 2,
                             ((g[..., 2] - g[..., 0]) - (p[..., 2] - p[..., 0])) / 2, ((g[..., 3] - g[..., 1]) - (p[..., 3] - p[..., 1])) / 2], -1)
            return torch.max(ciou, torch.exp(-d.pow(2).sum(-1).sqrt() / C).to(ciou.dtype))

        _T.TaskAlignedAssigner.iou_calculation = _iou_calc
        print('NWD assigner patch applied', flush=True)
    elif variant == 'x':
        extra['freeze'] = 8
    elif variant != 'none':
        raise SystemExit(f'unknown variant {variant}')
if epochs != 96:  # Ultralytics 'auto' gives MuSGD lr 0.01 momentum 0.9 only above 10,000 steps (96 ep); pin it for shorter screens
    extra.update(optimizer='MuSGD', lr0=0.01, momentum=0.9)
print('VARIANT', variant_arg, 'extra', extra, flush=True)
cfg, w = ('yolo26x-p2.yaml', 'yolo26x.pt') if 'x' in variants else ('yolo26l-p2.yaml', 'yolo26l.pt')
if os.environ.get('INIT_FROM'):  # 2026-11-01: fine-tune from a trained checkpoint (weights transferred by name and shape); low lr, 1 warm-up epoch
    # INIT_FROM must be a KL-STRIPPED checkpoint (4 box outputs): kl_mod widens the head to 8 outputs after loading, so an 8-output
    # checkpoint would not transfer its box layers (smoke test 2026-11-01: 16 tensors skipped, mAP 0.002)
    w = os.environ['INIT_FROM']; lr0 = float(os.environ.get('LR0', '0.002'))
    extra.update(optimizer='MuSGD', lr0=lr0, momentum=0.9, warmup_epochs=1, warmup_bias_lr=lr0)
    print('INIT_FROM', w, 'lr0', extra['lr0'], flush=True)
if 'dfl' in variants:
    cfg = 'yolo26l-p2dfl.yaml'  # yolo26-p2dfl.yaml in the run folder (Ultralytics strips the scale letter to find it)
imgsz = extra.pop('imgsz', 512)
kw = dict(mosaic=0.0, scale=0.0)  # our best recipe; a variant may override (mosaic / scale25)
kw.update(extra)
(YOLO(w) if os.environ.get('INIT_FROM') else YOLO(cfg).load(w)).train(data=data, imgsz=imgsz, epochs=epochs, batch=batch, seed=seed, workers=4, device=0,
                        project=project, name=name, exist_ok=True, plots=False, save_period=8, fraction=fraction, **kw)
