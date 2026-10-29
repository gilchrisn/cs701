"""KL box loss (He et al. 2019, "Bounding Box Regression with Uncertainty", arXiv 1809.08545) for Ultralytics 8.4.182 YOLO.
apply(): every box head's last 1x1 conv gets 4 -> 8 outputs (ltrb + per-edge log-variance; old weights kept, new rows zero), the
CIoU term is unchanged, the L1 term is replaced by the KL loss (smooth-L1 form, log-variance clamped >= -7). Call it before the
model is built (train_yolo.py --kl does).
strip(ckpt): prediction loads the stock head, which would silently reshape an 8-channel output, so this writes <ckpt>_stripped.pt
with the variance rows removed and checks that it predicts the same boxes. Use the stripped file for scoring and submission:
    python -m sarbench.yolo_kl strip runs/final/weights/last.pt"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import loss as L
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist

CALLS = {}
_orig_forward_head = Detect.forward_head


def count(key):
    CALLS[key] = CALLS.get(key, 0) + 1


def expand(det):
    for heads in (det.cv2, getattr(det, 'one2one_cv2', None)):
        for seq in heads or []:
            old = seq[-1]
            if old.out_channels != 4:
                continue
            new = nn.Conv2d(old.in_channels, 8, 1).to(old.weight.device, old.weight.dtype)
            with torch.no_grad():
                new.weight.zero_()
                new.bias.zero_()
                new.weight[:4] = old.weight
                new.bias[:4] = old.bias
            seq[-1] = new


def forward_head(self, x, box_head=None, cls_head=None):
    if box_head is None or cls_head is None:
        return {}
    bs = x[0].shape[0]
    raw = torch.cat([box_head[i](x[i]).view(bs, -1, x[i].shape[2] * x[i].shape[3]) for i in range(self.nl)], dim=-1)
    scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
    out = {"boxes": raw[:, :4], "scores": scores, "feats": x}
    if raw.shape[1] == 8:
        out["logvar"] = raw[:, 4:]
    return out


def apply():
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    orig_get_model = DetectionTrainer.get_model

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = orig_get_model(self, cfg, weights, verbose)
        expand(model.model[-1])
        count('expand')
        print('KL head expanded', flush=True)
        return model

    DetectionTrainer.get_model = get_model
    Detect.forward_head = forward_head
    orig_loss = L.v8DetectionLoss.loss

    def loss(self, preds, batch):
        lv = preds.get("logvar")
        self.bbox_loss._logvar = None if lv is None else lv.permute(0, 2, 1)
        return orig_loss(self, preds, batch)

    L.v8DetectionLoss.loss = loss

    def bbox_forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride):
        fg = fg_mask.nonzero(as_tuple=True)
        weight = target_scores[fg].sum(-1, keepdim=True)
        iou = bbox_iou(pred_bboxes[fg], target_bboxes[fg], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        lv = getattr(self, '_logvar', None)
        if lv is None:
            return loss_iou, torch.zeros_like(loss_iou)
        count('kl')
        d = (pred_dist[fg].float() - bbox2dist(anchor_points, target_bboxes)[fg].float()).abs()
        a = lv[fg].float().clamp(min=-7.0)
        sl1 = torch.where(d < 1.0, 0.5 * d ** 2, d - 0.5)
        kl = (torch.exp(-a) * sl1 + 0.5 * a).mean(-1, keepdim=True) * weight
        return loss_iou, kl.sum() / target_scores_sum

    L.BboxLoss.forward = bbox_forward


def strip(path):
    w = Path(path)
    ck = torch.load(w, map_location='cpu', weights_only=False)
    key0 = 'ema' if ck.get('ema') is not None else 'model'
    Detect.forward_head = forward_head
    img = torch.from_numpy(np.random.default_rng(0).random((1, 3, 512, 512), dtype=np.float32))
    with torch.no_grad():
        ref = ck[key0].float().eval()(img)[0]
    for key in ('ema', 'model'):
        m = ck.get(key)
        if m is None:
            continue
        det = m.model[-1]
        for heads in (det.cv2, getattr(det, 'one2one_cv2', None)):
            for seq in heads or []:
                old = seq[-1]
                if old.out_channels == 8:
                    new = nn.Conv2d(old.in_channels, 4, 1).to(old.weight.device, old.weight.dtype)
                    with torch.no_grad():
                        new.weight.copy_(old.weight[:4])
                        new.bias.copy_(old.bias[:4])
                    seq[-1] = new
    Detect.forward_head = _orig_forward_head
    with torch.no_grad():
        out = ck[key0].float().eval()(img)[0]
    ck[key0] = ck[key0].half()
    dst = w.with_name(w.stem + '_stripped.pt')
    torch.save(ck, dst)
    diff = float((out - ref).abs().max())
    print('STRIP CHECK max abs diff', diff, 'shapes', tuple(ref.shape), tuple(out.shape), '->', dst, flush=True)
    assert diff < 1e-3 and ref.shape == out.shape, 'stripped model does not reproduce the trained one'


if __name__ == '__main__' and sys.argv[1] == 'strip':
    strip(sys.argv[2])
