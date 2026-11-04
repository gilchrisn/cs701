"""Image-classification head INSIDE the YOLO26 detector (one network, one forward pass per image; QUESTIONS.md 'YOLO CLASSIFICATION HEAD').
Head (a submodule of Detect, so it is saved, EMA-averaged and optimised with the model): the 4 Detect inputs (neck outputs, strides
4/8/16/32) -> global average + global max pool per level -> concat -> LayerNorm -> Linear 512 -> GELU -> Dropout 0.2 -> Linear nc.
Training (patched E2ELoss): cross-entropy (label smoothing 0.1) of the image label (the class of the image's boxes; every image holds one
class) x W_CLS x batch size (Ultralytics' convention), added to YOLO's own loss; gradients flow into the shared backbone and neck.
Inference: Detect.forward keeps its input maps in self._feats; image_probs(det) turns them into class probabilities from the SAME forward
pass that produced the boxes. Use: import imgcls_mod; imgcls_mod.apply() before the model is built (train_variants 'imgcls')."""
import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import loss as L

W_CLS = 1.0


class ImgCls(nn.Module):
    def __init__(self, chans, nc):
        super().__init__()
        d = 2 * sum(chans)
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, nc))

    def forward(self, feats):
        return self.mlp(torch.cat([torch.cat([f.float().mean((2, 3)), f.float().amax((2, 3))], 1) for f in feats], 1))


def add_head(model):
    det = model.model[-1]
    chans = [det.cv2[j][0].conv.in_channels for j in range(det.nl)]
    det.imgcls = ImgCls(chans, det.nc).to(next(det.parameters()).device)
    return chans, sum(p.numel() for p in det.imgcls.parameters())


_orig_forward = Detect.forward


def _forward(self, x):
    if self._modules.get('imgcls') is not None:
        self._feats = list(x)  # shallow copy before Detect replaces the list items
    return _orig_forward(self, x)


Detect.forward = _forward


def image_probs(det):
    return det.imgcls(det._feats).float().softmax(1)


def apply():
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    orig_get_model = DetectionTrainer.get_model

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = orig_get_model(self, cfg, weights, verbose)
        print('IMGCLS head added on Detect inputs (channels, weights):', add_head(model), flush=True)
        return model

    DetectionTrainer.get_model = get_model
    orig_init, orig_call = L.E2ELoss.__init__, L.E2ELoss.__call__

    def init(self, model, *a, **k):
        orig_init(self, model, *a, **k)
        self._det_cls = model.model[-1]

    def call(self, preds, batch):
        total, items = orig_call(self, preds, batch)
        det = getattr(self, '_det_cls', None)
        if det is not None and getattr(det, 'imgcls', None) is not None and getattr(det, '_feats', None) is not None:
            logits = det.imgcls(det._feats)
            bs = logits.shape[0]; bi = batch['batch_idx'].long(); c = batch['cls'].view(-1).long()
            y = torch.full((bs,), -1, dtype=torch.long, device=logits.device)
            y[bi.to(logits.device)] = c.to(logits.device)  # one class per image
            keep = y >= 0
            if keep.any():
                ce = F.cross_entropy(logits[keep], y[keep], label_smoothing=0.1)
                total = total + W_CLS * ce * bs
                self._n_cls = getattr(self, '_n_cls', 0) + 1
                if self._n_cls % 50 == 1:
                    print(f'IMGCLS ce {float(ce.detach()):.3f} acc {float((logits[keep].argmax(1) == y[keep]).float().mean()):.2f} n {int(keep.sum())}', flush=True)
        return total, items

    L.E2ELoss.__init__, L.E2ELoss.__call__ = init, call
