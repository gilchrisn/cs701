"""The multi-task model: one shared ViT backbone feeds a classification head and a Faster R-CNN detector."""
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.ops import MultiScaleRoIAlign, box_iou

from .adapters import set_task


class SimpleFeaturePyramid(nn.Module):
    """ViTDet neck (Li et al., 2022): the single stride-16 ViT map -> 256-channel maps at strides 4, 8, 16, 32, 64."""

    def __init__(self, dim=768, channels=256):
        super().__init__()
        self.rescale = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(dim, dim // 2, 2, stride=2), nn.GroupNorm(32, dim // 2), nn.GELU(),
                          nn.ConvTranspose2d(dim // 2, dim // 4, 2, stride=2)),  # stride 4
            nn.ConvTranspose2d(dim, dim // 2, 2, stride=2),  # stride 8
            nn.Identity(),  # stride 16
            nn.MaxPool2d(2),  # stride 32
        ])
        self.project = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, channels, 1, bias=False), nn.GroupNorm(32, channels),
                          nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.GroupNorm(32, channels))
            for c in (dim // 4, dim // 2, dim, dim))

    def forward(self, x):
        levels = [project(rescale(x)) for rescale, project in zip(self.rescale, self.project)]
        levels.append(F.max_pool2d(levels[-1], 1, stride=2))  # stride 64, as torchvision's FPN LastLevelMaxPool
        return OrderedDict(zip(['0', '1', '2', '3', 'pool'], levels))


class SpatialPrior(nn.Module):
    """Optional (--spatial-prior). A light convolutional stem on the input image, after the spatial prior module of
    ViT-Adapter (Chen et al., ICLR 2023): maps at strides 4 and 8 with detail finer than one 16 px patch, added to the
    neck's stride-4 and stride-8 maps. The output projections start at zero: training starts from the reference model."""

    def __init__(self, channels=256, width=64):
        super().__init__()

        def block(c_in, c_out, stride):
            return nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False), nn.GroupNorm(16, c_out), nn.ReLU(inplace=True))

        self.stem = nn.Sequential(block(1, width, 2), block(width, width, 1), block(width, width, 2))  # stride 4
        self.down = block(width, 2 * width, 2)  # stride 8
        self.out4, self.out8 = nn.Conv2d(width, channels, 1), nn.Conv2d(2 * width, channels, 1)
        for conv in (self.out4, self.out8):
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(self, images):
        f4 = self.stem(images)
        return self.out4(f4), self.out8(self.down(f4))


def nwd_similarity(a, b, c=16.0):
    """Optional (--nwd). Normalized Gaussian Wasserstein distance (Wang et al., arXiv 2110.13389) between XYXY boxes:
    a box is the Gaussian N(centre, diag(w^2/4, h^2/4)); W2^2 = |centre_a - centre_b|^2 + |(w_a, h_a)/2 - (w_b, h_b)/2|^2;
    similarity = exp(-W2 / c). c = 16 px, the median box side at 512 input (the paper uses the mean size of its all-tiny
    dataset; our mean, 35.7 px, is pulled up by airports). Replaces IoU only where anchors/proposals are matched to
    ground truth in training; NMS and evaluation still use IoU."""
    ca, cb = (a[:, :2] + a[:, 2:]) / 2, (b[:, :2] + b[:, 2:]) / 2
    ha, hb = (a[:, 2:] - a[:, :2]) / 2, (b[:, 2:] - b[:, :2]) / 2
    w2 = ((ca[:, None] - cb[None]) ** 2).sum(-1) + ((ha[:, None] - hb[None]) ** 2).sum(-1)
    return torch.exp(-w2.clamp(min=0).sqrt() / c)


def iou_or_nwd(a, b):
    """max(IoU, NWD): NWD alone (c = 16) gives boxes over 32 px almost no positive anchors (84 % none, measured by
    tools/anchor_coverage.py); the max keeps IoU matching for large boxes and adds NWD positives for tiny ones.
    This combination is ours, not from the NWD paper."""
    return torch.maximum(box_iou(a, b), nwd_similarity(a, b))


class MultiTaskModel(nn.Module):
    """backbone tokens -> classification logits (mean token -> LayerNorm -> Linear) and Faster R-CNN detections."""

    def __init__(self, backbone, task_routing: bool, num_classes=9, spatial_prior=False, nwd=False):
        super().__init__()
        self.backbone = backbone
        self.task_routing = task_routing  # MoE-LoRA: the backbone runs once per task, with that task's routers
        self.cls_head = nn.Sequential(nn.LayerNorm(backbone.embed_dim), nn.Linear(backbone.embed_dim, num_classes))
        self.neck = SimpleFeaturePyramid(backbone.embed_dim)
        # Small objects (median ~16 px at 512 px, 99 % below ~240 px): two anchor scales per octave, from 8 px at
        # stride 4 up to 181 px at stride 64, times 3 aspect ratios = 6 anchors per location.
        anchors = AnchorGenerator(((8, 11), (16, 23), (32, 45), (64, 91), (128, 181)), ((0.5, 1.0, 2.0),) * 5)
        # Everything else is torchvision's Faster R-CNN default.
        self.rpn = RegionProposalNetwork(
            anchors, RPNHead(256, 6), fg_iou_thresh=0.7, bg_iou_thresh=0.3, batch_size_per_image=256,
            positive_fraction=0.5, pre_nms_top_n=dict(training=2000, testing=1000),
            post_nms_top_n=dict(training=2000, testing=1000), nms_thresh=0.7)
        self.roi_heads = RoIHeads(
            MultiScaleRoIAlign(['0', '1', '2', '3'], output_size=7, sampling_ratio=2),
            TwoMLPHead(256 * 7 * 7, 1024), FastRCNNPredictor(1024, num_classes + 1),  # + 1: background
            fg_iou_thresh=0.5, bg_iou_thresh=0.5, batch_size_per_image=512, positive_fraction=0.25,
            bbox_reg_weights=None, score_thresh=0.05, nms_thresh=0.5, detections_per_img=100)
        self.spatial_prior = SpatialPrior() if spatial_prior else None
        if nwd:  # matching anchors (RPN) and proposals (RoI heads) to ground truth by NWD instead of IoU
            self.rpn.box_similarity = iou_or_nwd
            self.roi_heads.box_similarity = iou_or_nwd

    def forward(self, images, targets=None):
        """images [B, 1, H, W], targets: per-image {'boxes' XYXY, 'labels' 1..9} (training only) ->
        (logits [B, num_classes], per-image detections (eval mode), the four Faster R-CNN losses (training mode))."""
        set_task(self.backbone, 0)  # 0 = classification, 1 = detection; a no-op without MoE-LoRA
        tokens = self.backbone(images)
        logits = self.cls_head(tokens.mean(dim=1))
        if self.task_routing:
            set_task(self.backbone, 1)
            tokens = self.backbone(images)
        h, w = images.shape[-2] // 16, images.shape[-1] // 16  # one token per 16 x 16 patch
        features = self.neck(tokens.transpose(1, 2).unflatten(2, (h, w)))  # tokens as a [B, 768, h, w] map
        if self.spatial_prior is not None:
            p4, p8 = self.spatial_prior(images)
            features['0'], features['1'] = features['0'] + p4, features['1'] + p8
        # Precision rule: the box coder casts anchors to the dtype of the regression output, so in bf16 every
        # coordinate near 512 would snap to a 2 px grid. RPN and RoIHeads therefore run in fp32.
        with torch.autocast(images.device.type, enabled=False):
            features = {name: f.float() for name, f in features.items()}
            image_list = ImageList(images, [tuple(images.shape[-2:])] * len(images))
            proposals, rpn_losses = self.rpn(image_list, features, targets)
            detections, roi_losses = self.roi_heads(features, proposals, image_list.image_sizes, targets)
        return logits, detections, {**rpn_losses, **roi_losses}
