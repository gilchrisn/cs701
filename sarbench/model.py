"""The multi-task model: one shared ViT backbone feeds a classification head and a Faster R-CNN detector."""
from collections import OrderedDict

import contextlib

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection import roi_heads as roi_heads_module
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import boxes as box_ops
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


class SimilarityRoIHeads(RoIHeads):
    """RoIHeads whose proposal-to-ground-truth matching uses self.box_similarity. torchvision 0.24's RoIHeads ignores
    that attribute and always calls box_iou (its comment cites pytorch issue 27495); the RPN does use it. Same code as
    torchvision's assign_targets_to_proposals otherwise. Used only with --nwd."""

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        matched_idxs, labels = [], []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            if gt_boxes_in_image.numel() == 0:  # background image
                device = proposals_in_image.device
                clamped = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:
                matched = self.proposal_matcher(self.box_similarity(gt_boxes_in_image, proposals_in_image))
                clamped = matched.clamp(min=0)
                labels_in_image = gt_labels_in_image[clamped].to(dtype=torch.int64)
                labels_in_image[matched == self.proposal_matcher.BELOW_LOW_THRESHOLD] = 0  # background
                labels_in_image[matched == self.proposal_matcher.BETWEEN_THRESHOLDS] = -1  # ignored by the sampler
            matched_idxs.append(clamped)
            labels.append(labels_in_image)
        return matched_idxs, labels


class DynamicRoIHeads(SimilarityRoIHeads):
    """Optional (--dynamic-rcnn). Dynamic R-CNN (Zhang et al., arXiv 2004.06002), ported from mmdetection's
    DynamicRoIHead and its config (copies in ~/cs701-runs/lit/src/): every training forward records, per image, the
    75th-largest IoU between proposals and ground truth, and the (10 x images)-th smallest mean |dx|, |dy| regression
    target of the positives; every 100 forwards the RoI positive/negative IoU threshold becomes max(0.4, mean recorded
    IoU) and the box loss's smooth-L1 beta becomes min(1.0, median recorded target). Starts at threshold 0.5 and beta
    1.0 (torchvision's fixed beta is 1/9). Not ported: the config's RPN NMS 0.85. Threshold and beta are buffers, so
    they survive --resume; the histories (at most 100 forwards) do not. Every update is printed."""
    iou_topk, beta_topk, interval, initial_iou, initial_beta = 75, 10, 100, 0.4, 1.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('dynamic_state', torch.tensor([self.proposal_matcher.high_threshold, self.initial_beta]))
        self.iou_history, self.beta_history, self.forwards = [], [], 0

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        thr = float(self.dynamic_state[0])
        self.proposal_matcher.high_threshold = self.proposal_matcher.low_threshold = thr
        cur = []
        for props, gts in zip(proposals, gt_boxes):
            if gts.numel():  # torchvision appends the ground truths; excluded here (mmdetection's sampler adds them
                # with IoU 1 before this statistic, a negligible shift at K = 75). Measured on this data (ref s0 RPN,
                # 96 training images): the 75th-largest IoU averages 0.25, so the threshold sits at the 0.4 floor.
                best = self.box_similarity(gts, props[:len(props) - len(gts)]).max(dim=0).values
                cur.append(best.topk(min(self.iou_topk, len(best))).values[-1].item())
        if cur:
            self.iou_history.append(sum(cur) / len(cur))
        return super().assign_targets_to_proposals(proposals, gt_boxes, gt_labels)

    def forward(self, features, proposals, image_shapes, targets=None):
        if not self.training:
            return super().forward(features, proposals, image_shapes, targets)
        proposals, _, labels, regression_targets = self.select_training_samples(proposals, targets)
        box_features = self.box_head(self.box_roi_pool(features, proposals, image_shapes))
        class_logits, box_regression = self.box_predictor(box_features)
        num_images = len(labels)
        labels, regression_targets = torch.cat(labels), torch.cat(regression_targets)
        loss_classifier = F.cross_entropy(class_logits, labels)
        pos = torch.where(labels > 0)[0]
        box_regression = box_regression.reshape(len(labels), -1, 4)[pos, labels[pos]]
        loss_box_reg = F.smooth_l1_loss(box_regression, regression_targets[pos], beta=float(self.dynamic_state[1]),
                                        reduction='sum') / labels.numel()  # torchvision's fastrcnn_loss, beta varied
        if len(pos):
            err = regression_targets[pos, :2].abs().mean(dim=1)
            self.beta_history.append(err.kthvalue(min(self.beta_topk * num_images, len(pos))).values.item())
        self.forwards += 1
        if self.forwards % self.interval == 0:
            thr = max(self.initial_iou, sum(self.iou_history) / max(1, len(self.iou_history)))
            beta = float(self.dynamic_state[1])
            if self.beta_history and float(np.median(self.beta_history)) >= 1e-15:
                beta = min(self.initial_beta, float(np.median(self.beta_history)))
            self.dynamic_state.copy_(torch.tensor([thr, beta]))
            self.iou_history, self.beta_history = [], []
            print(f'dynamic-rcnn: forward {self.forwards} iou_thr {thr:.4f} beta {beta:.4f}', flush=True)
        return [], {'loss_classifier': loss_classifier, 'loss_box_reg': loss_box_reg}


def kl_box_loss(deltas, alpha, labels, regression_targets):
    """KL-Loss box term (He et al., arXiv 1809.08545, smooth-L1 form): e^-α · smoothL1(Δ) + α/2 per positive delta, with
    torchvision's fastrcnn_loss beta (1/9) and normalisation (sum over positives / all sampled RoIs), so α = 0 gives
    exactly torchvision's box loss. deltas/alpha: [N, 4 * classes] (class-specific) or [N, 4] (class-agnostic)."""
    labels, targets = torch.cat(labels), torch.cat(regression_targets)
    pos, n = torch.where(labels > 0)[0], len(labels)
    pick = lambda t: t.reshape(n, -1, 4)[pos, labels[pos]] if t.shape[1] > 4 else t[pos]
    a = pick(alpha)
    return (torch.exp(-a) * F.smooth_l1_loss(pick(deltas), targets[pos], beta=1 / 9, reduction='none') + a / 2).sum() / n


class AgnosticPredictor(nn.Module):
    """Class scores (num_classes + 1) and ONE class-agnostic box delta per RoI, as in Cascade R-CNN's later stages."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score, self.bbox_pred = nn.Linear(in_channels, num_classes), nn.Linear(in_channels, 4)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.cls_score(x), self.bbox_pred(x)


class CascadeRoIHeads(RoIHeads):
    """Optional (--cascade). Cascade R-CNN (Cai & Vasconcelos, arXiv 1712.00726) with ONE extra stage (the paper and
    mmdetection's cascade-rcnn_r50_fpn config use three stages at IoU 0.5 / 0.6 / 0.7; two extra 2-fc heads, ~28 M,
    would break the 130 M total budget). Stage 1 is the unchanged reference head (IoU 0.5, class-specific deltas).
    Its boxes, refined with the assigned class (training) or the top foreground class, become stage 2's proposals:
    IoU 0.6, its own 2-fc head (~13.9 M), class-agnostic deltas with box-coder weights (20, 20, 10, 10) (mmdetection's
    stage-2 target_stds 0.05 / 0.1), loss weight 0.5 (mmdetection's stage_loss_weights). Ground-truth boxes appended
    as stage-1 proposals are dropped before refinement and re-added by stage 2 (mmdetection: pos_is_gts). Test: the
    class logits of both stages are averaged (mmdetection: _refine_roi) and boxes come from stage 2. Kept from
    torchvision rather than mmdetection: smooth-L1 beta 1/9 and default initialisation, as in stage 1."""
    stage2_iou, stage2_weights, stage2_loss_weight = 0.6, (20.0, 20.0, 10.0, 10.0), 0.5
    test_boxes, test_logits = 'stage2', 'mean'  # inference-only ablations (tools/infer.py); defaults = the method
    test_iou_score = True  # --iou-score runs only; False = class probability alone (ablation)
    sigma_t, test_var_vote = 0.02, True  # --kl-loss runs only: "We set σt to 0.02" (1809.08545); False = no voting

    def enable_kl_loss(self):
        """Optional (--kl-loss). KL-Loss (He et al., arXiv 1809.08545): both box regressors also predict α = log σ² for
        each delta they output (stage 1: per class and coordinate; stage 2: per coordinate) and train with
        kl_box_loss instead of smooth-L1. The α layers start at weight N(0, 1e-4), bias 0, so α ≈ 0 and the loss starts
        equal to the current one plus a constant. Test: variance voting on stage-2 boxes after NMS (_var_vote)."""
        n_cls = self.box_predictor.cls_score.out_features
        self.kl_alpha1, self.kl_alpha2 = nn.Linear(1024, 4 * n_cls), nn.Linear(1024, 4)
        for layer in (self.kl_alpha1, self.kl_alpha2):
            nn.init.normal_(layer.weight, std=1e-4)
            nn.init.zeros_(layer.bias)

    def _var_vote(self, logits, deltas, alpha, proposals, image_shapes):
        """torchvision's postprocess_detections for class-agnostic stage-2 deltas, plus variance voting after NMS:
        each kept box = sum_j p_j x_j / σ_j² / sum_j p_j / σ_j² per corner coordinate, over same-class candidates j with
        IoU > 0, p_j = exp(-(1 - IoU)² / σt); scores unchanged. torchvision predicts (dx, dy, dw, dh), not corners, so
        corner variances come from the first-order delta method: var(x1) = (w_prop / wx)² σ²_dx + (w_box / (2 ww))² σ²_dw
        (the box coder divides each delta by its weight; x1 = cx - w / 2)."""
        sizes, props = [len(p) for p in proposals], torch.cat(proposals)
        boxes = self.box_coder2.decode(deltas, proposals).reshape(-1, 4)
        wx, wy, ww, wh = self.box_coder2.weights
        var = alpha.exp()
        var_x = ((props[:, 2] - props[:, 0]) / wx) ** 2 * var[:, 0] + ((boxes[:, 2] - boxes[:, 0]) / (2 * ww)) ** 2 * var[:, 2]
        var_y = ((props[:, 3] - props[:, 1]) / wy) ** 2 * var[:, 1] + ((boxes[:, 3] - boxes[:, 1]) / (2 * wh)) ** 2 * var[:, 3]
        corner_var = torch.stack([var_x, var_y, var_x, var_y], dim=1)
        all_boxes, all_scores, all_labels = [], [], []
        for b, v, s, shape in zip(boxes.split(sizes), corner_var.split(sizes), F.softmax(logits, -1).split(sizes),
                                  image_shapes):
            b = box_ops.clip_boxes_to_image(b, shape)
            n, c = s.shape
            labels = torch.arange(1, c, device=s.device).repeat(n)  # same order as torchvision's flatten of [:, 1:]
            idx = torch.arange(n, device=s.device).repeat_interleave(c - 1)
            s = s[:, 1:].reshape(-1)
            keep = torch.where(s > self.score_thresh)[0]
            idx, s, labels = idx[keep], s[keep], labels[keep]
            keep = box_ops.remove_small_boxes(b[idx], min_size=1e-2)
            idx, s, labels = idx[keep], s[keep], labels[keep]
            keep = box_ops.batched_nms(b[idx], s, labels, self.nms_thresh)[: self.detections_per_img]
            iou = box_iou(b[idx[keep]], b[idx])
            p = ((iou > 0) & (labels[keep, None] == labels[None, :])) * torch.exp(-(1 - iou) ** 2 / self.sigma_t)
            # [kept, candidates, 4]; zero weight where p = 0, so one degenerate variance cannot NaN unrelated boxes
            w = torch.where(p[:, :, None] > 0, p[:, :, None] / v[idx][None].clamp_min(1e-12), 0.0)
            all_boxes.append((w * b[idx][None]).sum(dim=1) / w.sum(dim=1))
            all_scores.append(s[keep])
            all_labels.append(labels[keep])
        return all_boxes, all_scores, all_labels

    def enable_iou_branch(self):
        """Optional (--iou-score). IoU-Net's IoU predictor, joint variant only (Jiang et al., arXiv 1807.11590; their
        IoU-guided NMS and refinement need a separately trained network / Precise RoI Pooling, so they are left out).
        One Linear on the stage-2 fc features predicts the IoU of the stage-2 box with its matched ground truth (BCE on
        stage-2 positives, detached target, weight = stage2_loss_weight). Test: class probability x predicted IoU."""
        self.iou_pred = nn.Linear(self.box_predictor2.cls_score.in_features, 1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.box_head2 = TwoMLPHead(256 * 7 * 7, 1024)
        self.box_predictor2 = AgnosticPredictor(1024, self.box_predictor.cls_score.out_features)
        self.matcher2 = det_utils.Matcher(self.stage2_iou, self.stage2_iou, allow_low_quality_matches=False)
        self.box_coder2 = det_utils.BoxCoder(self.stage2_weights)
        self.iou_pred = None  # set in __init__, not on the class: a class attribute would hide the submodule
        self.kl_alpha1 = self.kl_alpha2 = None  # likewise (enable_kl_loss)

    @contextlib.contextmanager
    def _stage2(self):
        """Make torchvision's select_training_samples / postprocess_detections use stage 2's matcher and box coder."""
        saved = self.proposal_matcher, self.box_coder
        self.proposal_matcher, self.box_coder = self.matcher2, self.box_coder2
        try:
            yield
        finally:
            self.proposal_matcher, self.box_coder = saved

    def _refine(self, proposals, logits, deltas, image_shapes, labels=None, gt_boxes=None):
        """Stage-1 boxes for each RoI: the assigned class's deltas (training positives) or the top foreground class's."""
        boxes = self.box_coder.decode(deltas, proposals).reshape(len(logits), -1, 4)
        cls = logits[:, 1:].argmax(dim=1) + 1
        if labels is not None:
            assigned = torch.cat(labels)
            cls = torch.where(assigned > 0, assigned, cls)
        refined = boxes[torch.arange(len(boxes), device=boxes.device), cls].split([len(p) for p in proposals])
        out = []
        for i, (r, shape) in enumerate(zip(refined, image_shapes)):
            r = box_ops.clip_boxes_to_image(r.detach(), shape)
            if gt_boxes is not None and len(gt_boxes[i]):
                r = r[box_iou(proposals[i], gt_boxes[i]).max(dim=1).values < 0.9999]  # drop appended ground truths
            out.append(r)
        return out

    def _iou_loss(self, fc2, deltas2, refined, matched2, labels2, targets):
        """BCE between the predicted IoU and the (detached) IoU of each stage-2 positive's decoded box with its GT."""
        pos = torch.cat(labels2) > 0
        boxes = self.box_coder2.decode(deltas2.detach(), refined).reshape(-1, 4)
        gts = torch.cat([t['boxes'][m] if len(t['boxes']) else torch.zeros(len(m), 4, device=m.device)
                         for t, m in zip(targets, matched2)])
        lt, rb = torch.maximum(boxes[:, :2], gts[:, :2]), torch.minimum(boxes[:, 2:], gts[:, 2:])
        inter = (rb - lt).clamp(min=0).prod(dim=1)
        iou = inter / (box_ops.box_area(boxes) + box_ops.box_area(gts) - inter).clamp(min=1e-6)
        if not pos.any():
            return fc2.sum() * 0
        return F.binary_cross_entropy_with_logits(self.iou_pred(fc2[pos]).squeeze(1), iou[pos].detach())

    def forward(self, features, proposals, image_shapes, targets=None):
        n_cls = self.box_predictor2.cls_score.out_features
        if self.training:
            proposals, _, labels, regression_targets = self.select_training_samples(proposals, targets)
        fc1 = self.box_head(self.box_roi_pool(features, proposals, image_shapes))
        logits1, deltas1 = self.box_predictor(fc1)
        with torch.no_grad():
            refined = self._refine(proposals, logits1, deltas1, image_shapes,
                                   labels if self.training else None,
                                   [t['boxes'] for t in targets] if self.training else None)
        if self.training:
            loss_classifier, loss_box_reg = roi_heads_module.fastrcnn_loss(logits1, deltas1, labels, regression_targets)
            if self.kl_alpha1 is not None:
                loss_box_reg = kl_box_loss(deltas1, self.kl_alpha1(fc1), labels, regression_targets)
            with self._stage2():
                refined, matched2, labels2, targets2 = self.select_training_samples(refined, targets)
        fc2 = self.box_head2(self.box_roi_pool(features, refined, image_shapes))
        logits2, deltas2 = self.box_predictor2(fc2)
        if self.training:
            loss_cls2, loss_box2 = roi_heads_module.fastrcnn_loss(logits2, deltas2.repeat(1, n_cls), labels2, targets2)
            if self.kl_alpha2 is not None:
                loss_box2 = kl_box_loss(deltas2, self.kl_alpha2(fc2), labels2, targets2)
            w = self.stage2_loss_weight
            losses = {'loss_classifier': loss_classifier, 'loss_box_reg': loss_box_reg,
                      'loss_classifier_stage2': w * loss_cls2, 'loss_box_reg_stage2': w * loss_box2}
            if self.iou_pred is not None:
                losses['loss_iou_stage2'] = w * self._iou_loss(fc2, deltas2, refined, matched2, labels2, targets)
            return [], losses
        # test: stage-1 logits on the RPN proposals, stage-2 logits and boxes on the refined proposals
        # (at test time refined[i] has one row per proposals[i] row, so the two stages' rows correspond)
        logits = {'mean': (logits1 + logits2) / 2, 'stage1': logits1, 'stage2': logits2}[self.test_logits]
        if self.iou_pred is not None and self.test_iou_score and self.test_boxes == 'stage2':  # it predicts stage-2 boxes' IoU
            # foreground probabilities p_c -> p_c * IoU and background -> 1 - IoU * (1 - p_bg): the rows still sum to
            # 1, so torchvision's softmax inside postprocess_detections returns exactly p_c * IoU for every class
            p, iou = logits.softmax(dim=1), self.iou_pred(fc2).sigmoid()
            logits = torch.cat([1 - iou * (1 - p[:, :1]), p[:, 1:] * iou], dim=1).clamp_min(1e-12).log()
        if self.test_boxes == 'stage1':
            boxes, scores, labels = self.postprocess_detections(logits, deltas1, proposals, image_shapes)
        elif self.kl_alpha2 is not None and self.test_var_vote:
            boxes, scores, labels = self._var_vote(logits, deltas2, self.kl_alpha2(fc2), refined, image_shapes)
        else:
            with self._stage2():
                boxes, scores, labels = self.postprocess_detections(logits, deltas2.repeat(1, n_cls),
                                                                    refined, image_shapes)
        return [{'boxes': b, 'labels': l, 'scores': s} for b, s, l in zip(boxes, scores, labels)], {}


class MultiTaskModel(nn.Module):
    """backbone tokens -> classification logits (mean token -> LayerNorm -> Linear) and Faster R-CNN detections."""

    def __init__(self, backbone, task_routing: bool, num_classes=9, spatial_prior=False, nwd=False,
                 dynamic_rcnn=False, cascade=False, nwd_rpn=False, iou_score=False, kl_loss=False):
        super().__init__()
        if nwd + dynamic_rcnn + cascade > 1:  # each replaces the RoI heads; combining them is not implemented
            raise ValueError('use at most one of nwd, dynamic_rcnn, cascade')
        if nwd and nwd_rpn:
            raise ValueError('nwd already includes nwd_rpn')
        if iou_score and not cascade:
            raise ValueError('iou_score is implemented on the Cascade stage-2 head only')
        if kl_loss and not cascade:
            raise ValueError('kl_loss is implemented on the Cascade RoI heads only')
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
        heads = (CascadeRoIHeads if cascade else DynamicRoIHeads if dynamic_rcnn else
                 SimilarityRoIHeads if nwd else RoIHeads)  # at most one of the three options at a time
        self.roi_heads = heads(
            MultiScaleRoIAlign(['0', '1', '2', '3'], output_size=7, sampling_ratio=2),
            TwoMLPHead(256 * 7 * 7, 1024), FastRCNNPredictor(1024, num_classes + 1),  # + 1: background
            fg_iou_thresh=0.5, bg_iou_thresh=0.5, batch_size_per_image=512, positive_fraction=0.25,
            bbox_reg_weights=None, score_thresh=0.05, nms_thresh=0.5, detections_per_img=100)
        if iou_score:
            self.roi_heads.enable_iou_branch()
        if kl_loss:
            self.roi_heads.enable_kl_loss()
        self.spatial_prior = SpatialPrior() if spatial_prior else None
        if nwd:  # matching anchors (RPN) and proposals (SimilarityRoIHeads) to ground truth by max(IoU, NWD)
            self.rpn.box_similarity = iou_or_nwd
            self.roi_heads.box_similarity = iou_or_nwd
        if nwd_rpn:  # anchors only (the NWD paper's recommended placement, arXiv 2110.13389 sec. 5); RoI heads keep IoU
            self.rpn.box_similarity = iou_or_nwd

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
