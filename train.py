"""Train one model jointly for classification and detection, then score it on val and test.
Example: python train.py --backbone vit --init pretrained --adapt lora"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sarbench.adapters import add_adapters
from sarbench.backbones import build_backbone
from sarbench.data import SARMultiTask, collate, to_original_xywh
from sarbench.metrics import classification_metrics, detection_metrics
from sarbench.model import MultiTaskModel


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'dataset' / 'SARFact-Course-20K')
    p.add_argument('--backbone', required=True, choices=['vit', 'terramind'])
    p.add_argument('--init', required=True, choices=['pretrained', 'scratch'])
    p.add_argument('--adapt', required=True, choices=['full', 'lora', 'moelora'])
    p.add_argument('--epochs', type=int, default=24)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--accum', type=int, default=1, help='gradient accumulation: --batch-size / --accum images per forward pass')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--backbone-lr', type=float, help='for the trainable backbone parameters (default: --lr)')
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--eval-every', type=int, default=4, help='validate every N epochs, and after the last one')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, help='use only the first N images of every split (quick checks)')
    p.add_argument('--seed', type=int, default=0)
    # Candidate changes, each off by default (the reference model)
    p.add_argument('--spatial-prior', action='store_true', help='conv stem adding stride-4/8 detail to the neck')
    p.add_argument('--nwd', action='store_true', help='match anchors/proposals to ground truth by max(IoU, NWD)')
    p.add_argument('--nwd-rpn', action='store_true', help='max(IoU, NWD) for RPN anchors only; RoI heads keep IoU')
    p.add_argument('--iou-score', action='store_true', help='with --cascade: IoU-Net branch, score = p_cls x IoU')
    p.add_argument('--dynamic-rcnn', action='store_true',
                   help='Dynamic R-CNN: RoI IoU threshold and smooth-L1 beta follow training statistics')
    p.add_argument('--cascade', action='store_true', help='Cascade R-CNN with one extra RoI stage at IoU 0.6')
    p.add_argument('--unfreeze-last', type=int, default=0, help='also train the last N ViT blocks (with LoRA)')
    p.add_argument('--unfreeze-lr', type=float, default=2e-5, help='learning rate of the unfrozen pretrained weights')
    p.add_argument('--det-weight', type=float, default=1.0, help='weight of the Faster R-CNN losses vs cross-entropy')
    p.add_argument('--log-input', action='store_true', help='log-scale the 8-bit pixels before normalising')
    p.add_argument('--copy-paste', action='store_true', help='paste copies of small objects within each training image')
    p.add_argument('--out', type=Path, default=Path('runs'))
    p.add_argument('--resume', action='store_true', help='continue from <out>/checkpoint.pt if it exists')
    args = p.parse_args()
    if args.init == 'scratch' and args.adapt != 'full':
        p.error('--init scratch is only valid with --adapt full')
    if args.backbone_lr is None:
        args.backbone_lr = args.lr
    return args


def log(message, file):
    """Print to the console and to the run's log.txt."""
    print(message, flush=True)
    print(message, file=file, flush=True)


def make_loader(args, split):
    """Batches of `split`; only the training set is shuffled and augmented."""
    train = split == 'train'
    dataset = SARMultiTask(args.data, split, train=train, limit=args.limit, log_input=args.log_input,
                           copy_paste=args.copy_paste)
    return DataLoader(dataset, args.batch_size // args.accum, shuffle=train, num_workers=args.workers, collate_fn=collate,
                      persistent_workers=train and args.workers > 0)  # training workers live across epochs


def lr_factor(step, warmup, total):
    """Learning-rate multiplier: linear warmup over `warmup` steps, then cosine decay to 0 at step `total`."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))  # --epochs 1: no decay phase


def train_one_epoch(model, loader, optimizer, schedule, epoch, log_file, accum=1, det_weight=1.0):
    """One pass over the training set; the loss is cross-entropy plus the four Faster R-CNN losses."""
    model.train()
    optimizer.zero_grad()  # no gradient carried over from an incomplete accumulation group
    start, seen = time.time(), 0
    for step, (images, labels, targets) in enumerate(loader, 1):
        targets = [{'boxes': t['boxes'].cuda(), 'labels': t['labels'].cuda()} for t in targets]
        with torch.autocast('cuda', dtype=torch.bfloat16):  # bf16 has fp32's range: no loss scaling needed
            logits, _, det_losses = model(images.cuda(), targets)
            losses = {'classification': F.cross_entropy(logits, labels.cuda()), **det_losses}
        loss = sum(losses.values()) if det_weight == 1 else \
            losses['classification'] + det_weight * sum(v for k, v in losses.items() if k != 'classification')
        (loss / accum).backward()
        if step % accum == 0:  # one optimizer step per --batch-size images
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
        seen += len(images)
        if step % 50 == 0 or step == len(loader):
            parts = ' '.join(f'{name} {value.item():.3f}' for name, value in losses.items())
            log(f'epoch {epoch} iter {step}/{len(loader)} loss {loss.item():.3f} ({parts}) '
                f'lr {optimizer.param_groups[1]["lr"]:.2e} {seen / (time.time() - start):.1f} images/s', log_file)
            start, seen = time.time(), 0


@torch.no_grad()
def evaluate(model, loader, gt_json):
    """Classification and COCO detection metrics on one split, and the detections in COCO result format."""
    model.eval()
    labels, predicted, detections, image_ids, probs = [], [], [], [], {}
    for images, y, targets in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, outputs, _ = model(images.cuda())
        labels += y.tolist()
        predicted += logits.argmax(dim=1).tolist()
        for target, pr in zip(targets, logits.float().softmax(dim=1).tolist()):
            probs[target['image_id']] = pr  # class probabilities, kept for post-hoc analysis
        for target, out in zip(targets, outputs):
            image_ids.append(target['image_id'])
            boxes = to_original_xywh(out['boxes'], target['orig_size']).tolist()
            detections += [{'image_id': target['image_id'], 'category_id': c, 'bbox': b, 'score': s}
                           for b, c, s in zip(boxes, out['labels'].tolist(), out['scores'].tolist())]
    metrics = {**classification_metrics(labels, predicted), **detection_metrics(gt_json, detections, image_ids)}
    return metrics, detections, probs


def headline(metrics):
    return ' '.join(f'{k} {metrics[k]:.4f}' for k in ('accuracy', 'macro_f1', 'balanced_accuracy', 'mAP', 'AP50'))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    run_dir = args.out
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / 'checkpoint.pt'
    resuming = args.resume and ckpt_path.exists()
    log_file = open(run_dir / 'log.txt', 'a' if resuming else 'w')
    log(json.dumps(vars(args), default=str), log_file)

    # Data
    train_loader, val_loader, test_loader = (make_loader(args, split) for split in ('train', 'val', 'test'))

    # Model: adapters are created on the CPU, so they are added before .cuda()
    backbone = build_backbone(args.backbone, pretrained=args.init == 'pretrained')
    if args.adapt != 'full':
        add_adapters(backbone, args.adapt)  # also freezes the backbone
    for blk in backbone.blocks[len(backbone.blocks) - args.unfreeze_last:] if args.unfreeze_last else []:
        blk.requires_grad_(True)  # partial fine-tuning of the last blocks, LoRA kept
    model = MultiTaskModel(backbone, task_routing=args.adapt == 'moelora', spatial_prior=args.spatial_prior,
                           nwd=args.nwd, nwd_rpn=args.nwd_rpn, dynamic_rcnn=args.dynamic_rcnn,
                           cascade=args.cascade, iou_score=args.iou_score).cuda()
    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'parameters: {parameters:,} total, {trainable:,} trainable', log_file)

    # Optimizer and schedule: trainable backbone parameters at --backbone-lr, all new modules at --lr
    is_unfrozen = lambda name: args.adapt != 'full' and '.down.' not in name and '.up.' not in name and 'router' not in name
    backbone_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad and not is_unfrozen(n)]
    unfrozen_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad and is_unfrozen(n)]
    head_params = [p for name, p in model.named_parameters() if not name.startswith('backbone.')]
    groups = [{'params': backbone_params, 'lr': args.backbone_lr}, {'params': head_params}]
    if unfrozen_params:
        groups.append({'params': unfrozen_params, 'lr': args.unfreeze_lr})
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    warmup, total = len(train_loader) // args.accum, args.epochs * (len(train_loader) // args.accum)  # one epoch of warmup
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, warmup, total))

    # Train, validating every --eval-every epochs; the reported model is the last-epoch model
    history, start, first_epoch, done_seconds = [], time.time(), 1, 0.0
    if resuming:  # a checkpoint is written after every epoch, so a crash loses at most one epoch
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        schedule.load_state_dict(ckpt['schedule'])
        torch.set_rng_state(ckpt['rng'])
        torch.cuda.set_rng_state(ckpt['cuda_rng'])
        history, first_epoch, done_seconds = ckpt['history'], ckpt['epoch'] + 1, ckpt['train_seconds']
        log(f'resumed after epoch {ckpt["epoch"]}', log_file)
    for epoch in range(first_epoch, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, schedule, epoch, log_file, args.accum, args.det_weight)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val, val_detections, val_probs = evaluate(model, val_loader, args.data / 'detection' / 'instances_val.json')
            history.append({'epoch': epoch, **val})
            log(f'epoch {epoch} val: {headline(val)}', log_file)
        tmp = run_dir / 'checkpoint.tmp'
        torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'schedule': schedule.state_dict(), 'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
                    'history': history, 'train_seconds': done_seconds + time.time() - start}, tmp)
        tmp.replace(ckpt_path)
    train_hours = (done_seconds + time.time() - start) / 3600

    # Evaluate on test, once
    test, test_detections, test_probs = evaluate(model, test_loader, args.data / 'detection' / 'instances_test.json')
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    log(f'test: {headline(test)}\ntraining {train_hours:.2f} h, peak GPU memory {peak_gb:.1f} GB', log_file)

    # Save
    summary = {'args': vars(args), 'parameters': parameters, 'trainable_parameters': trainable,
               'train_hours': train_hours, 'peak_gpu_memory_gb': peak_gb,
               'val_history': history, 'val': history[-1], 'test': test}
    (run_dir / 'metrics.json').write_text(json.dumps(summary, indent=2, default=str))
    (run_dir / 'predictions_val.json').write_text(json.dumps(val_detections))
    (run_dir / 'predictions_test.json').write_text(json.dumps(test_detections))
    (run_dir / 'class_probs_val.json').write_text(json.dumps(val_probs))
    torch.save(model.state_dict(), run_dir / 'model.pt')
    ckpt_path.unlink(missing_ok=True)  # the run is complete; model.pt is the result


if __name__ == '__main__':
    main()
