"""The course data in TA Zichen's folder layout (see README): every image has one class label and >= 1 box.
Loads images and boxes at a fixed input size and maps predicted boxes back to original pixels."""
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.ops import box_convert
from torchvision.transforms import v2

# Mean and std of all pixels of the 9,392 train images, read as gray, resized to
# 512 x 512 and scaled to [0, 1] (computed once).
MEAN, STD = 0.2526, 0.2133
# Optional (--log-input): pixels x in [0, 1] mapped to log1p(255 x) / log1p(255), which turns multiplicative speckle
# into additive noise; mean and std of that over our 7,984 training images at 512 px.
LOG_MEAN, LOG_STD = 0.6588, 0.2300


class SARMultiTask(Dataset):
    """The images of classification/{split}.csv with their boxes from detection/instances_{split}.json.
    An item is (image [1, size, size], label 0..8, target), target = {'boxes': XYXY on the resized
    image, 'labels': 1..9, 'image_id': COCO id, 'orig_size': (h, w)}."""

    def __init__(self, root, split, train=False, size=512, limit=None, log_input=False, copy_paste=False, scale_jitter=None):
        self.copy_paste = copy_paste if train else False  # False, 'same' (objects of the image itself) or 'other'
        self.scale_jitter = scale_jitter if train else None  # (lo, hi) or None
        self.root = Path(root)
        with open(self.root / 'classification' / f'{split}.csv') as f:
            self.rows = list(csv.DictReader(f))[:limit]
        coco = json.loads((self.root / 'detection' / f'instances_{split}.json').read_text())
        self.coco_images = {img['file_name']: img for img in coco['images']}  # key = CSV 'image'
        self.annotations = defaultdict(list)  # COCO image id -> its boxes
        for ann in coco['annotations']:
            self.annotations[ann['image_id']].append(ann)
        self.donors = defaultdict(list)  # (class, height, width) -> row indices; donors for copy_paste='other'
        for k, row in enumerate(self.rows):
            info = self.coco_images[row['image']]
            self.donors[(row['label'], info['height'], info['width'])].append(k)
        flips = [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()] if train else []
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ConvertBoundingBoxFormat('XYXY'),  # COCO [x, y, w, h] -> [x1, y1, x2, y2]
            v2.Resize((size, size)),  # stretches non-square images; boxes are scaled with the pixels
            *flips,  # overhead imagery has no canonical "up"
            v2.ToDtype(torch.float32, scale=True),
            *([v2.Lambda(lambda x: torch.log1p(255 * x) / math.log1p(255), tv_tensors.Image)] if log_input else []),  # pixels only: without the type, v2.Lambda also transformed the boxes
            v2.Normalize([LOG_MEAN], [LOG_STD]) if log_input else v2.Normalize([MEAN], [STD]),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        info = self.coco_images[row['image']]
        anns = self.annotations[info['id']]
        orig_size = (info['height'], info['width'])
        boxes = tv_tensors.BoundingBoxes([a['bbox'] for a in anns], format='XYWH', canvas_size=orig_size,
                                         dtype=torch.float32)  # integer boxes would be truncated by Resize
        image = Image.open(self.root / row['image']).convert('L')  # PIL: torchvision.io cannot read .bmp
        labels = [a['category_id'] for a in anns]
        if self.copy_paste == 'same':
            image, boxes, labels = paste_small_objects(image, boxes, labels, orig_size)
        elif self.copy_paste == 'other' and (donor := self.donor(i, row['label'], orig_size)):  # no donor: no paste
            image, boxes, labels = paste_small_objects(image, boxes, labels, orig_size, donor)
        image, boxes = self.transform(image, boxes)
        labels = torch.tensor(labels)
        if self.scale_jitter:
            image, boxes, labels = scale_jitter(image, boxes, labels, self.scale_jitter)
        target = {'boxes': boxes, 'labels': labels,
                  'image_id': info['id'], 'orig_size': orig_size}
        return image, int(row['label']), target

    def donor(self, i, label, orig_size):
        """Another training image of the same class and the same original size (so the image label stays right and the
        pixels share a scale), as (PIL image, XYWH boxes, labels); None if there is no such image."""
        group = self.donors[(label, *orig_size)]
        if len(group) < 2:
            return None
        j = i
        while j == i:
            j = random.choice(group)
        info = self.coco_images[self.rows[j]['image']]
        anns = self.annotations[info['id']]
        return (Image.open(self.root / self.rows[j]['image']).convert('L'), [a['bbox'] for a in anns],
                [a['category_id'] for a in anns])



def paste_small_objects(image, boxes, labels, orig_size, donor=None, p=0.5, max_side_frac=0.125, copies=3, tries=20):
    """Optional (--copy-paste, training only). With probability p, copy up to `copies` small objects of THIS image
    (box side <= max_side_frac of the image side, i.e. <= 64 px at 512) to random free spots of the same image, and add
    their boxes. Same image, so the image class and the sensor's statistics stay right (after Kisantal et al. 2019,
    "Augmentation for small object detection", which pasted with masks; we paste rectangles).
    donor = (image, XYWH boxes, labels) of another image of the same class and size (--paste-from-other): the objects
    are taken from it instead (cross-image copy-paste, as in Ghiasi et al. 2021 "Simple Copy-Paste", also with masks)."""
    if random.random() > p:
        return image, boxes, labels
    h, w = orig_size
    xywh = boxes.tolist()
    image = image.copy()
    src_image, src_xywh, src_labels = donor if donor else (image, xywh, labels)
    small = [k for k, (x, y, bw, bh) in enumerate(src_xywh) if max(bw, bh) <= max_side_frac * max(h, w) and bw >= 2 and bh >= 2]
    if not small:
        return image, boxes, labels
    occupied = [list(b) for b in xywh]
    new_boxes, new_labels = list(xywh), list(labels)
    for k in random.sample(small, min(copies, len(small))):
        x, y, bw, bh = src_xywh[k]
        x0, y0, x1, y1 = int(x), int(y), int(round(x + bw)), int(round(y + bh))
        patch = src_image.crop((x0, y0, x1, y1))
        pw, ph = patch.size
        for _ in range(tries):
            nx, ny = random.randint(0, max(w - pw, 0)), random.randint(0, max(h - ph, 0))
            if all(nx + pw <= ox or ox + ow <= nx or ny + ph <= oy or oy + oh <= ny for ox, oy, ow, oh in occupied):
                image.paste(patch, (nx, ny))
                occupied.append([nx, ny, pw, ph])
                new_boxes.append([nx, ny, pw, ph])
                new_labels.append(src_labels[k])
                break
    boxes = tv_tensors.BoundingBoxes(new_boxes, format='XYWH', canvas_size=orig_size, dtype=torch.float32)
    return image, boxes, new_labels


def scale_jitter(image, boxes, labels, scale_range, tries=10):
    """Optional (--scale-jitter LO HI, training only). Resize the normalised [1, S, S] image by s ~ U(LO, HI), then cut a
    random S x S window (s > 1) or place it at a random spot on an S x S canvas of 0, the dataset mean after
    normalisation (s < 1); boxes move with the pixels, are clipped to the window and dropped if under 1 px wide or high.
    A window that keeps no box is redrawn (the image label needs its objects), up to `tries` times, else the image is
    left as it is. Large-scale jitter (Ghiasi et al. 2021, used by ViTDet) with a narrower range for our tiny objects."""
    size = image.shape[-1]
    for _ in range(tries):
        n = max(round(size * random.uniform(*scale_range)), 1)
        scaled = F.interpolate(image[None], size=(n, n), mode='bilinear', align_corners=False, antialias=n < size)[0]
        if n >= size:  # crop
            x0, y0 = random.randint(0, n - size), random.randint(0, n - size)
            out = scaled[:, y0:y0 + size, x0:x0 + size]
        else:  # pad
            x0, y0 = -random.randint(0, size - n), -random.randint(0, size - n)
            out = image.new_zeros(image.shape)
            out[:, -y0:-y0 + n, -x0:-x0 + n] = scaled
        b = (boxes.as_subclass(torch.Tensor) * (n / size) - boxes.new_tensor([x0, y0, x0, y0])).clamp(0, size)
        keep = ((b[:, 2] - b[:, 0]) >= 1) & ((b[:, 3] - b[:, 1]) >= 1)
        if keep.any():
            return out, tv_tensors.BoundingBoxes(b[keep], format='XYXY', canvas_size=(size, size)), labels[keep]
    return image, boxes, labels


def collate(batch):
    """Stack images and labels; targets stay a list because images have different numbers of boxes."""
    images, labels, targets = zip(*batch)
    return torch.stack(images), torch.tensor(labels), list(targets)


def to_original_xywh(boxes_xyxy, orig_size, size=512):
    """Undo the resize: XYXY boxes on the size x size input -> COCO [x, y, w, h] in original pixels."""
    h, w = orig_size
    return box_convert(boxes_xyxy * boxes_xyxy.new_tensor([w, h, w, h]) / size, 'xyxy', 'xywh')
