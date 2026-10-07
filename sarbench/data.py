"""The course data in TA Zichen's folder layout (see README): every image has one class label and >= 1 box.
Loads images and boxes at a fixed input size and maps predicted boxes back to original pixels."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
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

    def __init__(self, root, split, train=False, size=512, limit=None, log_input=False, copy_paste=False):
        self.copy_paste = copy_paste and train
        self.root = Path(root)
        with open(self.root / 'classification' / f'{split}.csv') as f:
            self.rows = list(csv.DictReader(f))[:limit]
        coco = json.loads((self.root / 'detection' / f'instances_{split}.json').read_text())
        self.coco_images = {img['file_name']: img for img in coco['images']}  # key = CSV 'image'
        self.annotations = defaultdict(list)  # COCO image id -> its boxes
        for ann in coco['annotations']:
            self.annotations[ann['image_id']].append(ann)
        flips = [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()] if train else []
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ConvertBoundingBoxFormat('XYXY'),  # COCO [x, y, w, h] -> [x1, y1, x2, y2]
            v2.Resize((size, size)),  # stretches non-square images; boxes are scaled with the pixels
            *flips,  # overhead imagery has no canonical "up"
            v2.ToDtype(torch.float32, scale=True),
            *([v2.Lambda(lambda x: torch.log1p(255 * x) / math.log1p(255))] if log_input else []),
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
        if self.copy_paste:
            image, boxes, labels = paste_small_objects(image, boxes, labels, orig_size)
        image, boxes = self.transform(image, boxes)
        target = {'boxes': boxes, 'labels': torch.tensor(labels),
                  'image_id': info['id'], 'orig_size': orig_size}
        return image, int(row['label']), target



def paste_small_objects(image, boxes, labels, orig_size, p=0.5, max_side_frac=0.125, copies=3, tries=20):
    """Optional (--copy-paste, training only). With probability p, copy up to `copies` small objects of THIS image
    (box side <= max_side_frac of the image side, i.e. <= 64 px at 512) to random free spots of the same image, and add
    their boxes. Same image, so the image class and the sensor's statistics stay right (after Kisantal et al. 2019,
    "Augmentation for small object detection", which pasted with masks; we paste rectangles)."""
    import random
    if random.random() > p:
        return image, boxes, labels
    h, w = orig_size
    xywh = boxes.tolist()
    small = [k for k, (x, y, bw, bh) in enumerate(xywh) if max(bw, bh) <= max_side_frac * max(h, w) and bw >= 2 and bh >= 2]
    if not small:
        return image, boxes, labels
    image = image.copy()
    occupied = [list(b) for b in xywh]
    new_boxes, new_labels = list(xywh), list(labels)
    for k in random.sample(small, min(copies, len(small))):
        x, y, bw, bh = xywh[k]
        x0, y0, x1, y1 = int(x), int(y), int(round(x + bw)), int(round(y + bh))
        patch = image.crop((x0, y0, x1, y1))
        pw, ph = patch.size
        for _ in range(tries):
            nx, ny = random.randint(0, max(w - pw, 0)), random.randint(0, max(h - ph, 0))
            if all(nx + pw <= ox or ox + ow <= nx or ny + ph <= oy or oy + oh <= ny for ox, oy, ow, oh in occupied):
                image.paste(patch, (nx, ny))
                occupied.append([nx, ny, pw, ph])
                new_boxes.append([nx, ny, pw, ph])
                new_labels.append(labels[k])
                break
    boxes = tv_tensors.BoundingBoxes(new_boxes, format='XYWH', canvas_size=orig_size, dtype=torch.float32)
    return image, boxes, new_labels


def collate(batch):
    """Stack images and labels; targets stay a list because images have different numbers of boxes."""
    images, labels, targets = zip(*batch)
    return torch.stack(images), torch.tensor(labels), list(targets)


def to_original_xywh(boxes_xyxy, orig_size, size=512):
    """Undo the resize: XYXY boxes on the size x size input -> COCO [x, y, w, h] in original pixels."""
    h, w = orig_size
    return box_convert(boxes_xyxy * boxes_xyxy.new_tensor([w, h, w, h]) / size, 'xyxy', 'xywh')
