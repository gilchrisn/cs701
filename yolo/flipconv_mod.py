"""Flip augmentation that maps boxes with the dataset's own formula. In the labelled data, flipped copies follow x' = W - 1 - x on both
box edges (measured by analysis/flip_convention.py), while Ultralytics' RandomFlip maps x' = W - x, so every flipped training image would
otherwise have its boxes 1 original pixel off.
patch(): after Ultralytics flips the boxes, shift them by -1 ORIGINAL pixel along the flipped axis (= -r training px, r = letterbox
resize ratio = min(img_h / h0, img_w / w0)); images are flipped exactly as before. Use with flipud 0.5 (variant 'flipconv')."""
import numpy as np
from ultralytics.data.augment import RandomFlip

_orig = RandomFlip.apply_instances


def _apply_instances(self, labels, params):
    labels = _orig(self, labels, params)
    if params['flip'] and params['direction'] in ('horizontal', 'vertical'):
        inst = labels['instances']
        h, w = labels['img'].shape[:2]
        h0, w0 = labels['ori_shape']
        r = min(h / h0, w / w0)
        axis = 0 if params['direction'] == 'horizontal' else 1
        d = r / (w if axis == 0 else h) if inst.normalized else r
        inst.convert_bbox(format='xywh')
        inst.bboxes[:, axis] -= d
        labels['instances'] = inst
    return labels


def patch():
    RandomFlip.apply_instances = _apply_instances


if __name__ == '__main__':  # unit check: a 256 px original letterboxed to 512 (r = 2), box [10, 20, 30, 50] (xyxy, original px)
    from ultralytics.utils.instance import Instances
    patch()
    for direction in ('horizontal', 'vertical'):
        b = np.array([[10, 20, 30, 50]], dtype=np.float32) * 2  # at training scale
        inst = Instances(b.copy(), bbox_format='xyxy', normalized=False)
        lab = {'img': np.zeros((512, 512, 3), np.uint8), 'instances': inst, 'ori_shape': (256, 256)}
        f = RandomFlip(p=1.0, direction=direction)
        lab = f.apply_instances(lab, {'flip': True, 'h': 512, 'w': 512, 'direction': direction, 'flip_idx': None})
        lab['instances'].convert_bbox(format='xyxy')
        got = lab['instances'].bboxes[0] / 2  # back to original px
        x0, y0, x1, y1 = 10, 20, 30, 50
        want = [256 - 1 - x1, y0, 256 - 1 - x0, y1] if direction == 'horizontal' else [x0, 256 - 1 - y1, x1, 256 - 1 - y0]
        print(direction, 'got', np.round(got, 3).tolist(), 'dataset convention', want, 'match', np.allclose(got, want))
        assert np.allclose(got, want)
