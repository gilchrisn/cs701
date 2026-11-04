"""Boxes + image-class probabilities from ONE forward pass of an imgcls_mod model (the predictor's own copy holds the neck maps).
Usage: python pred_imgcls.py <stripped weights> <images.json> <image root> <out dir>  -> predictions.json (COCO) + class_probs.json"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imgcls_mod
from ultralytics import YOLO
w, ij, root, out = sys.argv[1:5]
ims = json.load(open(ij)); ims = ims['images'] if isinstance(ims, dict) else ims
m = YOLO(w); preds, probs = [], {}
for k in range(0, len(ims), 32):
    b = ims[k:k + 32]
    res = m.predict([os.path.join(root, i['file_name']) for i in b], imgsz=512, conf=1e-5, max_det=100, device=0, verbose=False, batch=len(b))
    p = imgcls_mod.image_probs(m.predictor.model.model.model[-1]).cpu().tolist()
    assert len(p) == len(b)
    for im, r, pi in zip(b, res, p):
        probs[im['id']] = pi
        for (x1, y1, x2, y2), s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            if x2 > x1 and y2 > y1: preds.append({'image_id': im['id'], 'category_id': int(c) + 1, 'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': s})
os.makedirs(out, exist_ok=True)
json.dump(preds, open(f'{out}/predictions.json', 'w')); json.dump(probs, open(f'{out}/class_probs.json', 'w'))
print(len(ims), 'images', len(preds), 'boxes ->', out)
