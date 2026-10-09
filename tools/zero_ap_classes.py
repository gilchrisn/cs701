import json, numpy as np, torch
from collections import defaultdict, Counter
from torchvision.ops import box_iou
R='/home/gilchris/cs701-runs/2026-10-09_cascade24_s0/'
g=json.load(open('/home/gilchris/cs701-data/holdout15_seed0/detection/instances_val.json')); names={c['id']:c['name'] for c in g['categories']}
dets=json.load(open(R+'predictions_val.json')); by=defaultdict(list)
for d in dets: by[d['image_id']].append(d)
xy=lambda b:[b[0],b[1],b[0]+b[2],b[1]+b[3]]
for cname in ('bridge','playground','wind_turbine','oil_tank'):
    best_same,best_any,any_cls=[],[],Counter()
    for a in g['annotations']:
        if names[a['category_id']]!=cname: continue
        D=by[a['image_id']]
        if not D: best_same.append(0); best_any.append(0); continue
        iou=box_iou(torch.tensor([xy(a['bbox'])]),torch.tensor([xy(d['bbox']) for d in D]))[0]
        same=torch.tensor([d['category_id']==a['category_id'] for d in D])
        best_same.append(float((iou*same).max())); j=int(iou.argmax()); best_any.append(float(iou[j]))
        if iou[j]>=0.5: any_cls[names[D[j]['category_id']]]+=1
    bs,ba=np.array(best_same),np.array(best_any)
    print(f"{cname:12s} n={len(bs)} same-class best IoU>=0.5 {np.mean(bs>=0.5):.2f} | ANY-class best IoU>=0.5 {np.mean(ba>=0.5):.2f}, class of that best box: {dict(any_cls.most_common(3))}")
for cname in ('bridge','playground','wind_turbine','oil_tank'):
    A=[a for a in g['annotations'] if names[a['category_id']]==cname]
    ar=[max(a['bbox'][2],a['bbox'][3])/max(1e-6,min(a['bbox'][2],a['bbox'][3])) for a in A]
    per_img=Counter(a['image_id'] for a in A)
    print(f"{cname:12s} median aspect {np.median(ar):.2f}, boxes per image median {np.median(list(per_img.values())):.0f}")
