"""Overlap groups of the validation and test images, each compared with all 9,392 labelled images.
The held-out split's groups (tools/holdout15_seed0_scene_groups.json) were made earlier with the same rules by a script we did not keep;
they are stored, not recomputed here. Stored result of this script: results/test_groups.txt.
Rules:
  G2: max edge-cosine over {identity, h-flip, v-flip} of the query >= 0.8
  G1: not G2, ORB+RANSAC (identity or h-flipped query) >= 30 inliers spread over >= 8 of 64 cells, candidates = top-40 by 3 descriptors
  G0: rest (new scene)
"""
import argparse, json, os, sys, numpy as np, cv2
from PIL import Image, ImageFilter
from multiprocessing import Pool
cv2.setNumThreads(1)
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--data', required=True, help='data root holding holdout15_seed0/, val/ and test/ (images.json + images)')
ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'test_groups.json'))
ap.add_argument('--procs', type=int, default=20)
ap.add_argument('--limit', type=int, help='smoke test: only the first N val and N test images are queries')
args = ap.parse_args()
R = os.path.expanduser(args.data).rstrip('/') + '/'
tr = json.load(open(R + 'holdout15_seed0/detection/instances_train.json'))['images']
ho = json.load(open(R + 'holdout15_seed0/detection/instances_val.json'))['images']
va = json.load(open(R + 'val/images.json'))['images']
te = json.load(open(R + 'test/images.json'))['images']
if args.limit: va, te = va[:args.limit], te[:args.limit]
P = [R + 'holdout15_seed0/' + im['file_name'] for im in tr + ho] + [R + 'val/' + im['file_name'] for im in va] + \
    [R + 'test/' + im['file_name'] for im in te]
S = np.array(['lab'] * (len(tr) + len(ho)) + ['val'] * len(va) + ['test'] * len(te))

def edge(p, fl=None):
    g = Image.open(p).convert('L')
    if fl is not None: g = g.transpose(fl)
    e = g.resize((64, 64)).filter(ImageFilter.FIND_EDGES).resize((32, 32))
    a = np.asarray(e, dtype=np.float32).ravel(); a = (a - a.mean()) / (a.std() + 1e-8); return a / np.linalg.norm(a)

def glob(args):
    p, fl = args
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if fl: g = cv2.flip(g, 1)
    t = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel(); t = (t - t.mean()) / (t.std() + 1e-6)
    f = np.abs(np.fft.fft2(cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) - g.mean()))
    f = np.log1p(np.fft.fftshift(f)).ravel(); f = (f - f.mean()) / (f.std() + 1e-6)
    return t / np.linalg.norm(t), f / np.linalg.norm(f)

def orb(p, fl=False):
    g = cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (512, 512))
    if fl: g = cv2.flip(g, 1)
    k, d = cv2.ORB_create(nfeatures=1500, fastThreshold=10).detectAndCompute(g, None)
    return (np.array([kk.pt for kk in k], np.float32) if k else np.zeros((0, 2), np.float32)), d

def match(pq, dq, pc, dc):
    if dq is None or dc is None or len(dq) < 8 or len(dc) < 8: return 0, 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(dq, dc, k=2)
    good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.8 * b.distance]
    if len(good) < 8: return 0, 0
    src = pq[[g.queryIdx for g in good]]; dst = pc[[g.trainIdx for g in good]]
    H, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=6.0)
    if mask is None: return 0, 0
    k = mask.ravel() > 0
    occ = min(len(set(map(tuple, (src[k] // 64).astype(int).tolist()))), len(set(map(tuple, (dst[k] // 64).astype(int).tolist()))))
    return int(k.sum()), occ

def verify(job):
    qi, ci, cf = job
    for fl, cands in ((False, ci), (True, cf)):
        pq, dq = orb(P[qi], fl)
        for c in cands:
            pc, dc = orb(P[c]); n, occ = match(pq, dq, pc, dc)
            if n >= 30 and occ >= 8: return qi, True
    return qi, False

def top(Xq, Xr, ref, k=40):
    M = Xq @ Xr.T; t = np.argpartition(-M, k, axis=1)[:, :k]; return ref[t]

if __name__ == '__main__':
    lab = np.where(S == 'lab')[0]; Q = np.where(S != 'lab')[0]
    with Pool(args.procs) as p:
        E = np.stack(p.map(edge, P, chunksize=64))
        EH = np.stack(p.map(__import__('functools').partial(edge, fl=Image.FLIP_LEFT_RIGHT), [P[i] for i in Q], chunksize=32))
        EV = np.stack(p.map(__import__('functools').partial(edge, fl=Image.FLIP_TOP_BOTTOM), [P[i] for i in Q], chunksize=32))
        G = p.map(glob, [(x, False) for x in P], chunksize=64); GF = p.map(glob, [(P[i], True) for i in Q], chunksize=32)
    T = np.stack([g[0] for g in G]); F = np.stack([g[1] for g in G]); TF = np.stack([g[0] for g in GF]); FF = np.stack([g[1] for g in GF])
    cos = np.maximum.reduce([(E[Q] @ E[lab].T).max(1), (EH @ E[lab].T).max(1), (EV @ E[lab].T).max(1)])
    g2 = cos >= 0.8
    need = np.where(~g2)[0]
    ci = [set() for _ in need]; cf = [set() for _ in need]
    for Xq, Xr in ((E[Q[need]], E[lab]), (T[Q[need]], T[lab]), (F[Q[need]], F[lab])):
        t = top(Xq, Xr, lab)
        for r in range(len(need)): ci[r].update(t[r].tolist())
    for Xq, Xr in ((EH[need], E[lab]), (TF[need], T[lab]), (FF[need], F[lab])):
        t = top(Xq, Xr, lab)
        for r in range(len(need)): cf[r].update(t[r].tolist())
    jobs = [(int(Q[r]), sorted(ci[j]), sorted(cf[j])) for j, r in enumerate(need)]
    print('ORB jobs', len(jobs), flush=True)
    with Pool(args.procs) as p: res = dict(p.map(verify, jobs, chunksize=4))
    grp = np.array(['G2' if g2[r] else ('G1' if res[int(Q[r])] else 'G0') for r in range(len(Q))])
    out = {}
    for split in ('val', 'test'):
        m = S[Q] == split; n = m.sum()
        print(f"{split} n={n}: " + '  '.join(f"{k} {np.sum(grp[m] == k)} ({100 * np.mean(grp[m] == k):.1f}%)" for k in ('G0', 'G1', 'G2')))
    ids = [im['id'] for im in va] + [im['id'] for im in te]
    json.dump({'split': S[Q].tolist(), 'id': ids, 'group': grp.tolist(), 'cos': cos.round(4).tolist()},
              open(args.out, 'w'))
