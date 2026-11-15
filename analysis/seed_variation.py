"""Run-to-run variation of held-out mAP: root mean square over the eight pairs of runs that differ only in the random seed
(0.55 points in the report; one point = 0.01 mAP). Reads figures/data/seed_pairs.csv (held-out mAP of seed 0 and seed 1 of each pair,
taken from our scoring of the stored predictions, column 'full' of runs/bias/history_groups_2026-10-30.csv in our run store).
Usage: python analysis/seed_variation.py"""
import csv
import math
import os

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'data', 'seed_pairs.csv')
rows = list(csv.DictReader(open(path)))
gaps = [(float(r['mAP_seed0']) - float(r['mAP_seed1'])) * 100 for r in rows]
for r, g in zip(rows, gaps):
    print(f"{r['pair']:40s} seed 0 {float(r['mAP_seed0']):.4f}  seed 1 {float(r['mAP_seed1']):.4f}  gap {g:+.2f} points")
print(f'root mean square over {len(gaps)} pairs: {math.sqrt(sum(g * g for g in gaps) / len(gaps)):.2f} points of held-out mAP')
