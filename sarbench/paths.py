"""Data folder of a finished run. Runs record the --data path they were trained with; that path is stale when the data has moved
since (laptop: ~/cs701-data -> ~/cs701/data on 2026-10-19) or when the run was trained on the school server and pulled."""
from pathlib import Path

MOVED = {'/home/gilchris/cs701-data': '/home/gilchris/cs701/data',
         '/home/yudong/gilchris/sar_det/data': '/home/gilchris/cs701/data'}


def data_root(recorded):
    p = str(recorded)
    if not Path(p).exists():
        for old, new in MOVED.items():
            if p.startswith(old):
                return Path(new + p[len(old):])
    return Path(p)
