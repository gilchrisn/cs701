"""Data folder of a finished run. Runs record the --data path they were trained with; that path is stale when the data has moved
since or when the run was trained on another machine. SARBENCH_MOVED maps old to new prefixes: "old1=new1;old2=new2"."""
import os
from pathlib import Path

MOVED = dict(pair.split('=', 1) for pair in os.environ.get('SARBENCH_MOVED', '').split(';') if '=' in pair)


def data_root(recorded):
    p = str(recorded)
    if not Path(p).exists():
        for old, new in MOVED.items():
            if p.startswith(old):
                return Path(new + p[len(old):])
    return Path(p)
