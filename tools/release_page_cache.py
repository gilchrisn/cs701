"""WSL-only workaround: GPU memory under WSL is backed by Windows commit charge, and the Linux page cache of the
VM holds commit. Every --every seconds, ask the kernel to drop the cached pages of the given directories
(posix_fadvise DONTNEED; no root needed). Example: python tools/release_page_cache.py ~/cs701-data ~/cs701-venv"""
import os
import sys
import time

every = 120
dirs = [os.path.expanduser(d) for d in sys.argv[1:]]
while True:
    for root in dirs:
        for d, _, files in os.walk(root):
            for f in files:
                p = os.path.join(d, f)
                try:
                    if os.path.islink(p):
                        continue
                    fd = os.open(p, os.O_RDONLY)
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    os.close(fd)
                except OSError:
                    pass
    time.sleep(every)
