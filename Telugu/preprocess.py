"""Step 1 - feature extraction (Telugu).

    python preprocess.py [--workers 4]

Reads ../data/Telugu/raw/{train,val}.tsv and writes mel / pitch / energy /
phoneme files + stats to ../data/Telugu/preprocessed/.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from fs2.preprocess_core import preprocess  # noqa: E402
from fs2.utils import load_config  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    a = ap.parse_args()
    preprocess(load_config(a.config), num_workers=a.workers)
