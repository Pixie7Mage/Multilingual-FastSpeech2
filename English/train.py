"""Step 2 - train FastSpeech 2 (English).

    python train.py [--max_steps N] [--device cuda]

Resumes automatically from checkpoints/latest.pth.  At the end writes
checkpoints/final_fp16.pth (inference checkpoint) and
../Results/English/{training_loss.png, validation_loss.png}.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from fs2.train_core import train  # noqa: E402
from fs2.utils import load_config  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--max_steps", type=int, default=None, help="override train.step.total_step")
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    train(load_config(a.config), max_steps=a.max_steps, device=a.device, num_workers=a.workers)
