"""Quick check: does the aligner loss differ between the training-loop setting
(train mode, grads on, batches of 16) and evaluation (eval mode, no grad)?

    python check_align_mode.py --device cuda:0
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from fs2.dataset import TTSDataset, collate, BucketSampler  # noqa: E402
from fs2.model.loss import FastSpeech2Loss  # noqa: E402
from fs2.train_core import run_model, to_device  # noqa: E402
from fs2.utils import load_config, load_meta_from_preprocessed, build_model, get_device  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "latest.pth"))
ap.add_argument("--device", default=None)
ap.add_argument("--n_batches", type=int, default=8)
a = ap.parse_args()

cfg = load_config(a.config)
device = get_device(a.device)
meta = load_meta_from_preprocessed(cfg)
model = build_model(cfg, meta).to(device)
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"])
loss_fn = FastSpeech2Loss(cfg["train"])
ds = TTSDataset("train", cfg)
loader = DataLoader(ds, batch_sampler=BucketSampler(ds.lengths(), cfg["train"]["optimizer"]["batch_size"], seed=1),
                    collate_fn=collate, num_workers=0)
print(f"checkpoint step {ck['step']}  torch {torch.__version__}  device {device}")
for i, batch in enumerate(loader):
    if i >= a.n_batches:
        break
    batch = to_device(batch, device)
    res = {}
    for name, train_mode, grad in [("train+grad", True, True), ("eval+nograd", False, False),
                                   ("train+nograd", True, False), ("eval+grad", False, True)]:
        model.train(train_mode)
        with torch.set_grad_enabled(grad):
            out = run_model(model, batch)
            _, p = loss_fn(batch, out, 10000)
        res[name] = p["align_ctc"]
        model.zero_grad(set_to_none=True)
    print(f"batch {i}: " + "  ".join(f"{k}={v:.3f}" for k, v in res.items()))
