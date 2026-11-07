"""Training loop (adapted from the reference train.py / evaluate.py).

Differences from the reference:
  * durations come from the internal aligner (MAS), alignment losses added;
  * validation over the WHOLE 1-hour validation set every `val_step` (the
    reference used one batch), logged to CSV for the loss-curve plots;
  * auto-resume from checkpoints/latest.pth, best-validation checkpoint,
    fp16 inference-only export (<100 MB) at the end;
  * optional mixed precision on CUDA.
"""
import csv
import glob
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import TTSDataset, collate, BucketSampler
from .model.loss import FastSpeech2Loss
from .model.optimizer import ScheduledOptim
from .utils import build_model, load_meta_from_preprocessed, save_checkpoint, export_fp16, get_device
from .plots import plot_loss_curves

KEYS = ["total", "mel", "postnet_mel", "pitch", "energy", "duration", "align_ctc", "align_bin"]


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def run_model(model, batch):
    return model(batch["speakers"], batch["texts"], batch["src_lens"], batch["mels"], batch["mel_lens"],
                 batch["pitch"], batch["energy"], batch["attn_prior"])


@torch.no_grad()
def validate(model, loader, loss_fn, device, step):
    model.eval()
    agg, n = {k: 0.0 for k in KEYS}, 0
    first = None
    for batch in loader:
        batch = to_device(batch, device)
        out = run_model(model, batch)
        _, parts = loss_fn(batch, out, step)
        bs = batch["texts"].shape[0]
        for k in KEYS:
            agg[k] += parts[k] * bs
        n += bs
        if first is None:
            first = (batch, out)
    model.train()
    return {k: v / max(n, 1) for k, v in agg.items()}, first


def save_debug_plot(batch, out, path):
    i = 0
    tl, ml = int(batch["src_lens"][i]), int(batch["mel_lens"][i])
    fig, ax = plt.subplots(3, 1, figsize=(10, 9))
    ax[0].imshow(batch["mels"][i, :ml].cpu().numpy().T, origin="lower", aspect="auto")
    ax[0].set_title("Ground-truth mel")
    pm = out["postnet_mel"][i, :ml].float().cpu().numpy().T
    ax[1].imshow(pm, origin="lower", aspect="auto")
    ax[1].set_title("Predicted mel (teacher-forced durations)")
    ax[2].imshow(out["attn_soft"][i, 0, :ml, :tl].float().cpu().numpy().T, origin="lower", aspect="auto")
    ax[2].set_title("Learned soft alignment (text x frames)")
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close(fig)


def _append_csv(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step"] + KEYS + (["lr"] if "lr" in row else []))
        if new:
            w.writeheader()
        w.writerow(row)


def train(cfg, max_steps=None, device=None, num_workers=4):
    device = get_device(device)
    tc = cfg["train"]
    torch.manual_seed(tc.get("seed", 1234))
    np.random.seed(tc.get("seed", 1234))
    ckpt_dir, log_dir = cfg["path"]["ckpt_path"], cfg["path"]["log_path"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "debug"), exist_ok=True)

    meta = load_meta_from_preprocessed(cfg)
    train_set, val_set = TTSDataset("train", cfg), TTSDataset("val", cfg)
    bs = tc["optimizer"]["batch_size"]
    train_loader = DataLoader(train_set, batch_sampler=BucketSampler(train_set.lengths(), bs, seed=tc.get("seed", 1234)),
                              collate_fn=collate, num_workers=num_workers, pin_memory=device.type == "cuda",
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_set, batch_sampler=BucketSampler(val_set.lengths(), bs, shuffle=False),
                            collate_fn=collate, num_workers=min(2, num_workers))

    model = build_model(cfg, meta).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] device={device}  params={n_params/1e6:.2f}M  train={len(train_set)}  val={len(val_set)}")

    step, best_val = 0, float("inf")
    latest = os.path.join(ckpt_dir, "latest.pth")
    ck = None
    if os.path.exists(latest):
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        step = ck["step"]
        best_val = ck.get("best_val") or float("inf")
        print(f"[train] resumed from step {step}")
    optim = ScheduledOptim(model, tc, cfg["model"], step)
    if ck is not None and ck.get("optimizer"):
        optim.load_state_dict(ck["optimizer"])
    loss_fn = FastSpeech2Loss(tc)

    total_steps = max_steps or tc["step"]["total_step"]
    s = tc["step"]
    use_amp = bool(tc.get("fp16_run", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_acc = tc["optimizer"].get("grad_acc_step", 1)
    train_csv, val_csv = os.path.join(log_dir, "train_log.csv"), os.path.join(log_dir, "val_log.csv")

    run = {k: 0.0 for k in KEYS}
    nrun, t0 = 0, time.time()
    model.train()
    done = step >= total_steps
    try:
        while not done:
            for batch in train_loader:
                batch = to_device(batch, device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    out = run_model(model, batch)
                    loss, parts = loss_fn(batch, out, step)
                if not torch.isfinite(loss):
                    print(f"[train][warn] non-finite loss at step {step}, batch skipped")
                    optim.zero_grad()
                    continue
                scaler.scale(loss / grad_acc).backward()
                for k in KEYS:
                    run[k] += parts[k]
                nrun += 1
                if nrun % grad_acc:
                    continue
                scaler.unscale_(optim._optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["optimizer"]["grad_clip_thresh"])
                optim._update_learning_rate()
                scaler.step(optim._optimizer)
                scaler.update()
                optim.zero_grad()
                step += 1

                if step % s["log_step"] == 0:
                    avg = {k: run[k] / nrun for k in KEYS}
                    lr = optim._optimizer.param_groups[0]["lr"]
                    _append_csv(train_csv, {"step": step, **{k: round(v, 5) for k, v in avg.items()}, "lr": lr})
                    print(f"step {step}/{total_steps} | " + " ".join(f"{k}={avg[k]:.4f}" for k in KEYS)
                          + f" | lr={lr:.2e} | {(time.time()-t0)/s['log_step']:.2f}s/it", flush=True)
                    run, nrun, t0 = {k: 0.0 for k in KEYS}, 0, time.time()

                if step % s["val_step"] == 0 or step == total_steps:
                    vl, first = validate(model, val_loader, loss_fn, device, step)
                    _append_csv(val_csv, {"step": step, **{k: round(v, 5) for k, v in vl.items()}})
                    print(f"[val] step {step} | " + " ".join(f"{k}={vl[k]:.4f}" for k in KEYS), flush=True)
                    if first is not None and step % s.get("synth_step", s["val_step"]) == 0:
                        save_debug_plot(first[0], first[1], os.path.join(log_dir, "debug", f"step_{step:07d}.png"))
                    score = vl["mel"] + vl["postnet_mel"] + vl["duration"] + vl["pitch"] + vl["energy"]
                    if score < best_val and step >= tc.get("best_ckpt_min_step", 0):
                        best_val = score
                        save_checkpoint(os.path.join(ckpt_dir, "best.pth"), model, None, step, cfg, meta, best_val)
                    plot_loss_curves(log_dir, cfg["path"]["result_path"], cfg.get("language_name", ""))

                if step % s["save_step"] == 0 or step == total_steps:
                    save_checkpoint(latest, model, optim, step, cfg, meta, best_val)
                    save_checkpoint(os.path.join(ckpt_dir, f"step_{step:07d}.pth"), model, None, step, cfg, meta, best_val)
                    olds = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pth")))
                    for o in olds[:-tc.get("keep_last_ckpts", 2)]:
                        os.remove(o)
                if step >= total_steps:
                    done = True
                    break
    except KeyboardInterrupt:
        print("[train] interrupted - saving latest checkpoint")
        save_checkpoint(latest, model, optim, step, cfg, meta, best_val)
        raise

    save_checkpoint(latest, model, optim, step, cfg, meta, best_val)
    src = os.path.join(ckpt_dir, "latest.pth")
    export_fp16(src, os.path.join(ckpt_dir, "final_fp16.pth"))
    if os.path.exists(os.path.join(ckpt_dir, "best.pth")):
        export_fp16(os.path.join(ckpt_dir, "best.pth"), os.path.join(ckpt_dir, "best_fp16.pth"))
    plot_loss_curves(log_dir, cfg["path"]["result_path"], cfg.get("language_name", ""))
    print(f"[train] done at step {step}. Inference checkpoint: {os.path.join(ckpt_dir, 'final_fp16.pth')}")
