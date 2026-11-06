import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COMPONENTS = ["total", "mel", "postnet_mel", "duration", "pitch", "energy", "align_ctc", "align_bin"]


def _read(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def _smooth(y, k=5):
    if len(y) < k:
        return y
    return np.convolve(np.pad(y, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def plot_loss_curves(log_dir, result_dir, name=""):
    tr, va = _read(os.path.join(log_dir, "train_log.csv")), _read(os.path.join(log_dir, "val_log.csv"))
    os.makedirs(result_dir, exist_ok=True)
    for data, other, fname, title in [(tr, None, "training_loss.png", "Training loss"),
                                      (va, tr, "validation_loss.png", "Validation loss")]:
        if data is None:
            continue
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        for ax, k in zip(axes.flat, COMPONENTS):
            if k not in data:
                ax.axis("off")
                continue
            if data is tr:
                ax.plot(data["step"], data[k], alpha=0.3, color="C0")
                ax.plot(data["step"], _smooth(data[k]), color="C0", label="train (smoothed)")
            else:
                ax.plot(data["step"], data[k], "o-", ms=3, color="C3", label="validation")
                if other is not None and k in other:
                    ax.plot(other["step"], _smooth(other[k]), color="C0", alpha=0.6, label="train (smoothed)")
            ax.set_title(k)
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            if k == "total" or k == "align_ctc":
                lo = np.percentile(data[k], 1)
                hi = np.percentile(data[k], 97)
                if hi > lo:
                    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.3 * (hi - lo))
        fig.suptitle(f"{name} FastSpeech 2 — {title}".strip())
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, fname), dpi=110)
        plt.close(fig)


# ----------------------------------------------------------------------------
# Variance-adaptor plots
# ----------------------------------------------------------------------------
def _xticks(ax, phones, durs, hop_s):
    centers, edges, t = [], [0.0], 0.0
    for d in durs:
        centers.append(t + d * hop_s / 2)
        t += d * hop_s
        edges.append(t)
    ax.set_xticks(centers)
    ax.set_xticklabels(phones, rotation=90, fontsize=7)
    for e in edges:
        ax.axvline(e, color="grey", lw=0.3, alpha=0.5)
    return np.array(edges)


def plot_variance(sid, text, phones, durs, pitch_hz, energy, mel, real_f0, real_energy, hop_s, out_dir, title_extra=""):
    """durs: frames per phoneme; pitch_hz / energy: per phoneme predictions (denormalised)."""
    T = mel.shape[0]
    t_frames = np.arange(T) * hop_s
    phone_frames = np.repeat(np.arange(len(phones)), durs)
    title = f"{sid}: {text}"
    if len(title) > 120:
        title = title[:117] + "..."

    # ---------------- duration
    fig, ax = plt.subplots(figsize=(max(10, len(phones) * 0.28), 4.2))
    x = np.arange(len(phones))
    colors = ["#bbbbbb" if p in ("_", ",", ".", "?", "!") else "C0" for p in phones]
    ax.bar(x, durs, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(phones, rotation=90, fontsize=7)
    ax.set_ylabel("predicted duration (frames)")
    ax2 = ax.twinx()
    ax2.set_ylim(np.array(ax.get_ylim()) * hop_s * 1000)
    ax2.set_ylabel("ms")
    ax.set_title(f"Duration — {title}\n total {durs.sum()} frames = {durs.sum()*hop_s:.2f} s {title_extra}", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.join(out_dir, "duration"), exist_ok=True)
    plt.savefig(os.path.join(out_dir, "duration", f"{sid}.png"), dpi=120)
    plt.close(fig)

    # ---------------- pitch & energy
    for kind, pred, real, ylabel in [("pitch", pitch_hz, real_f0, "F0 (Hz)"),
                                     ("energy", energy, real_energy, "energy (L2 norm of STFT frame)")]:
        fig, axes = plt.subplots(2, 1, figsize=(max(10, len(phones) * 0.28), 7),
                                 gridspec_kw={"height_ratios": [1, 1.2]}, sharex=True)
        axes[0].imshow(mel.T, origin="lower", aspect="auto", extent=[0, T * hop_s, 0, mel.shape[1]], cmap="magma")
        axes[0].set_ylabel("mel bin")
        axes[0].set_title(f"{kind.capitalize()} — {title}", fontsize=9)
        frame_pred = pred[phone_frames] if len(phone_frames) else np.zeros(T)
        ax = axes[1]
        ax.step(t_frames[:len(frame_pred)], frame_pred, where="post", color="C1", lw=2,
                label=f"predicted {kind} (phoneme level, variance adaptor)")
        if real is not None:
            r = real.astype(float).copy()
            if kind == "pitch":
                r[r <= 0] = np.nan
            ax.plot(t_frames[:len(r)], r[:T], color="C0", lw=0.8, alpha=0.7,
                    label=f"{kind} measured on the synthesised audio")
        _xticks(ax, phones, durs, hop_s)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        os.makedirs(os.path.join(out_dir, kind), exist_ok=True)
        plt.savefig(os.path.join(out_dir, kind, f"{sid}.png"), dpi=120)
        plt.close(fig)
