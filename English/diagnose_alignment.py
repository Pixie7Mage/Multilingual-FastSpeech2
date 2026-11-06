"""Diagnose aligner failures: per-utterance alignment loss on train vs val.

    python diagnose_alignment.py [--ckpt checkpoints/latest.pth] [--n_plots 4]

Writes ../Results/<lang>/logs/align_diag/{per_utt.csv, *.png} and prints:
  * train vs val distribution of the forward-sum (CTC) loss, frames/phoneme, length
  * the worst utterances (with transcript) -> listen to these wavs to check that
    the transcript actually matches the audio
  * mean CTC per speaker (a few bad speakers => bad recordings / transcripts)
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from fs2.dataset import TTSDataset, collate  # noqa: E402
from fs2.model.aligner import ForwardSumLoss, binarize_attention  # noqa: E402
from fs2.utils import load_config, load_meta_from_preprocessed, build_model, get_device  # noqa: E402


@torch.no_grad()
def score_split(model, ds, device, ctc):
    rows = []
    for i in range(len(ds)):
        b = collate([ds[i]])
        texts, src_lens = b["texts"].to(device), b["src_lens"].to(device)
        mels, mel_lens, prior = b["mels"].to(device), b["mel_lens"].to(device), b["attn_prior"].to(device)
        emb = model.align_text_emb(texts)
        soft, logprob = model.aligner(mels.transpose(1, 2), emb.transpose(1, 2), None, prior)
        loss = ctc(logprob, src_lens, mel_lens).item()
        hard = binarize_attention(soft, src_lens, mel_lens)
        dur = hard.sum(2)[0, 0].cpu().numpy()
        # fraction of tokens that MAS gave exactly 1 frame (degenerate alignment indicator)
        one = float((dur[: int(src_lens[0])] <= 1).mean())
        maxshare = float(dur.max() / max(1, int(mel_lens[0])))
        uid, spk, phones, text, n = ds.items[i]
        rows.append({"id": uid, "speaker": spk, "ctc": loss, "n_phones": len(phones), "n_frames": n,
                     "frames_per_phone": n / max(1, len(phones)), "frac_1frame": one,
                     "max_token_share": maxshare, "text": text, "_soft": soft[0, 0].cpu().numpy()})
    return rows


def summarize(name, rows):
    c = np.array([r["ctc"] for r in rows])
    f = np.array([r["frames_per_phone"] for r in rows])
    d = np.array([r["frac_1frame"] for r in rows])
    m = np.array([r["max_token_share"] for r in rows])
    print(f"\n=== {name}: {len(rows)} utts ===")
    print(f"  ctc            mean={c.mean():.2f} median={np.median(c):.2f} p90={np.percentile(c, 90):.2f} "
          f"frac>5={np.mean(c > 5):.1%}")
    print(f"  frames/phone   mean={f.mean():.2f} min={f.min():.2f} max={f.max():.2f}")
    print(f"  tokens w/ 1 frame (MAS) mean={d.mean():.1%}   largest token share of frames mean={m.mean():.1%}")


def plot(r, path):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(r["_soft"].T, origin="lower", aspect="auto")
    ax.set_title(f"{r['id']} ctc={r['ctc']:.2f}  | {r['text'][:90]}", fontsize=8)
    ax.set_xlabel("frames")
    ax.set_ylabel("tokens")
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "latest.pth"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--n_plots", type=int, default=4)
    ap.add_argument("--n_worst", type=int, default=15)
    a = ap.parse_args()

    cfg = load_config(a.config)
    device = get_device(a.device)
    meta = load_meta_from_preprocessed(cfg)
    model = build_model(cfg, meta)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()})
    model.eval().to(device)
    print(f"[diag] checkpoint step {ck.get('step')}")

    out_dir = os.path.join(cfg["path"]["log_path"], "align_diag")
    os.makedirs(out_dir, exist_ok=True)
    ctc = ForwardSumLoss()
    res = {s: score_split(model, TTSDataset(s, cfg), device, ctc) for s in ["train", "val"]}

    for s, rows in res.items():
        summarize(s, rows)
    for s, rows in res.items():
        rows_sorted = sorted(rows, key=lambda r: r["ctc"])
        print(f"\n--- worst {a.n_worst} {s} utterances (listen to these: data/<lang>/raw/wavs/<id>.wav) ---")
        for r in rows_sorted[-a.n_worst:][::-1]:
            print(f"  {r['id']}  ctc={r['ctc']:.2f}  f/ph={r['frames_per_phone']:.1f}  {r['text'][:100]}")
        for k, r in enumerate(rows_sorted[: a.n_plots // 2] + rows_sorted[-(a.n_plots - a.n_plots // 2):]):
            plot(r, os.path.join(out_dir, f"{s}_{'best' if k < a.n_plots // 2 else 'worst'}_{r['id']}.png"))

    spk = defaultdict(lambda: defaultdict(list))
    for s, rows in res.items():
        for r in rows:
            spk[r["speaker"]][s].append(r["ctc"])
    print("\n--- speakers by mean val ctc (worst 10) ---")
    ranked = sorted(spk.items(), key=lambda kv: -np.mean(kv[1]["val"]) if kv[1]["val"] else 0)
    for name, d in ranked[:10]:
        print(f"  {name}: train={np.mean(d['train']) if d['train'] else float('nan'):.2f} (n={len(d['train'])})"
              f"  val={np.mean(d['val']):.2f} (n={len(d['val'])})")

    with open(os.path.join(out_dir, "per_utt.csv"), "w", newline="", encoding="utf-8") as f:
        keys = [k for k in res["train"][0] if not k.startswith("_")]
        w = csv.DictWriter(f, fieldnames=["split"] + keys)
        w.writeheader()
        for s, rows in res.items():
            for r in rows:
                w.writerow({"split": s, **{k: r[k] for k in keys}})
    print(f"\n[diag] wrote {out_dir}/per_utt.csv and alignment plots")


if __name__ == "__main__":
    main()
