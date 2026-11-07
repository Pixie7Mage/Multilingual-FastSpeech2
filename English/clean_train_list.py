"""Prepare train.txt / val.txt for the v2 run without re-extracting any audio features.

    python clean_train_list.py [--per_utt ../Results/English/logs/align_diag/per_utt.csv] [--drop_pct 5]

1. The first time it runs, it backs up train.txt / val.txt as train_full.txt / val_full.txt.
   It always reads from those backups, so running it again is safe.
2. It re-phonemises every line whose transcript contains a digit with the current
   front-end (so long digit strings are read digit by digit).  Mel, pitch and energy
   are frame-level and do not depend on the phonemes, so they stay valid.
3. With --per_utt it drops the worst --drop_pct % of TRAIN clips by alignment
   (forward-sum) loss.  These are usually clips whose transcript does not match the
   audio.  The validation set is never filtered, so val losses and WER stay comparable.
"""
import argparse
import csv
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402

from fs2.text import TextProcessor  # noqa: E402
from fs2.utils import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--per_utt", default=None, help="per_utt.csv written by diagnose_alignment.py")
    ap.add_argument("--drop_pct", type=float, default=5.0)
    a = ap.parse_args()
    cfg = load_config(a.config)
    pp = cfg["path"]["preprocessed_path"]
    tp = TextProcessor(cfg["language"], **cfg.get("text", {}))

    bad = set()
    if a.per_utt:
        rows = [r for r in csv.DictReader(open(a.per_utt, encoding="utf-8")) if r["split"] == "train"]
        ctc = np.array([float(r["ctc"]) for r in rows])
        thr = np.percentile(ctc, 100 - a.drop_pct)
        bad = {r["id"] for r in rows if float(r["ctc"]) > thr}
        print(f"[clean] dropping {len(bad)} train clips with align ctc > {thr:.2f} (worst {a.drop_pct}%)")

    for split in ["train", "val"]:
        src, dst = os.path.join(pp, f"{split}_full.txt"), os.path.join(pp, f"{split}.txt")
        if not os.path.exists(src):
            shutil.copy(dst, src)
        out, n_re, n_drop, frames = [], 0, 0, 0
        for line in open(src, encoding="utf-8"):
            uid, spk, phones, text, n = line.rstrip("\n").split("|")
            if split == "train" and uid in bad:
                n_drop += 1
                continue
            if any(c.isdigit() for c in text):
                new = " ".join(tp(text)[0])
                n_re += new != phones
                phones = new
            out.append("|".join([uid, spk, phones, text, n]))
            frames += int(n)
        with open(dst, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
        hrs = frames * cfg["preprocessing"]["stft"]["hop_length"] / cfg["preprocessing"]["audio"]["sampling_rate"] / 3600
        print(f"[clean] {split}: kept {len(out)} clips ({hrs:.2f} h), re-phonemised {n_re}, dropped {n_drop}")


if __name__ == "__main__":
    main()
