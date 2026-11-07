"""Feature extraction (adapted from the reference preprocessor/preprocessor.py).

For every utterance in raw_dir/{train,val}.tsv:
  text  -> normalised text -> phoneme ids (language front-end)
  audio -> 22.05 kHz, silence-trimmed, peak-normalised
        -> 80-bin log-mel (HiFi-GAN compatible), frame energy, F0 (WORLD dio+stonemask)
F0 unvoiced frames are linearly interpolated; F0 and energy are z-normalised
with training-set statistics (as in the reference).  Unlike the reference,
features are stored at FRAME level; phoneme-level averages are computed during
training from the learned alignment (no MFA TextGrids needed).
"""
import json
import os
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from .audio import load_wav, trim_and_normalize, mel_spectrogram, extract_pitch, interpolate_unvoiced
from .text import TextProcessor

_CFG = None
_TP = None


def _init(cfg):
    global _CFG, _TP
    _CFG = cfg
    _TP = TextProcessor(cfg["language"], **cfg.get("text", {}))


def _process(item):
    uid, spk, _, text = item
    cfg = _CFG
    out_dir = cfg["path"]["preprocessed_path"]
    sr = cfg["preprocessing"]["audio"]["sampling_rate"]
    try:
        phones, ids = _TP(text)
    except Exception as e:
        return None, f"{uid}: text error {e}"
    wav = load_wav(os.path.join(cfg["path"]["raw_path"], "wavs", uid + ".wav"), sr)
    wav = trim_and_normalize(wav, sr, top_db=cfg["preprocessing"]["audio"].get("trim_top_db", 40))
    mel, energy = mel_spectrogram(wav, cfg)
    T = mel.shape[0]
    max_frames = cfg["model"]["max_seq_len"]
    if T > max_frames:
        return None, f"{uid}: too long ({T} frames)"
    if T < len(ids) * 1.2:
        return None, f"{uid}: fewer frames than phonemes"
    f0 = extract_pitch(wav, T, cfg)
    if (f0 > 0).sum() < 5:
        return None, f"{uid}: unvoiced"
    f0_i = interpolate_unvoiced(f0)
    np.save(os.path.join(out_dir, "mel", uid + ".npy"), mel)
    np.save(os.path.join(out_dir, "pitch", uid + ".npy"), f0_i)
    np.save(os.path.join(out_dir, "energy", uid + ".npy"), energy)
    return {"id": uid, "speaker": spk, "text": text, "phones": " ".join(phones), "n_frames": int(T)}, None


def _read_tsv(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                items.append((p[0], p[1], p[2], "\t".join(p[3:])))
    return items


def preprocess(cfg, num_workers=4):
    out_dir = cfg["path"]["preprocessed_path"]
    for d in ["mel", "pitch", "energy"]:
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)
    splits = {s: _read_tsv(os.path.join(cfg["path"]["raw_path"], f"{s}.tsv")) for s in ["train", "val"]}
    results, log = {}, []
    for split, items in splits.items():
        res = []
        if num_workers > 1:
            with Pool(num_workers, initializer=_init, initargs=(cfg,)) as pool:
                for r, err in tqdm(pool.imap_unordered(_process, items, chunksize=8), total=len(items), desc=split):
                    (res.append(r) if r else log.append(err))
        else:
            _init(cfg)
            for it in tqdm(items, desc=split):
                r, err = _process(it)
                (res.append(r) if r else log.append(err))
        results[split] = sorted(res, key=lambda r: r["id"])

    # ---- statistics on TRAIN only
    hop, sr = cfg["preprocessing"]["stft"]["hop_length"], cfg["preprocessing"]["audio"]["sampling_rate"]
    pv, ev = [], []
    for r in results["train"]:
        pv.append(np.load(os.path.join(out_dir, "pitch", r["id"] + ".npy")))
        ev.append(np.load(os.path.join(out_dir, "energy", r["id"] + ".npy")))
    pv, ev = np.concatenate(pv), np.concatenate(ev)
    p_mean, p_std = float(pv.mean()), float(pv.std())
    e_mean, e_std = float(ev.mean()), float(ev.std())

    # ---- normalise in place, track min/max
    pmin = emin = np.inf
    pmax = emax = -np.inf
    for split in results:
        for r in results[split]:
            for name, m, s in [("pitch", p_mean, p_std), ("energy", e_mean, e_std)]:
                f = os.path.join(out_dir, name, r["id"] + ".npy")
                v = (np.load(f) - m) / s
                np.save(f, v.astype(np.float32))
                if name == "pitch":
                    pmin, pmax = min(pmin, v.min()), max(pmax, v.max())
                else:
                    emin, emax = min(emin, v.min()), max(emax, v.max())

    speakers = sorted({r["speaker"] for s in results for r in results[s]})
    tp = TextProcessor(cfg["language"], **cfg.get("text", {}))
    stats = {"pitch": [float(pmin), float(pmax), p_mean, p_std],
             "energy": [float(emin), float(emax), e_mean, e_std]}
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(out_dir, "speakers.json"), "w") as f:
        json.dump({s: i for i, s in enumerate(speakers)}, f, indent=2)
    with open(os.path.join(out_dir, "symbols.json"), "w", encoding="utf-8") as f:
        json.dump(tp.symbols, f, ensure_ascii=False)
    for split in results:
        with open(os.path.join(out_dir, f"{split}.txt"), "w", encoding="utf-8") as f:
            for r in results[split]:
                f.write("|".join([r["id"], r["speaker"], r["phones"], r["text"], str(r["n_frames"])]) + "\n")
    with open(os.path.join(out_dir, "preprocess_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    summary = {s: {"utterances": len(results[s]),
                   "hours_after_trim": round(sum(r["n_frames"] for r in results[s]) * hop / sr / 3600, 3)}
               for s in results}
    summary["skipped"] = len(log)
    summary["n_speakers"] = len(speakers)
    summary["n_symbols"] = len(tp.symbols)
    summary["pitch_hz_mean_std"] = [round(p_mean, 2), round(p_std, 2)]
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary
