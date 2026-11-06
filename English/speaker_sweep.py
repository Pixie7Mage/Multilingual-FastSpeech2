"""Pick the most intelligible speaker / speed by ASR word error rate (no retraining).

    pip install faster-whisper jiwer
    python speaker_sweep.py --ckpt checkpoints/best_fp16.pth --n_speakers 12

For each of the N speakers with the most training audio and each duration_control
value, synthesises a set of ordinary held-out sentences (validation transcripts,
never seen in training) plus the 5 test sentences, transcribes them with Whisper
and reports WER.  Ranking uses the held-out sentences only (the test sentences
are deliberately hard: rare words, heteronyms, a typo, a code-mixed word).
Writes ../Results/English/speaker_sweep.csv; best setting's audio goes to
../Results/English/speaker_sweep/<speaker>_d<x>/.
"""
import argparse
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from fs2.infer_core import read_sentences  # noqa: E402
from fs2.text import TextProcessor  # noqa: E402
from fs2.text import english  # noqa: E402
from fs2.utils import load_config, load_for_inference, get_vocoder, get_device  # noqa: E402


def norm(t):
    t = english.normalize(t).lower()  # numbers/times -> words, same as the TTS front-end
    t = re.sub(r"[^a-z' ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    cfg = load_config(os.path.join(HERE, "config.yaml"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "best_fp16.pth"))
    ap.add_argument("--n_speakers", type=int, default=12)
    ap.add_argument("--durations", default="1.0,1.15")
    ap.add_argument("--n_val", type=int, default=15, help="held-out validation sentences to score")
    ap.add_argument("--whisper", default="small.en")
    ap.add_argument("--device", default=None)
    ap.add_argument("--asr_device", default="auto", choices=["auto", "cuda", "cpu"])
    a = ap.parse_args()
    device = get_device(a.device)

    from faster_whisper import WhisperModel
    import jiwer
    asr = None
    if a.asr_device in ("auto", "cuda") and device.type == "cuda":
        try:  # faster-whisper's GPU build needs CUDA-12 cuBLAS, which CUDA-13 images lack
            asr = WhisperModel(a.whisper, device="cuda", compute_type="float16")
            list(asr.transcribe(np.zeros(16000, dtype=np.float32), language="en")[0])
        except Exception as e:
            print(f"[sweep] Whisper on GPU unavailable ({str(e).splitlines()[0]}); using CPU")
            asr = None
    if asr is None:
        asr = WhisperModel(a.whisper, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)

    model, mcfg, meta, step = load_for_inference(a.ckpt, device)
    tp = TextProcessor(mcfg["language"], **{**mcfg.get("text", {}), **cfg.get("text", {})})
    vocoder = get_vocoder(device)
    sr = mcfg["preprocessing"]["audio"]["sampling_rate"]

    # speakers ranked by amount of training audio
    pp = cfg["path"]["preprocessed_path"]
    frames = {}
    with open(os.path.join(pp, "train.txt"), encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("|")
            frames[p[1]] = frames.get(p[1], 0) + int(p[4])
    speakers = sorted(frames, key=frames.get, reverse=True)[: a.n_speakers]

    # held-out sentences: short-to-medium validation transcripts without digits
    val = []
    with open(os.path.join(pp, "val.txt"), encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("|")
            if not re.search(r"\d", p[3]) and 6 <= len(p[3].split()) <= 20:
                val.append((f"V{len(val):02d}", p[3]))
    rng = np.random.default_rng(0)
    val = [val[i] for i in sorted(rng.choice(len(val), min(a.n_val, len(val)), replace=False))]
    tests = read_sentences(os.path.join(HERE, "test_sentences.tsv"))
    print(f"[sweep] ckpt step {step}, {len(speakers)} speakers x {a.durations}, {len(val)} held-out + {len(tests)} test sentences")

    @torch.no_grad()
    def tts(text, spk, d):
        _, ids = tp(text)
        out = model(torch.tensor([meta["speakers"][spk]], device=device), torch.tensor([ids], device=device),
                    torch.tensor([len(ids)], device=device), d_control=d)
        wav = vocoder(out["postnet_mel"][0].float().T.unsqueeze(0)).squeeze().cpu().numpy()
        return np.clip(wav, -1, 1).astype(np.float32)

    def asr_text(wav):
        segs, _ = asr.transcribe(librosa.resample(wav, orig_sr=sr, target_sr=16000), language="en", beam_size=5)
        return " ".join(s.text.strip() for s in segs)

    rows, cache = [], {}
    for spk in speakers:
        for d in [float(x) for x in a.durations.split(",")]:
            res = {}
            for group, sents in [("val", val), ("test", tests)]:
                refs, hyps = [], []
                for sid, text in sents:
                    wav = tts(text, spk, d)
                    cache[(spk, d, sid)] = wav
                    refs.append(norm(text))
                    hyps.append(norm(asr_text(wav)) or "<empty>")
                res[group] = jiwer.wer(refs, hyps)
            rows.append({"speaker": spk, "duration_control": d, "train_min": round(frames[spk] * 256 / sr / 60, 1),
                         "wer_heldout": round(res["val"], 3), "wer_test": round(res["test"], 3)})
            print(f"  {spk} d={d}: held-out WER={res['val']:.1%}  test WER={res['test']:.1%}", flush=True)

    rows.sort(key=lambda r: r["wer_heldout"])
    out_csv = os.path.join(cfg["path"]["result_path"], "speaker_sweep.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    best = rows[0]
    bdir = os.path.join(cfg["path"]["result_path"], "speaker_sweep", f"{best['speaker']}_d{best['duration_control']}")
    os.makedirs(bdir, exist_ok=True)
    for (spk, d, sid), wav in cache.items():
        if spk == best["speaker"] and d == best["duration_control"]:
            sf.write(os.path.join(bdir, f"{sid}.wav"), wav, sr, subtype="PCM_16")
    print("\n=== top 5 (by held-out WER) ===")
    for r in rows[:5]:
        print(f"  {r['speaker']}  d={r['duration_control']}  held-out={r['wer_heldout']:.1%}  test={r['wer_test']:.1%}  ({r['train_min']} min train)")
    print(f"\n[sweep] wrote {out_csv}; best audio in {bdir}")
    print(f"Use it:  python inference.py --ckpt {a.ckpt} --speaker {best['speaker']} --duration_control {best['duration_control']}")


if __name__ == "__main__":
    main()
