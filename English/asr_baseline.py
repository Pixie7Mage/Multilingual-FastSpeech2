"""Where is intelligibility lost?  ASR word error rate on the same held-out sentences for
   (1) real recordings, (2) real mel -> HiFi-GAN (vocoder only), (3) FastSpeech2 + HiFi-GAN.

    python asr_baseline.py --device cuda:0 --asr_device cpu [--n_val 30]

(1) is the ceiling for this ASR on Indian-accented speech, (2)-(1) is the vocoder's loss,
(3)-(2) is the acoustic model's loss.  (3) uses each sentence's own speaker.
"""
import argparse
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from fs2.model.aligner import beta_binomial_prior  # noqa: E402
from fs2.text import TextProcessor, english  # noqa: E402
from fs2.utils import load_config, load_for_inference, get_vocoder, get_device  # noqa: E402


def norm(t):
    t = english.normalize(t).lower()
    t = re.sub(r"[^a-z' ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    cfg = load_config(os.path.join(HERE, "config.yaml"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "best_fp16.pth"))
    ap.add_argument("--n_val", type=int, default=30)
    ap.add_argument("--whisper", default="small.en")
    ap.add_argument("--device", default=None)
    ap.add_argument("--asr_device", default="cpu", choices=["cpu", "cuda"])
    a = ap.parse_args()
    device = get_device(a.device)

    from faster_whisper import WhisperModel
    import jiwer
    asr = WhisperModel(a.whisper, device=a.asr_device, compute_type="int8" if a.asr_device == "cpu" else "float16",
                       cpu_threads=os.cpu_count() or 4)

    model, mcfg, meta, step = load_for_inference(a.ckpt, device)
    tp = TextProcessor(mcfg["language"], **{**mcfg.get("text", {}), **cfg.get("text", {})})
    vocoder = get_vocoder(device)
    sr = mcfg["preprocessing"]["audio"]["sampling_rate"]
    pp, raw = cfg["path"]["preprocessed_path"], cfg["path"]["raw_path"]

    items = []
    with open(os.path.join(pp, "val.txt"), encoding="utf-8") as f:
        for line in f:
            uid, spk, _, text, _ = line.rstrip("\n").split("|")
            if not re.search(r"\d", text) and 6 <= len(text.split()) <= 20:
                items.append((uid, spk, text))
    random.Random(0).shuffle(items)
    items = items[: a.n_val]

    def asr_text(wav, wav_sr):
        segs, _ = asr.transcribe(librosa.resample(wav.astype(np.float32), orig_sr=wav_sr, target_sr=16000),
                                 language="en", beam_size=5)
        return norm(" ".join(s.text.strip() for s in segs)) or "<empty>"

    refs, hyp = [], {"1_real": [], "2_vocoded": [], "3_tts": [], "4_tts_true_prosody": []}
    examples = []
    with torch.no_grad():
        for k, (uid, spk, text) in enumerate(items):
            refs.append(norm(text))
            real, rsr = librosa.load(os.path.join(raw, "wavs", uid + ".wav"), sr=None)
            hyp["1_real"].append(asr_text(real, rsr))
            mel = torch.from_numpy(np.load(os.path.join(pp, "mel", uid + ".npy"))).to(device)
            voc = vocoder(mel.T.unsqueeze(0)).squeeze().cpu().numpy()
            hyp["2_vocoded"].append(asr_text(np.clip(voc, -1, 1), sr))
            _, ids = tp(text)
            out = model(torch.tensor([meta["speakers"][spk]], device=device), torch.tensor([ids], device=device),
                        torch.tensor([len(ids)], device=device))
            wav = vocoder(out["postnet_mel"][0].float().T.unsqueeze(0)).squeeze().cpu().numpy()
            hyp["3_tts"].append(asr_text(np.clip(wav, -1, 1), sr))
            # (4) same model, but durations from the aligner and real pitch/energy (teacher-forced)
            pitch = torch.from_numpy(np.load(os.path.join(pp, "pitch", uid + ".npy"))).to(device)[None]
            energy = torch.from_numpy(np.load(os.path.join(pp, "energy", uid + ".npy"))).to(device)[None]
            prior = torch.from_numpy(beta_binomial_prior(len(ids), mel.shape[0])).to(device)[None]
            tf = model(torch.tensor([meta["speakers"][spk]], device=device), torch.tensor([ids], device=device),
                       torch.tensor([len(ids)], device=device), mel[None], torch.tensor([mel.shape[0]], device=device),
                       pitch, energy, prior)
            wav_tf = vocoder(tf["postnet_mel"][0].float().T.unsqueeze(0)).squeeze().cpu().numpy()
            hyp["4_tts_true_prosody"].append(asr_text(np.clip(wav_tf, -1, 1), sr))
            if k < 4:
                examples.append((refs[-1], hyp["1_real"][-1], hyp["2_vocoded"][-1], hyp["3_tts"][-1],
                                 hyp["4_tts_true_prosody"][-1]))
            print(f"  {k + 1}/{len(items)}", end="\r", flush=True)

    print(f"\n[baseline] checkpoint step {step}, {len(items)} held-out validation sentences, ASR={a.whisper}")
    for k, v in hyp.items():
        print(f"  {k:20s} WER = {jiwer.wer(refs, v):.1%}")
    print("\nexamples:")
    for e in examples:
        for lab, s in zip(["REF ", "real", "voc ", "tts ", "tts+true prosody"], e):
            print(f"  {lab}: {s}")
        print()


if __name__ == "__main__":
    main()
