"""Inference: text -> phonemes -> FastSpeech 2 -> mel -> HiFi-GAN -> wav,
plus variance-adaptor plots (duration / pitch / energy) for each sentence."""
import json
import os

import numpy as np
import soundfile as sf
import torch

from .audio import extract_pitch, mel_spectrogram
from .plots import plot_variance
from .text import TextProcessor
from .utils import get_vocoder, load_for_inference, get_device


def read_sentences(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            sid, text = line.split("\t", 1)
            out.append((sid.strip(), text.strip()))
    return out


@torch.no_grad()
def synthesize(ckpt, sentences, out_root, device=None, speaker=None, p_control=1.0, e_control=1.0,
               d_control=1.0, lexicon_override=None):
    device = get_device(device)
    model, cfg, meta, step = load_for_inference(ckpt, device)
    text_cfg = dict(cfg.get("text", {}))
    if lexicon_override is not None:
        text_cfg["lexicon_override"] = lexicon_override
    tp = TextProcessor(cfg["language"], **text_cfg)
    assert tp.symbols == meta["symbols"], "symbol table mismatch between checkpoint and front-end"
    vocoder = get_vocoder(device)
    spk_name = speaker or meta.get("default_speaker") or next(iter(meta["speakers"]))
    spk_id = meta["speakers"][spk_name]
    p_mean, p_std = meta["stats"]["pitch"][2:4]
    e_mean, e_std = meta["stats"]["energy"][2:4]
    sr = cfg["preprocessing"]["audio"]["sampling_rate"]
    hop = cfg["preprocessing"]["stft"]["hop_length"]
    hop_s = hop / sr

    audio_dir = os.path.join(out_root, "test_generated_audios")
    plot_dir = os.path.join(out_root, "variance_adaptor_plots")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    summary = {"checkpoint": os.path.basename(ckpt), "step": step, "speaker": spk_name,
               "controls": {"pitch": p_control, "energy": e_control, "duration": d_control}, "sentences": {}}

    for sid, text in sentences:
        phones, ids = tp(text)
        texts = torch.tensor([ids], device=device)
        out = model(torch.tensor([spk_id], device=device), texts, torch.tensor([len(ids)], device=device),
                    p_control=p_control, e_control=e_control, d_control=d_control)
        mel = out["postnet_mel"][0].float()                  # (T, 80)
        wav = vocoder(mel.T.unsqueeze(0)).squeeze().cpu().numpy()
        wav = np.clip(wav, -1, 1)
        sf.write(os.path.join(audio_dir, f"{sid}.wav"), wav, sr, subtype="PCM_16")

        durs = out["durations"][0].long().cpu().numpy()
        pitch_hz = out["pitch_pred"][0].float().cpu().numpy() * p_std + p_mean
        energy = out["energy_pred"][0].float().cpu().numpy() * e_std + e_mean
        mel_np = mel.cpu().numpy()
        # measure what the vocoder actually produced (for analysis)
        real_f0 = extract_pitch(wav.astype(np.float32), mel_np.shape[0], cfg)
        _, real_en = mel_spectrogram(wav.astype(np.float32), cfg)
        real_en = real_en[:mel_np.shape[0]]
        plot_variance(sid, text, phones, durs, pitch_hz, energy, mel_np, real_f0, real_en, hop_s, plot_dir)

        voiced = real_f0[real_f0 > 0]
        summary["sentences"][sid] = {
            "text": text, "phonemes": " ".join(phones), "n_phonemes": len(phones),
            "duration_s": round(len(wav) / sr, 3),
            "per_phoneme": [{"phone": p, "frames": int(d), "ms": round(d * hop_s * 1000, 1),
                             "pitch_hz": round(float(ph), 1), "energy": round(float(en), 3)}
                            for p, d, ph, en in zip(phones, durs, pitch_hz, energy)],
            "pred_pitch_hz_mean": round(float(np.mean(pitch_hz)), 1),
            "pred_pitch_hz_range": [round(float(pitch_hz.min()), 1), round(float(pitch_hz.max()), 1)],
            "measured_f0_hz_mean": round(float(voiced.mean()), 1) if len(voiced) else None,
            "speech_rate_phones_per_s": round(sum(1 for p in phones if p not in "_,.?!") / (len(wav) / sr), 2),
        }
        print(f"[{sid}] {len(wav)/sr:.2f}s  {' '.join(phones)}")

    with open(os.path.join(out_root, "inference_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary
