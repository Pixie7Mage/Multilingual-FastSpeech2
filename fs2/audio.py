"""Audio feature extraction.

Replaces the reference repo's audio/stft.py, whose librosa.filters.mel call
uses a positional-argument API removed in librosa >= 0.10.  The mel
computation below is identical to the one HiFi-GAN was trained with
(reflect padding of (n_fft-hop)/2, center=False, log(clamp(x,1e-5))), so the
pretrained universal HiFi-GAN vocoder shipped with the reference repo can be
used directly.
"""
import numpy as np
import torch
import librosa
import pyworld as pw

_mel_basis = {}
_window = {}


def mel_spectrogram(wav, cfg):
    """wav: float32 numpy [-1,1] -> (log-mel [T,n_mels], energy [T])"""
    a = cfg["preprocessing"]
    n_fft, hop, win = a["stft"]["filter_length"], a["stft"]["hop_length"], a["stft"]["win_length"]
    sr, n_mels = a["audio"]["sampling_rate"], a["mel"]["n_mel_channels"]
    fmin, fmax = a["mel"]["mel_fmin"], a["mel"]["mel_fmax"]
    key = (sr, n_fft, n_mels, fmin, fmax)
    if key not in _mel_basis:
        _mel_basis[key] = torch.from_numpy(
            librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)).float()
        _window[win] = torch.hann_window(win)
    y = torch.from_numpy(wav).float().unsqueeze(0)
    pad = int((n_fft - hop) / 2)
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(y, n_fft, hop_length=hop, win_length=win, window=_window[win],
                      center=False, pad_mode="reflect", normalized=False, onesided=True,
                      return_complex=True)
    mag = torch.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-9)  # (1, F, T)
    mel = torch.matmul(_mel_basis[key], mag)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    energy = torch.norm(mag, dim=1)  # L2 norm of each STFT frame (as in reference)
    return mel[0].T.numpy().astype(np.float32), energy[0].numpy().astype(np.float32)


def extract_pitch(wav, n_frames, cfg):
    a = cfg["preprocessing"]
    sr, hop = a["audio"]["sampling_rate"], a["stft"]["hop_length"]
    fmin = a["pitch"].get("f0_min", 60.0)
    fmax = a["pitch"].get("f0_max", 600.0)
    x = wav.astype(np.float64)
    f0, t = pw.dio(x, sr, f0_floor=fmin, f0_ceil=fmax, frame_period=hop / sr * 1000)
    f0 = pw.stonemask(x, f0, t, sr)
    f0 = f0[:n_frames]
    if len(f0) < n_frames:
        f0 = np.pad(f0, (0, n_frames - len(f0)))
    return f0.astype(np.float32)


def interpolate_unvoiced(f0):
    """Linear interpolation over unvoiced (f0==0) frames, as in the reference."""
    f0 = f0.copy()
    voiced = np.where(f0 > 0)[0]
    if len(voiced) < 2:
        return None
    f0 = np.interp(np.arange(len(f0)), voiced, f0[voiced]).astype(np.float32)
    return f0


def load_wav(path, sr):
    wav, _ = librosa.load(path, sr=sr)
    return wav.astype(np.float32)


def trim_and_normalize(wav, sr, top_db=40, pad_ms=50):
    yt, idx = librosa.effects.trim(wav, top_db=top_db, frame_length=1024, hop_length=256)
    pad = int(sr * pad_ms / 1000)
    s, e = max(0, idx[0] - pad), min(len(wav), idx[1] + pad)
    wav = wav[s:e]
    peak = np.abs(wav).max() + 1e-8
    return (wav / peak * 0.95).astype(np.float32)
