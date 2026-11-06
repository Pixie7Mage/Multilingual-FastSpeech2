"""Variance adaptor (duration / pitch / energy) — adapted from ming024/FastSpeech2.

Modifications w.r.t. the reference model/modules.py:
  * Phoneme-level pitch/energy targets are computed on the fly by averaging the
    frame-level contours over the (MAS) durations, since durations now come
    from the internal aligner instead of MFA TextGrids at preprocessing time.
  * Vectorised, device-agnostic length regulator (torch.repeat_interleave).
  * Fixed a reference bug where the energy predictor used the *pitch* control.
  * Predicted durations are clamped to >= 1 frame at inference (MAS guarantees
    every token >= 1 frame during training, so a 0-frame token would delete
    a phoneme).
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = int(torch.max(lengths).item())
    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(lengths.shape[0], -1)
    return ids >= lengths.unsqueeze(1)  # True = padding


def average_by_duration(values, durs):
    """values: (B, T_mel) frame-level; durs: (B, T_text) int -> (B, T_text) mean per token."""
    cs = torch.nn.functional.pad(torch.cumsum(values, dim=1), (1, 0))
    end = torch.cumsum(durs, dim=1).long()
    start = end - durs.long()
    end = end.clamp(max=cs.shape[1] - 1)
    start = start.clamp(max=cs.shape[1] - 1)
    s = torch.gather(cs, 1, end) - torch.gather(cs, 1, start)
    return s / durs.float().clamp(min=1)


class LengthRegulator(nn.Module):
    def forward(self, x, durations, max_len=None):
        out, lens = [], []
        for b in range(x.shape[0]):
            e = torch.repeat_interleave(x[b], durations[b].long().clamp(min=0), dim=0)
            out.append(e)
            lens.append(e.shape[0])
        T = max_len if max_len is not None else max(lens)
        T = max(T, 1)
        padded = x.new_zeros(x.shape[0], T, x.shape[2])
        for b, e in enumerate(out):
            n = min(e.shape[0], T)
            padded[b, :n] = e[:n]
        return padded, torch.tensor(lens, device=x.device, dtype=torch.long)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy predictor (2x Conv1D-ReLU-LN-Dropout + Linear)."""

    def __init__(self, model_config):
        super().__init__()
        d_in = model_config["transformer"]["encoder_hidden"]
        f = model_config["variance_predictor"]["filter_size"]
        k = model_config["variance_predictor"]["kernel_size"]
        p = model_config["variance_predictor"]["dropout"]
        self.conv_layer = nn.Sequential(OrderedDict([
            ("conv1d_1", Conv(d_in, f, k, (k - 1) // 2)), ("relu_1", nn.ReLU()),
            ("layer_norm_1", nn.LayerNorm(f)), ("dropout_1", nn.Dropout(p)),
            ("conv1d_2", Conv(f, f, k, (k - 1) // 2)), ("relu_2", nn.ReLU()),
            ("layer_norm_2", nn.LayerNorm(f)), ("dropout_2", nn.Dropout(p)),
        ]))
        self.linear_layer = nn.Linear(f, 1)

    def forward(self, x, mask):
        out = self.linear_layer(self.conv_layer(x)).squeeze(-1)
        if mask is not None:
            out = out.masked_fill(mask, 0.0)
        return out


class VarianceAdaptor(nn.Module):
    def __init__(self, model_config, stats):
        super().__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()

        n_bins = model_config["variance_embedding"]["n_bins"]
        pmin, pmax = stats["pitch"][:2]
        emin, emax = stats["energy"][:2]
        self.pitch_bins = nn.Parameter(torch.linspace(pmin, pmax, n_bins - 1), requires_grad=False)
        self.energy_bins = nn.Parameter(torch.linspace(emin, emax, n_bins - 1), requires_grad=False)
        d = model_config["transformer"]["encoder_hidden"]
        self.pitch_embedding = nn.Embedding(n_bins, d)
        self.energy_embedding = nn.Embedding(n_bins, d)

    def forward(self, x, src_mask, mel_mask=None, max_len=None, durations=None,
                pitch_frame=None, energy_frame=None, p_control=1.0, e_control=1.0, d_control=1.0):
        log_d_pred = self.duration_predictor(x, src_mask)

        pitch_target = energy_target = None
        if durations is not None and pitch_frame is not None:
            pitch_target = average_by_duration(pitch_frame, durations).masked_fill(src_mask, 0.0)
            energy_target = average_by_duration(energy_frame, durations).masked_fill(src_mask, 0.0)

        # pitch (phoneme level)
        p_pred = self.pitch_predictor(x, src_mask)
        if pitch_target is not None:
            p_emb = self.pitch_embedding(torch.bucketize(pitch_target, self.pitch_bins))
        else:
            p_pred = p_pred * p_control
            p_emb = self.pitch_embedding(torch.bucketize(p_pred, self.pitch_bins))
        x = x + p_emb

        # energy (phoneme level)
        e_pred = self.energy_predictor(x, src_mask)
        if energy_target is not None:
            e_emb = self.energy_embedding(torch.bucketize(energy_target, self.energy_bins))
        else:
            e_pred = e_pred * e_control
            e_emb = self.energy_embedding(torch.bucketize(e_pred, self.energy_bins))
        x = x + e_emb

        if durations is not None:
            d_used = durations
            x, mel_len = self.length_regulator(x, durations, max_len)
        else:
            d_used = torch.clamp(torch.round((torch.exp(log_d_pred) - 1) * d_control), min=1)
            d_used = d_used.masked_fill(src_mask, 0)
            x, mel_len = self.length_regulator(x, d_used, None)
            mel_mask = get_mask_from_lengths(mel_len)

        return x, p_pred, e_pred, log_d_pred, d_used, mel_len, mel_mask, pitch_target, energy_target
