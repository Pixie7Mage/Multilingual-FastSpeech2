"""Unsupervised text-to-mel alignment (new module, not in the reference repo).

The reference FastSpeech 2 implementation needs phoneme durations from the
Montreal Forced Aligner (MFA).  MFA has no pretrained Telugu acoustic model or
lexicon, and for Svarah (117 Indian-English speakers) MFA's US-English model
gives noisy alignments.  We therefore replace the external aligner with the
alignment-learning framework of Badlani et al. 2022, "One TTS Alignment To Rule
Them All" (also used in NVIDIA FastPitch / RAD-TTS):

  * ConvAttention: soft alignment between text embeddings and mel frames;
  * beta-binomial alignment prior to encourage near-diagonal alignments;
  * forward-sum (CTC) loss that maximises the likelihood of all monotonic
    alignments;
  * Monotonic Alignment Search (MAS / Viterbi) to binarise the soft alignment
    into hard durations, which are used as duration targets for the
    duration predictor and for the length regulator during training;
  * binarisation (KL) loss that pulls the soft alignment towards the hard one.

Everything else in FastSpeech 2 (encoder, variance adaptor, decoder, postnet)
is unchanged.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import betabinom

try:
    from numba import jit
except ImportError:  # pragma: no cover - fallback if numba unavailable
    def jit(*a, **k):
        def deco(f):
            return f
        return deco


# ----------------------------------------------------------------------------
# Monotonic alignment search
# ----------------------------------------------------------------------------
@jit(nopython=True)
def _mas_width1(log_attn):
    """log_attn: (T_mel, T_text) -> hard alignment (T_mel, T_text) one-hot rows."""
    t_mel, t_text = log_attn.shape
    neg_inf = -1e30
    log_p = np.full((t_mel, t_text), neg_inf, dtype=np.float32)
    prev_ind = np.zeros((t_mel, t_text), dtype=np.int64)
    log_p[0, 0] = log_attn[0, 0]
    for i in range(1, t_mel):
        for j in range(t_text):
            prev_log = log_p[i - 1, j]
            prev_j = j
            if j - 1 >= 0 and log_p[i - 1, j - 1] >= prev_log:
                prev_log = log_p[i - 1, j - 1]
                prev_j = j - 1
            log_p[i, j] = log_attn[i, j] + prev_log
            prev_ind[i, j] = prev_j
    opt = np.zeros((t_mel, t_text), dtype=np.float32)
    j = t_text - 1
    for i in range(t_mel - 1, -1, -1):
        opt[i, j] = 1.0
        j = prev_ind[i, j]
    return opt


def binarize_attention(attn_soft, text_lens, mel_lens):
    """attn_soft: (B,1,T_mel,T_text) -> hard (B,1,T_mel,T_text)"""
    with torch.no_grad():
        log_attn = torch.log(attn_soft.data.clamp(min=1e-8)).cpu().numpy()
        out = np.zeros_like(log_attn)
        for b in range(log_attn.shape[0]):
            tm, tt = int(mel_lens[b]), int(text_lens[b])
            out[b, 0, :tm, :tt] = _mas_width1(np.ascontiguousarray(log_attn[b, 0, :tm, :tt]))
    return torch.from_numpy(out).to(attn_soft.device)


# ----------------------------------------------------------------------------
# Beta-binomial prior
# ----------------------------------------------------------------------------
_prior_cache = {}


def beta_binomial_prior(text_len, mel_len, scaling=1.0):
    key = (text_len, mel_len)
    if key in _prior_cache:
        return _prior_cache[key]
    x = np.arange(0, text_len)[None, :]
    i = np.arange(1, mel_len + 1)[:, None]
    a, b = scaling * i, scaling * (mel_len + 1 - i)
    # one vectorised call (fast enough that a large cache is unnecessary; the old
    # 20000-entry cache was duplicated in every DataLoader worker and kept growing)
    prior = betabinom(text_len - 1, a, b).pmf(x).astype(np.float32)  # (T_mel, T_text)
    if len(_prior_cache) < 500:
        _prior_cache[key] = prior
    return prior


# ----------------------------------------------------------------------------
# Attention module
# ----------------------------------------------------------------------------
class ConvAttention(nn.Module):
    def __init__(self, n_mel_channels=80, n_text_channels=256, n_att_channels=80, temperature=0.0005):
        super().__init__()
        self.temperature = temperature
        self.key_proj = nn.Sequential(
            nn.Conv1d(n_text_channels, n_text_channels * 2, 3, padding=1), nn.ReLU(),
            nn.Conv1d(n_text_channels * 2, n_att_channels, 1))
        self.query_proj = nn.Sequential(
            nn.Conv1d(n_mel_channels, n_mel_channels * 2, 3, padding=1), nn.ReLU(),
            nn.Conv1d(n_mel_channels * 2, n_mel_channels, 1), nn.ReLU(),
            nn.Conv1d(n_mel_channels, n_att_channels, 1))

    def forward(self, queries, keys, mask=None, attn_prior=None):
        """queries: (B, n_mel, T_mel); keys: (B, C_text, T_text); mask: (B, T_text) True=pad
        returns attn_soft (B,1,T_mel,T_text), attn_logprob (B,1,T_mel,T_text)"""
        keys_enc = self.key_proj(keys)
        queries_enc = self.query_proj(queries)
        # squared L2 distance via |q|^2 + |k|^2 - 2 q.k  (avoids a B x C x T_mel x T_text tensor)
        q2 = (queries_enc ** 2).sum(1)[:, :, None]           # (B, T_mel, 1)
        k2 = (keys_enc ** 2).sum(1)[:, None, :]              # (B, 1, T_text)
        qk = torch.bmm(queries_enc.transpose(1, 2), keys_enc)  # (B, T_mel, T_text)
        attn = -self.temperature * (q2 + k2 - 2 * qk).clamp(min=0)
        attn = attn[:, None]                                 # (B,1,T_mel,T_text)
        if attn_prior is not None:
            attn = F.log_softmax(attn, dim=3) + torch.log(attn_prior[:, None] + 1e-8)
        attn_logprob = attn.clone()
        if mask is not None:
            attn = attn.masked_fill(mask[:, None, None, :], -float("inf"))
        attn = F.softmax(attn, dim=3)
        return attn, attn_logprob


class ForwardSumLoss(nn.Module):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=3)
        self.ctc = nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, text_lens, mel_lens):
        key_lens, query_lens = text_lens, mel_lens
        attn_logprob = F.pad(attn_logprob, (1, 0), value=self.blank_logprob)
        total = 0.0
        for b in range(attn_logprob.shape[0]):
            tgt = torch.arange(1, int(key_lens[b]) + 1, device=attn_logprob.device).unsqueeze(0)
            cur = attn_logprob[b].permute(1, 0, 2)  # (T_mel,1,T_text+1)
            cur = cur[: int(query_lens[b]), :, : int(key_lens[b]) + 1]
            cur = self.log_softmax(cur[None])[0]
            total = total + self.ctc(cur, tgt, input_lengths=query_lens[b:b + 1], target_lengths=key_lens[b:b + 1])
        return total / attn_logprob.shape[0]


class BinLoss(nn.Module):
    def forward(self, hard, soft):
        log_sum = torch.log(torch.clamp(soft[hard == 1], min=1e-12)).sum()
        return -log_sum / hard.sum()
