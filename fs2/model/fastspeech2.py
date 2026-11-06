"""FastSpeech 2 — adapted from ming024/FastSpeech2 (model/fastspeech2.py).

Encoder / Decoder / PostNet (transformer/) and the variance adaptor are the
reference components.  Added: an internal alignment module (aligner.py) that
provides phoneme durations during training instead of MFA.
"""
import torch
import torch.nn as nn

from ..transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor, get_mask_from_lengths
from .aligner import ConvAttention, binarize_attention


class FastSpeech2(nn.Module):
    def __init__(self, model_config, n_symbols, n_speakers, stats, n_mels=80):
        super().__init__()
        cfg = dict(model_config)
        cfg["n_symbols"] = n_symbols
        self.model_config = cfg
        self.encoder = Encoder(cfg)
        self.variance_adaptor = VarianceAdaptor(cfg, stats)
        self.decoder = Decoder(cfg)
        self.mel_linear = nn.Linear(cfg["transformer"]["decoder_hidden"], n_mels)
        self.postnet = PostNet(n_mel_channels=n_mels)
        d = cfg["transformer"]["encoder_hidden"]
        acfg = cfg.get("aligner", {})
        self.aligner = ConvAttention(n_mel_channels=n_mels, n_text_channels=d,
                                     n_att_channels=acfg.get("attn_channels", 80),
                                     temperature=acfg.get("temperature", 0.0005))
        # Separate text embedding for the aligner. When the aligner shares the encoder's
        # embedding, the mel/duration/pitch/energy losses keep moving its keys and the
        # duration loss feeds back into the alignment that produces its own targets,
        # which collapsed the alignment (most tokens 1 frame, one token ~30% of frames).
        self.aligner_emb = nn.Embedding(n_symbols, d, padding_idx=0) if acfg.get("own_embedding", False) else None
        self.speaker_emb = nn.Embedding(n_speakers, d) if n_speakers > 1 else None

    def align_text_emb(self, texts):
        return self.aligner_emb(texts) if self.aligner_emb is not None else self.encoder.src_word_emb(texts)

    def forward(self, speakers, texts, src_lens, mels=None, mel_lens=None, pitch=None, energy=None,
                attn_prior=None, p_control=1.0, e_control=1.0, d_control=1.0, durations=None):
        max_src_len = texts.shape[1]
        src_mask = get_mask_from_lengths(src_lens, max_src_len)
        enc = self.encoder(texts, src_mask)
        if self.speaker_emb is not None:
            enc = enc + self.speaker_emb(speakers).unsqueeze(1)

        out = {"src_mask": src_mask, "src_lens": src_lens}
        mel_mask = None
        max_mel_len = None
        if mels is not None:
            max_mel_len = mels.shape[1]
            mel_mask = get_mask_from_lengths(mel_lens, max_mel_len)
            emb = self.align_text_emb(texts)  # (B, T_text, d)
            attn_soft, attn_logprob = self.aligner(mels.transpose(1, 2), emb.transpose(1, 2),
                                                   src_mask, attn_prior)
            attn_hard = binarize_attention(attn_soft, src_lens, mel_lens)
            durations = attn_hard.sum(2)[:, 0, :].long()  # (B, T_text)
            out.update(attn_soft=attn_soft, attn_logprob=attn_logprob, attn_hard=attn_hard)

        (x, p_pred, e_pred, log_d_pred, d_used, mel_len, mel_mask, p_tgt, e_tgt) = self.variance_adaptor(
            enc, src_mask, mel_mask, max_mel_len, durations, pitch, energy, p_control, e_control, d_control)

        x, mel_mask_dec = self.decoder(x, mel_mask)
        mel = self.mel_linear(x)
        post = self.postnet(mel) + mel
        out.update(mel=mel, postnet_mel=post, pitch_pred=p_pred, energy_pred=e_pred,
                   log_d_pred=log_d_pred, durations=d_used, mel_lens=mel_len, mel_mask=mel_mask_dec,
                   pitch_target=p_tgt, energy_target=e_tgt)
        return out
