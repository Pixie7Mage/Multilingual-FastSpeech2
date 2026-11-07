"""FastSpeech 2 loss — reference loss (mel L1 + postnet L1 + MSE on log-duration,
pitch, energy) plus the alignment losses (forward-sum CTC + binarisation)."""
import torch
import torch.nn as nn

from .aligner import ForwardSumLoss, BinLoss


class FastSpeech2Loss(nn.Module):
    def __init__(self, train_config):
        super().__init__()
        a = train_config.get("aligner", {})
        self.ctc_w = a.get("ctc_weight", 1.0)
        self.bin_w = a.get("bin_weight", 1.0)
        self.bin_start = a.get("bin_start_step", 5000)
        self.bin_warmup = a.get("bin_warmup_steps", 5000)
        self.mse, self.mae = nn.MSELoss(), nn.L1Loss()
        self.ctc, self.bin = ForwardSumLoss(), BinLoss()

    def forward(self, batch, out, step):
        src_valid = ~out["src_mask"]
        mel_valid = ~out["mel_mask"]
        T = mel_valid.shape[1]
        mel_t = batch["mels"][:, :T]
        mel_valid3 = mel_valid.unsqueeze(-1)

        mel_loss = self.mae(out["mel"].masked_select(mel_valid3), mel_t.masked_select(mel_valid3))
        post_loss = self.mae(out["postnet_mel"].masked_select(mel_valid3), mel_t.masked_select(mel_valid3))

        log_d_t = torch.log(out["durations"].float() + 1).detach()
        dur_loss = self.mse(out["log_d_pred"].masked_select(src_valid), log_d_t.masked_select(src_valid))
        pitch_loss = self.mse(out["pitch_pred"].masked_select(src_valid), out["pitch_target"].detach().masked_select(src_valid))
        energy_loss = self.mse(out["energy_pred"].masked_select(src_valid), out["energy_target"].detach().masked_select(src_valid))

        ctc_loss = self.ctc(out["attn_logprob"], batch["src_lens"], batch["mel_lens"])
        if step >= self.bin_start:
            w = min(1.0, (step - self.bin_start + 1) / max(1, self.bin_warmup))
            bin_loss = self.bin(out["attn_hard"], out["attn_soft"]) * w
        else:
            bin_loss = torch.zeros((), device=mel_loss.device)

        total = mel_loss + post_loss + dur_loss + pitch_loss + energy_loss + self.ctc_w * ctc_loss + self.bin_w * bin_loss
        return total, {
            "total": total.item(), "mel": mel_loss.item(), "postnet_mel": post_loss.item(),
            "pitch": pitch_loss.item(), "energy": energy_loss.item(), "duration": dur_loss.item(),
            "align_ctc": ctc_loss.item(), "align_bin": bin_loss.item(),
        }
