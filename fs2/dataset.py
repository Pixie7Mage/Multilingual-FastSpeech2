import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .model.aligner import beta_binomial_prior


class TTSDataset(Dataset):
    def __init__(self, split, cfg):
        self.dir = cfg["path"]["preprocessed_path"]
        with open(os.path.join(self.dir, "symbols.json"), encoding="utf-8") as f:
            self.sym2id = {s: i for i, s in enumerate(json.load(f))}
        with open(os.path.join(self.dir, "speakers.json")) as f:
            self.spk2id = json.load(f)
        self.prior_scaling = cfg["train"].get("aligner", {}).get("prior_scaling", 1.0)
        self.items = []
        with open(os.path.join(self.dir, f"{split}.txt"), encoding="utf-8") as f:
            for line in f:
                uid, spk, phones, text, n = line.rstrip("\n").split("|")
                self.items.append((uid, spk, phones.split(" "), text, int(n)))

    def __len__(self):
        return len(self.items)

    def lengths(self):
        return [it[4] for it in self.items]

    def __getitem__(self, i):
        uid, spk, phones, text, n = self.items[i]
        mel = np.load(os.path.join(self.dir, "mel", uid + ".npy"))
        pitch = np.load(os.path.join(self.dir, "pitch", uid + ".npy"))
        energy = np.load(os.path.join(self.dir, "energy", uid + ".npy"))
        ids = np.array([self.sym2id[p] for p in phones], dtype=np.int64)
        return {"id": uid, "speaker": self.spk2id[spk], "text": ids, "mel": mel, "pitch": pitch,
                "energy": energy, "prior": beta_binomial_prior(len(ids), mel.shape[0], self.prior_scaling)}


def collate(batch):
    B = len(batch)
    tl = [len(b["text"]) for b in batch]
    ml = [b["mel"].shape[0] for b in batch]
    Tt, Tm = max(tl), max(ml)
    texts = torch.zeros(B, Tt, dtype=torch.long)
    mels = torch.zeros(B, Tm, batch[0]["mel"].shape[1])
    pitch = torch.zeros(B, Tm)
    energy = torch.zeros(B, Tm)
    prior = torch.zeros(B, Tm, Tt)
    for i, b in enumerate(batch):
        texts[i, :tl[i]] = torch.from_numpy(b["text"])
        mels[i, :ml[i]] = torch.from_numpy(b["mel"])
        pitch[i, :ml[i]] = torch.from_numpy(b["pitch"])
        energy[i, :ml[i]] = torch.from_numpy(b["energy"])
        prior[i, :ml[i], :tl[i]] = torch.from_numpy(b["prior"])
    return {"ids": [b["id"] for b in batch], "speakers": torch.tensor([b["speaker"] for b in batch]),
            "texts": texts, "src_lens": torch.tensor(tl), "mels": mels, "mel_lens": torch.tensor(ml),
            "pitch": pitch, "energy": energy, "attn_prior": prior}


class BucketSampler(Sampler):
    """Groups utterances of similar length into batches (less padding)."""

    def __init__(self, lengths, batch_size, bucket_mult=8, shuffle=True, seed=0):
        self.lengths, self.bs, self.mult, self.shuffle = lengths, batch_size, bucket_mult, shuffle
        self.epoch, self.seed = 0, seed

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(idx)
        chunk = self.bs * self.mult
        batches = []
        for s in range(0, len(idx), chunk):
            c = sorted(idx[s:s + chunk], key=lambda i: self.lengths[i])
            batches += [c[k:k + self.bs] for k in range(0, len(c), self.bs)]
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return (len(self.lengths) + self.bs - 1) // self.bs
