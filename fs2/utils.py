import json
import os
import zipfile

import torch
import yaml

from .model.fastspeech2 import FastSpeech2

PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(path):
    """Loads a language config.yaml; relative paths are resolved against the yaml's folder."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = os.path.dirname(os.path.abspath(path))
    for k, v in cfg["path"].items():
        if isinstance(v, str) and not os.path.isabs(v):
            cfg["path"][k] = os.path.normpath(os.path.join(base, v))
    t = cfg.get("text", {})
    if t.get("lexicon_override") and not os.path.isabs(t["lexicon_override"]):
        t["lexicon_override"] = os.path.normpath(os.path.join(base, t["lexicon_override"]))
    return cfg


def get_device(prefer=None):
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(cfg, meta):
    return FastSpeech2(cfg["model"], n_symbols=len(meta["symbols"]), n_speakers=len(meta["speakers"]),
                       stats=meta["stats"], n_mels=cfg["preprocessing"]["mel"]["n_mel_channels"])


def load_meta_from_preprocessed(cfg):
    d = cfg["path"]["preprocessed_path"]
    meta = {}
    for k, fn in [("symbols", "symbols.json"), ("speakers", "speakers.json"), ("stats", "stats.json")]:
        with open(os.path.join(d, fn), encoding="utf-8") as f:
            meta[k] = json.load(f)
    # default inference speaker = speaker with the most training audio
    counts = {}
    tr = os.path.join(d, "train.txt")
    if os.path.exists(tr):
        with open(tr, encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("|")
                counts[p[1]] = counts.get(p[1], 0) + int(p[4])
    meta["default_speaker"] = max(counts, key=counts.get) if counts else next(iter(meta["speakers"]))
    return meta


def save_checkpoint(path, model, optim, step, cfg, meta, best_val=None):
    torch.save({"model": model.state_dict(), "optimizer": optim._optimizer.state_dict() if optim else None,
                "step": step, "config": cfg, "meta": meta, "best_val": best_val}, path)


def export_fp16(src_ckpt, dst):
    """Inference-only checkpoint (fp16 weights, no optimizer) — fits GitHub's 100 MB limit."""
    ck = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    sd = {k: (v.half() if v.is_floating_point() else v) for k, v in ck["model"].items()}
    torch.save({"model": sd, "step": ck["step"], "config": ck["config"], "meta": ck["meta"], "fp16": True}, dst)
    return dst


def load_for_inference(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg, meta = ck["config"], ck["meta"]
    model = build_model(cfg, meta)
    sd = {k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()}
    model.load_state_dict(sd)
    model.eval().to(device)
    return model, cfg, meta, ck.get("step")


def get_vocoder(device):
    from . import hifigan
    with open(os.path.join(PKG_DIR, "hifigan", "config.json")) as f:
        hcfg = hifigan.AttrDict(json.load(f))
    ckpt = os.path.join(PKG_DIR, "hifigan", "generator_universal.pth.tar")
    if not os.path.exists(ckpt):
        with zipfile.ZipFile(ckpt + ".zip") as z:
            z.extractall(os.path.join(PKG_DIR, "hifigan"))
    voc = hifigan.Generator(hcfg)
    voc.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["generator"])
    voc.eval()
    voc.remove_weight_norm()
    return voc.to(device)
