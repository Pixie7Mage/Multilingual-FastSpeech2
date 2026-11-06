"""Dataset download + selection/subsampling (≈5 h train / ≈1 h validation).

Downloads the parquet shards straight from the Hugging Face Hub with
huggingface_hub (no `datasets` audio decoding, which is fragile across
versions), decodes the embedded audio bytes with soundfile/librosa, filters,
selects speakers/utterances, resamples to 22.05 kHz and writes

    <raw_dir>/wavs/<id>.wav
    <raw_dir>/train.tsv   id \t speaker \t duration \t text
    <raw_dir>/val.tsv
    <raw_dir>/selection_stats.json

Selection strategy (documented in the report):
  1. Filter: 1.0 s <= duration <= max_dur s, non-empty transcript, transcript
     only in the language's script (no stray Latin letters for Telugu), no
     bracketed noise tags.
  2. Speakers:
       English / Svarah  -> multi-speaker (117 speakers, ~5 min each); a
                            pseudo speaker-id is derived from the metadata
                            columns (Svarah on HF has no speaker_id column).
       Telugu / IndicTTS -> single speaker (the gender/voice with the most
                            audio); falls back to both voices if <6 h.
  3. Validation: random utterances (seeded) until ~1 h, drawn proportionally
     from every speaker.
  4. Training: random utterances from the remainder until ~5 h, excluding any
     utterance whose normalised transcript also appears in validation, so the
     splits are disjoint both by utterance and by sentence text.
"""
import argparse
import glob
import hashlib
import io
import json
import os
import random
import re
from collections import defaultdict

import numpy as np
import soundfile as sf
import librosa

DATASETS = {
    "svarah": {"repo": "ai4bharat/Svarah", "lang": "en"},
    "indictts_te": {"repo": "SPRINGLab/IndicTTS_Telugu", "lang": "te"},
}


def _download(repo, cache_dir, token=None):
    from huggingface_hub import snapshot_download
    for rev in [None, "refs/convert/parquet"]:
        try:
            path = snapshot_download(repo_id=repo, repo_type="dataset", revision=rev,
                                     allow_patterns=["*.parquet", "**/*.parquet"],
                                     cache_dir=cache_dir, token=token)
        except Exception as e:  # noqa
            print(f"[download] revision={rev} failed: {e}")
            continue
        files = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
        if files:
            print(f"[download] {len(files)} parquet file(s) from {repo} (revision={rev})")
            return files
    raise RuntimeError(
        f"Could not download parquet files for {repo}. If the dataset is gated (Svarah is), "
        f"accept its terms on huggingface.co and pass --hf_token or run `huggingface-cli login`.")


def _iter_rows(files, batch_size=64):
    import pyarrow.parquet as pq
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield row


def _audio_field(row):
    for k, v in row.items():
        if isinstance(v, dict) and ("bytes" in v or "array" in v):
            return k
    raise KeyError(f"No audio column found in row keys {list(row)}")


def _decode(audio):
    if audio.get("bytes"):
        try:
            wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
        except Exception:
            wav, sr = librosa.load(io.BytesIO(audio["bytes"]), sr=None)
    elif audio.get("array") is not None:
        wav, sr = np.asarray(audio["array"], dtype=np.float32), audio["sampling_rate"]
    else:
        raise ValueError("empty audio")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32), sr


def _clean_text(text, lang):
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    if lang == "en":
        if re.search(r"[\[\]<>{}#@]", t):
            return None
        if not re.search(r"[A-Za-z]", t):
            return None
        if re.search(r"[^A-Za-z0-9 .,?!'\-:;\"%&$₹()/]", t):
            return None
        t = t.replace("(", "").replace(")", "").replace("/", " ")
    else:
        if re.search(r"[A-Za-z\[\]<>{}#@]", t):
            return None
        if not re.search(r"[ఀ-౿]", t):
            return None
    return t if len(t.split()) >= 2 else None


def _norm_key(t):
    return re.sub(r"[^\w]", "", t.lower())


def collect(args):
    info = DATASETS[args.dataset]
    lang = info["lang"]
    files = _download(info["repo"], args.cache_dir, args.hf_token)
    tmp_dir = os.path.join(args.raw_dir, "_all_wavs")
    os.makedirs(tmp_dir, exist_ok=True)

    items = []
    n_seen = n_bad = 0
    meta_cols = ["gender", "age-group", "primary_language", "native_place_state", "native_place_district",
                 "highest_qualification", "job_category", "occupation_domain"]
    for row in _iter_rows(files):
        n_seen += 1
        if args.max_rows and n_seen > args.max_rows:
            break
        try:
            akey = _audio_field(row)
            audio = row[akey]
            text = _clean_text(row.get("text") or row.get("sentence") or row.get("transcription"), lang)
            if text is None:
                n_bad += 1
                continue
            wav, sr = _decode(audio)
            dur = len(wav) / sr
            if not (args.min_dur <= dur <= args.max_dur):
                n_bad += 1
                continue
            if lang == "en":
                spk_src = "|".join(str(row.get(c, "")) for c in meta_cols)
                speaker = "svarah_" + hashlib.md5(spk_src.encode()).hexdigest()[:8]
            else:
                g = row.get("gender")
                g = {0: "female", 1: "male"}.get(g, g) if isinstance(g, int) else (g or "unk")
                speaker = f"te_{g}"
            uid = f"{lang}_{len(items):06d}"
            if sr != args.sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=args.sr)
            sf.write(os.path.join(tmp_dir, uid + ".wav"), wav, args.sr, subtype="PCM_16")
            items.append({"id": uid, "speaker": speaker, "duration": round(dur, 3), "text": text,
                          "gender": str(row.get("gender", "")), "orig_sr": sr})
        except Exception as e:  # noqa
            n_bad += 1
            if n_bad < 10:
                print("[warn] skipping row:", e)
        if n_seen % 500 == 0:
            print(f"[collect] rows={n_seen} kept={len(items)} hours={sum(i['duration'] for i in items)/3600:.2f}")
    print(f"[collect] rows={n_seen} kept={len(items)} rejected={n_bad}")
    return items, tmp_dir, lang


def select(items, args, lang):
    rng = random.Random(args.seed)
    by_spk = defaultdict(list)
    for it in items:
        by_spk[it["speaker"]].append(it)
    hours = {s: sum(i["duration"] for i in v) / 3600 for s, v in by_spk.items()}
    need = args.train_hours + args.val_hours

    if lang == "te":
        best = max(hours, key=hours.get)
        speakers = [best] if hours[best] >= need * 1.02 else sorted(hours, key=hours.get, reverse=True)
        print(f"[select] Telugu voices (hours): {hours} -> using {speakers}")
    else:
        speakers = sorted(by_spk)
        print(f"[select] English: {len(speakers)} speakers, {sum(hours.values()):.2f} h available")

    pool = [it for s in speakers for it in by_spk[s]]
    total_h = sum(i["duration"] for i in pool) / 3600
    if total_h < need:
        print(f"[select][warn] only {total_h:.2f} h available (< {need} h); scaling splits proportionally")
        scale = total_h / need * 0.98
        tr_h, va_h = args.train_hours * scale, args.val_hours * scale
    else:
        tr_h, va_h = args.train_hours, args.val_hours

    # --- validation: proportional per speaker
    val, val_keys = [], set()
    spk_pool = {s: rng.sample(by_spk[s], len(by_spk[s])) for s in speakers}
    spk_h = {s: sum(i["duration"] for i in by_spk[s]) / 3600 for s in speakers}
    tot = sum(spk_h.values())
    for s in speakers:
        quota = va_h * spk_h[s] / tot * 3600
        acc = 0.0
        while spk_pool[s] and acc < quota:
            it = spk_pool[s].pop()
            val.append(it)
            val_keys.add(_norm_key(it["text"]))
            acc += it["duration"]
    # --- train: random from remainder, text-disjoint from val
    rest = [it for s in speakers for it in spk_pool[s] if _norm_key(it["text"]) not in val_keys]
    rng.shuffle(rest)
    train, acc = [], 0.0
    for it in rest:
        if acc >= tr_h * 3600:
            break
        train.append(it)
        acc += it["duration"]
    assert not ({i["id"] for i in train} & {i["id"] for i in val})
    return train, val, speakers, hours


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--train_hours", type=float, default=5.0)
    ap.add_argument("--val_hours", type=float, default=1.0)
    ap.add_argument("--min_dur", type=float, default=1.0)
    ap.add_argument("--max_dur", type=float, default=15.0)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_rows", type=int, default=0, help="debug: stop after N rows")
    args = ap.parse_args()

    items, tmp_dir, lang = collect(args)
    train, val, speakers, hours = select(items, args, lang)
    wav_dir = os.path.join(args.raw_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    keep = {i["id"] for i in train + val}
    for it in items:
        src = os.path.join(tmp_dir, it["id"] + ".wav")
        if it["id"] in keep:
            os.replace(src, os.path.join(wav_dir, it["id"] + ".wav"))
        elif os.path.exists(src):
            os.remove(src)
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass
    for name, split in [("train", train), ("val", val)]:
        with open(os.path.join(args.raw_dir, f"{name}.tsv"), "w", encoding="utf-8") as f:
            for it in split:
                f.write(f"{it['id']}\t{it['speaker']}\t{it['duration']}\t{it['text']}\n")
    stats = {
        "dataset": args.dataset, "seed": args.seed,
        "available_hours_per_speaker": {k: round(v, 3) for k, v in hours.items()},
        "speakers_used": len(speakers),
        "train": {"utterances": len(train), "hours": round(sum(i['duration'] for i in train) / 3600, 3),
                  "speakers": len({i['speaker'] for i in train})},
        "val": {"utterances": len(val), "hours": round(sum(i['duration'] for i in val) / 3600, 3),
                "speakers": len({i['speaker'] for i in val})},
        "filters": {"min_dur": args.min_dur, "max_dur": args.max_dur},
        "text_overlap_train_val": 0,
    }
    with open(os.path.join(args.raw_dir, "selection_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: stats[k] for k in ["train", "val", "speakers_used"]}, indent=2))


if __name__ == "__main__":
    main()
