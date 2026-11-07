# FastSpeech 2 Text-to-Speech for English and Telugu — Technical Report

**Author:** {{AUTHOR}}  **Repository:** {{REPO_URL}}

> Text in `[[ ... ]]` brackets must be replaced with your own observations after listening to the audio and looking at the plots. Everything else is filled automatically by `python Report/build_report.py`.

## 1. Implementation approach

The starting point is the public PyTorch implementation of FastSpeech 2 by ming024 (github.com/ming024/FastSpeech2), which follows Ren et al., *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech* (2021). Two models with identical architecture were trained separately: one on English (Svarah) and one on Telugu (IndicTTS Telugu).

### 1.1 Components used from the reference implementation

- **Encoder / decoder**: 4 + 6 Feed-Forward Transformer blocks (hidden 256, 2 heads, conv filter 1024, kernels [9, 1]) — `fs2/transformer/` (copied from the reference).
- **Variance adaptor**: duration, pitch and energy predictors (2 × Conv1D-ReLU-LayerNorm-Dropout + Linear), length regulator, and 256-bin quantised pitch/energy embeddings — `fs2/model/modules.py`.
- **Mel-postnet** (5 conv layers) and the reference loss: L1 on mel before/after postnet + MSE on log-duration, pitch and energy.
- **Noam-style LR schedule** (`ScheduledOptim`) with warm-up and annealing — `fs2/model/optimizer.py`.
- **Pretrained HiFi-GAN (universal)** vocoder shipped with the reference repo — `fs2/hifigan/`.

### 1.2 Modifications (both languages)

| Area | Reference | This work | Why |
|---|---|---|---|
| Duration targets | Montreal Forced Aligner TextGrids | Internal unsupervised aligner (Badlani et al., 2022): conv attention + beta-binomial prior + forward-sum (CTC) loss + Monotonic Alignment Search + binarisation loss (`fs2/model/aligner.py`) | MFA has no Telugu acoustic model/lexicon; its US-English model fits Indian-accented Svarah poorly |
| Pitch / energy | Phoneme averages computed in preprocessing from MFA durations | Frame-level features stored; phoneme averages computed on the fly from the MAS durations | No external alignments needed |
| Mel extraction | `audio/stft.py` (librosa API broken in librosa ≥ 0.10) | HiFi-GAN-identical log-mel (`fs2/audio.py`) | Compatibility + exact vocoder match |
| Length regulator | Python loop, CUDA-only `device` global | Vectorised `repeat_interleave`, device-agnostic | Speed, runs on CPU/MPS too |
| Energy control | Bug: energy used the pitch control factor | Fixed | Correctness |
| Inference durations | clamp ≥ 0 (phonemes can vanish) | clamp ≥ 1 frame | MAS trains every token with ≥ 1 frame |
| Validation | one batch | entire 1-hour validation set, logged to CSV | Reliable loss curves |
| Checkpoints | full checkpoints only | auto-resume, best-val checkpoint, fp16 export < 100 MB | GitHub limit, Colab disconnects |

### 1.3 English-specific changes

- Front-end `fs2/text/english.py`: normalisation of clock times ("10:45 AM" → "ten forty five A M"), dates ("12 November 2026" → "the twelfth of November twenty twenty six"), ordinals, currency, abbreviations; then **g2p_en** (CMUdict + POS-tagger homograph disambiguation + neural G2P for OOV words) producing 70 stressed ARPAbet phonemes, plus `_` word-boundary and punctuation tokens.
- Multi-speaker conditioning: Svarah has 117 speakers with only ~5 minutes each, so a speaker embedding (reference `multi_speaker` option) is added to the encoder output. The HF release has no speaker-id column, so a pseudo speaker-id is derived from the speaker metadata (gender, age group, native language, state, district, education, job). Inference uses the speaker with the most training audio ({{EN_SPEAKER}}).
- Optional lexicon override (`English/lexicon_override.txt`) for the code-mixed Telugu word *andaroo* (E4).

### 1.4 Telugu-specific changes

- New front-end `fs2/text/telugu.py` — rule-based Telugu G2P: inherent /a/ after consonants, deletion by virama (్) for conjuncts and geminates (పట్టు → p a tt tt u), vowel signs → short/long vowels, anusvara (ం) → homorganic nasal (బలంగా → b a l a **ng** g aa, ప్రారంభ → … **m** bh …), visarga → /h/; 58-symbol inventory.
- Telugu number normalisation: cardinals, years (2026 → రెండు వేల ఇరవై ఆరు), ordinal suffix వ (12వ → పన్నెండవ), clock times (10:45 → పది నలభై ఐదు).
- Single-speaker model (IndicTTS voice with the most data: {{TE_SPEAKER}}).

## 2. Datasets, preprocessing and training configuration

### 2.1 Selection and subsampling (`fs2/prepare_data.py`)

1. Parquet shards are downloaded from the Hugging Face Hub (`ai4bharat/Svarah`, `SPRINGLab/IndicTTS_Telugu`) and decoded with soundfile.
2. Filtering: 1–15 s duration, ≥ 2 words, transcript in the expected script only (no Latin characters for Telugu; no bracketed noise tags for English).
3. Validation (~1 h): seeded random sampling, proportional to each speaker's share of the data.
4. Training (~5 h): seeded random sampling from the remaining utterances, **excluding any utterance whose normalised transcript appears in validation** — the splits are disjoint by utterance and by sentence (overlap = 0, asserted in code).

| | English (Svarah) | Telugu (IndicTTS) |
|---|---|---|
| Train utterances / hours | {{EN_TRAIN_N}} / {{EN_TRAIN_H}} h | {{TE_TRAIN_N}} / {{TE_TRAIN_H}} h |
| Validation utterances / hours | {{EN_VAL_N}} / {{EN_VAL_H}} h | {{TE_VAL_N}} / {{TE_VAL_H}} h |
| Speakers (train) | {{EN_SPK}} | {{TE_SPK}} |
| Utterances dropped in preprocessing | {{EN_SKIP}} | {{TE_SKIP}} |
| Training-set mean F0 ± std | {{EN_F0}} Hz | {{TE_F0}} Hz |

### 2.2 Preprocessing (`fs2/preprocess_core.py`)

Resample to 22 050 Hz → trim leading/trailing silence (40 dB, 50 ms margin) → peak-normalise → 80-bin log-mel (FFT 1024, hop 256, window 1024, 0–8 kHz) → frame energy (L2 norm of each STFT frame) → F0 with WORLD DIO + StoneMask, unvoiced frames linearly interpolated → F0 and energy z-normalised with training-set statistics. Text → normalised text → phoneme IDs. Utterances longer than 1400 frames or with fewer frames than phonemes are dropped.

### 2.3 Training configuration

Batch 16, Adam (β = 0.9/0.98, ε = 1e-9), Noam LR with 4 000 warm-up steps, ×0.3 annealing at 30k/40k/45k steps, gradient clipping 1.0, binarisation loss ramped in from step 4 000. English trained for {{EN_STEPS}} steps, Telugu for {{TE_STEPS}} steps on {{GPU}}. Model size ≈ 35 M parameters (reference sizes). Full settings: `English/config.yaml`, `Telugu/config.yaml`.

## 3. Training and validation loss

![English — training loss]({{RES}}/English/training_loss.png)
![English — validation loss]({{RES}}/English/validation_loss.png)
![Telugu — training loss]({{RES}}/Telugu/training_loss.png)
![Telugu — validation loss]({{RES}}/Telugu/validation_loss.png)

Final validation losses (mel / postnet / duration / pitch / energy): English {{EN_VAL_FINAL}}; Telugu {{TE_VAL_FINAL}}. Best validation step: English {{EN_VAL_BEST}}, Telugu {{TE_VAL_BEST}}.

[[Describe convergence: how fast the mel loss drops, when the alignment (align_ctc) stabilises, the bump in total loss when the binarisation loss is ramped in at 4k steps, the effect of LR annealing at 30k. Compare train vs validation curves: is there a growing gap (overfitting) — typically visible in the pitch/energy losses on 5 h of data — and for which language. Note any spikes/instability.]]

## 4. Generated speech — observations per test sentence

Audio: `Results/English/test_generated_audios/`, `Results/Telugu/test_generated_audios/`.

| ID | Duration (s) | Phonemes | Observation |
|---|---|---|---|
{{SENTENCE_TABLE}}

[[For each sentence comment on intelligibility, pronunciation and naturalness. Things to listen for: E1 — the 28-phoneme word "antidisestablishmentarianism" (OOV, neural G2P) and whether it is rushed or slurred; E2a/E2b — heteronym "conduct": noun KAA1NDAH0KT (stress on 1st syllable) vs verb KAA0NDAH1KT (2nd syllable), g2p_en resolves this from POS tags, check that the audio shows the stress shift; E3 — the split token "rhyth m" is read literally as "rhyth" + letter "M" (preserved text), check the pause; E4 — number/time/date expansion and the code-mixed word "andaroo". T1 — long technical loan compound న్యూరోఫార్మకోలాజికల్; T2a/T2b — పట్టు (grip) vs పట్టు (silk) are spelled and pronounced identically, so differences can only come from context/prosody; T3 — split word "అడ విలో" read with a word break; T4 — digit expansion (2026, 12వ, 10:45) and the comma pause.]]

## 5. Variance-adaptor analysis (duration, pitch F0, energy)

Plots for every sentence: `Results/<language>/variance_adaptor_plots/{duration,pitch,energy}/<ID>.png`. Pitch and energy plots show the phoneme-level prediction of the variance adaptor (orange) on top of the predicted mel-spectrogram, together with F0/energy measured on the synthesised waveform (blue), which shows how faithfully the decoder + vocoder realised the predicted prosody.

{{VARIANCE_FIGS}}

[[Per sentence, explain the patterns: e.g. longer durations for stressed vowels and sentence-final phonemes (phrase-final lengthening), long pauses predicted at commas/full stops, F0 declination across the sentence with a final fall for declaratives, F0 peaks on stressed syllables (E2a vs E2b "conduct"), lower energy and short durations for unstressed function words and word-boundary tokens, energy dips at stops/fricatives. For Telugu: geminate consonants (tt tt, c c) getting long total closure, long vowels (aa, ii, ee, oo) predicted longer than short ones, flatter F0 typical of Telugu read speech, the comma pause in T4.]]

## 6. English vs Telugu — comparison, limitations, failures, improvements

[[Compare: data (multi-speaker accented English, ~5 min/speaker vs. single-speaker studio Telugu), pitch range, speaking rate (see `speech_rate_phones_per_s` in Results/*/inference_summary.json: English {{EN_RATE}} vs Telugu {{TE_RATE}} phones/s), which model sounds cleaner and why (single speaker + phonemic script usually makes Telugu more stable). Failures observed on the test sentences. Limitations: 5 h of data, no MFA alignments, universal (not fine-tuned) vocoder, rule-based Telugu G2P ignores schwa/vowel-length subtleties in loan words, no Telugu-specific handling of English loan words, English G2P for code-mixed words. Improvements: more data/steps, fine-tune HiFi-GAN on each dataset, speaker-balanced selection or a single clean English speaker, better text normalisation, phoneme-level prosody/style embeddings, mixed English–Telugu phoneme set for code-mixing.]]

## 7. Reproducibility

Python ≥ 3.9, PyTorch ≥ 2.0; dependencies in `requirements.txt`. Seed 1234 for data selection and training.

```
pip install -r requirements.txt
# English (Svarah is gated: accept the terms on huggingface.co and create a token)
cd English
python prepare_data.py --hf_token hf_xxx   # download + 5 h / 1 h selection
python preprocess.py                       # features
python train.py                            # training (auto-resume)
python inference.py                        # test sentences + variance plots
# Telugu
cd ../Telugu
python prepare_data.py && python preprocess.py && python train.py && python inference.py
# report
python Report/build_report.py
```

Configuration files: `English/config.yaml`, `Telugu/config.yaml`. Checkpoints: `English/checkpoints/final_fp16.pth`, `Telugu/checkpoints/final_fp16.pth` ({{CKPT_NOTE}}).

## References

1. Y. Ren et al., "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech," ICLR 2021.
2. ming024, FastSpeech2 (PyTorch), github.com/ming024/FastSpeech2.
3. R. Badlani et al., "One TTS Alignment to Rule Them All," ICASSP 2022.
4. J. Kong et al., "HiFi-GAN," NeurIPS 2020.
5. AI4Bharat, Svarah: Indian-accented English speech benchmark (2023).
6. IIT Madras, Indic TTS database (Telugu).
