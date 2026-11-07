# FastSpeech 2 — English (Svarah) & Telugu (IndicTTS)

Implementation of FastSpeech 2 (Ren et al., 2021) built on the PyTorch reference implementation
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2), trained **separately** for English and Telugu
on ~5 h of training audio and ~1 h of disjoint validation audio per language.

Main change vs. the reference: **no Montreal Forced Aligner** — durations are learned inside the model with an
unsupervised aligner (attention + CTC forward-sum + Monotonic Alignment Search, Badlani et al. 2022), so the same
pipeline works for Telugu, which has no MFA model. Telugu support (rule-based G2P + Telugu number expansion) is new.
See `Report/` for all modifications.

```
project/
├── fs2/                     shared code = modified reference implementation
│   ├── transformer/         encoder/decoder/postnet (reference)
│   ├── model/               fastspeech2.py, modules.py (variance adaptor), aligner.py (new), loss.py, optimizer.py
│   ├── text/                english.py (normaliser + g2p_en), telugu.py (new Telugu G2P + numbers)
│   ├── hifigan/             pretrained universal HiFi-GAN vocoder (reference)
│   ├── prepare_data.py      download + 5 h / 1 h subsampling
│   ├── preprocess_core.py   mel / F0 / energy extraction
│   ├── train_core.py        training loop
│   └── infer_core.py        synthesis + variance-adaptor plots
├── English/  Telugu/        prepare_data.py, preprocess.py, train.py, inference.py, config.yaml,
│                            test_sentences.tsv, checkpoints/
├── Results/<lang>/          training_loss.png, validation_loss.png, test_generated_audios/,
│                            variance_adaptor_plots/{duration,pitch,energy}/, logs/
├── Report/                  build_report.py -> Technical_Report.pdf
├── colab_run.ipynb          one-click Colab notebook
├── run_all.sh               full pipeline for one language
└── requirements.txt
```

## 1. Setup

A GPU is needed for training (Colab T4 works). Python ≥ 3.9.

```bash
pip install -r requirements.txt
```

**English data is gated.** Log in at huggingface.co, open https://huggingface.co/datasets/ai4bharat/Svarah and accept the
terms, then create a token at https://huggingface.co/settings/tokens. Telugu
(`SPRINGLab/IndicTTS_Telugu`, ~6 GB) needs no token.

## 2. Run (per language)

```bash
cd English                                   # or: cd Telugu
python prepare_data.py --hf_token hf_xxx     # Telugu: python prepare_data.py
python preprocess.py
python train.py                              # 50k steps by default; --max_steps 30000 for a faster run
python inference.py                          # 5 test sentences -> audio + plots
```

or simply `HF_TOKEN=hf_xxx bash run_all.sh English` and `bash run_all.sh Telugu`.

**Google Colab:** upload `FastSpeech2_project.zip` to the root of Google Drive, open `colab_run.ipynb` in Colab
(GPU runtime), set `LANG`, `HF_TOKEN`, run all. Checkpoints are kept on Drive and training resumes automatically after a
disconnect. Use two Colab sessions to train English and Telugu in parallel.

Approximate time on a T4: data download+preprocessing 20–40 min, training ~0.25–0.35 s/step (50k steps ≈ 4 h).
Intelligible speech usually appears around 15–20k steps; check `Results/<lang>/logs/debug/` — the alignment image
should become a clean diagonal within the first few thousand steps.

## 3. Outputs

| What | Where |
|---|---|
| Inference checkpoint (fp16, < 100 MB) | `<lang>/checkpoints/final_fp16.pth` (also `best_fp16.pth`) |
| Loss curves | `Results/<lang>/training_loss.png`, `validation_loss.png` (raw CSVs in `logs/`) |
| Test audio | `Results/<lang>/test_generated_audios/*.wav` |
| Variance-adaptor plots | `Results/<lang>/variance_adaptor_plots/{duration,pitch,energy}/<ID>.png` |
| Per-phoneme duration/F0/energy values | `Results/<lang>/inference_summary.json` |
| Data selection / preprocessing stats | `data/<lang>/raw/selection_stats.json`, `data/<lang>/preprocessed/summary.json` |

## 4. Report

```bash
python Report/build_report.py --author "Your Name" --repo https://github.com/you/repo --gpu "Tesla T4"
```

This fills `Report/Technical_Report.md/.html` with all numbers and figures. Replace the yellow `[[...]]`
placeholders with your own listening/plot observations, rebuild, and export `Report/Technical_Report.pdf`
(automatic if Chrome is installed, otherwise open the HTML in a browser → Print → Save as PDF).

## 5. Submitting to GitHub

Commit only `final_fp16.pth` (and optionally `best_fp16.pth`) — `.gitignore` already excludes `latest.pth`,
`step_*.pth` and `data/`. If a checkpoint is ever > 100 MB, use Git LFS (`git lfs track "*.pth"`) or upload it to
Google Drive and put the link here:

- English checkpoint: [[link if not in repo]]
- Telugu checkpoint: [[link if not in repo]]

## 6. Custom text

```bash
python inference.py --text "Any sentence you like." --id demo --duration_control 1.1 --pitch_control 1.0
```
