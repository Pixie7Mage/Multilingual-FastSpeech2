# FastSpeech 2 — English (Svarah) \& Telugu (IndicTTS)

Implementation of FastSpeech 2 (Ren et al., 2021) built on the PyTorch reference implementation
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2), trained **separately** for English and Telugu
on \~5 h of training audio and \~1 h of disjoint validation audio per language.

Main change vs. the reference: **no Montreal Forced Aligner** — durations are learned inside the model with an
unsupervised aligner (attention + CTC forward-sum + Monotonic Alignment Search, Badlani et al. 2022), so the same
pipeline works for Telugu, which has no MFA model. Telugu support (rule-based G2P + Telugu number expansion) is new.
See `Report/` for all modifications.

```
project/
├── fs2/                     shared code = modified reference implementation
│   ├── transformer/         encoder/decoder/postnet (reference)
│   ├── model/               fastspeech2.py, modules.py (variance adaptor), aligner.py (new), loss.py, optimizer.py
│   ├── text/                english.py (normaliser + g2p\_en), telugu.py (new Telugu G2P + numbers)
│   ├── hifigan/             pretrained universal HiFi-GAN vocoder (reference)
│   ├── prepare\_data.py      download + 5 h / 1 h subsampling
│   ├── preprocess\_core.py   mel / F0 / energy extraction
│   ├── train\_core.py        training loop
│   └── infer\_core.py        synthesis + variance-adaptor plots
├── English/  Telugu/        prepare\_data.py, preprocess.py, train.py, inference.py, config.yaml,
│                            test\_sentences.tsv, checkpoints/
├── Results/<lang>/          training\_loss.png, validation\_loss.png, test\_generated\_audios/,
│                            variance\_adaptor\_plots/{duration,pitch,energy}/, logs/
├── Report/                  build\_report.py -> Technical\_Report.pdf
├── colab\_run.ipynb          one-click Colab notebook
├── run\_all.sh               full pipeline for one language
└── requirements.txt
```

## 1\. Setup

A GPU is needed for training (Colab T4 works). Python ≥ 3.9.

```bash
pip install -r requirements.txt
```

**English data is gated.** Log in at huggingface.co, open https://huggingface.co/datasets/ai4bharat/Svarah and accept the
terms, then create a token at https://huggingface.co/settings/tokens. Telugu
(`SPRINGLab/IndicTTS\_Telugu`, \~6 GB) needs no token.

## 2\. Run (per language)

```bash
cd English                                   # or: cd Telugu
python prepare\_data.py --hf\_token hf\_xxx     # Telugu: python prepare\_data.py
python preprocess.py
python train.py                              # 50k steps by default; --max\_steps 30000 for a faster run
python inference.py                          # 5 test sentences -> audio + plots
```

or simply `HF\_TOKEN=hf\_xxx bash run\_all.sh English` and `bash run\_all.sh Telugu`.

**Google Colab:** upload `FastSpeech2\_project.zip` to the root of Google Drive, open `colab\_run.ipynb` in Colab
(GPU runtime), set `LANG`, `HF\_TOKEN`, run all. Checkpoints are kept on Drive and training resumes automatically after a
disconnect. Use two Colab sessions to train English and Telugu in parallel.

Approximate time on a T4: data download+preprocessing 20–40 min, training \~0.25–0.35 s/step (50k steps ≈ 4 h).
Intelligible speech usually appears around 15–20k steps; check `Results/<lang>/logs/debug/` — the alignment image
should become a clean diagonal within the first few thousand steps.

## 3\. Outputs

|What|Where|
|-|-|
|Inference checkpoint (fp16, < 100 MB)|`<lang>/checkpoints/final\_fp16.pth` (also `best\_fp16.pth`)|
|Loss curves|`Results/<lang>/training\_loss.png`, `validation\_loss.png` (raw CSVs in `logs/`)|
|Test audio|`Results/<lang>/test\_generated\_audios/\*.wav`|
|Variance-adaptor plots|`Results/<lang>/variance\_adaptor\_plots/{duration,pitch,energy}/<ID>.png`|
|Per-phoneme duration/F0/energy values|`Results/<lang>/inference\_summary.json`|
|Data selection / preprocessing stats|`data/<lang>/raw/selection\_stats.json`, `data/<lang>/preprocessed/summary.json`|

## 4\. Report

```bash
python Report/build\_report.py --author "Your Name" --repo https://github.com/you/repo --gpu "Tesla T4"
```

This fills `Report/Technical\_Report.md/.html` with all numbers and figures. Replace the yellow `\[\[...]]`
placeholders with your own listening/plot observations, rebuild, and export `Report/Technical\_Report.pdf`
(automatic if Chrome is installed, otherwise open the HTML in a browser → Print → Save as PDF).

## 5\. Submitting to GitHub

Commit only `final\_fp16.pth` (and optionally `best\_fp16.pth`) — `.gitignore` already excludes `latest.pth`,
`step\_\*.pth` and `data/`. If a checkpoint is ever > 100 MB, use Git LFS (`git lfs track "\*.pth"`) or upload it to
Google Drive and put the link here:

* English checkpoint: \[\[link if not in repo]]
* Telugu checkpoint: \[\[link if not in repo]]

## 6\. Custom text

```bash
python inference.py --text "Any sentence you like." --id demo --duration\_control 1.1 --pitch\_control 1.0
```

\## Trained Model Checkpoints



The trained English FastSpeech 2 checkpoints are available through Google Drive.



\*\*English checkpoints:\*\*

\- \[Best Checkpoint (`best\_fp16.pth`)](https://drive.google.com/drive/folders/1kRxKmoZm9PmsJplgwsW38LgPqzzbKc7x)

\- \[Final Checkpoint (`final\_fp16.pth`)](https://drive.google.com/drive/folders/1kRxKmoZm9PmsJplgwsW38LgPqzzbKc7x)



The best checkpoint was selected as the primary model based on listening comparisons. The final checkpoint corresponds to the completion of 50,000 training steps.



The files are hosted externally because of GitHub's file-size limitations.
