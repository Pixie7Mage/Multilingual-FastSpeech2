# FastSpeech 2 — English (Svarah) & Telugu (IndicTTS)

Implementation of **FastSpeech 2** (Ren et al., 2021), based on the PyTorch reference implementation [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2), trained **separately** for English and Telugu using approximately 5 hours of training audio and 1 hour of disjoint validation audio per language.

The main modification to the reference implementation is the removal of **Montreal Forced Aligner (MFA)**. Instead, durations are learned inside the model using an unsupervised aligner based on attention, CTC forward-sum loss and Monotonic Alignment Search (MAS), following Badlani et al. (2022). This allows the same pipeline to support Telugu, for which an MFA model is unavailable.

Telugu-specific modifications include rule-based Grapheme-to-Phoneme (G2P) conversion, Telugu text normalization and number expansion.

See `Report/` for the complete implementation details, modifications, experiments and analysis.

---

## Project Structure

```text
Multilingual-FastSpeech2/
│
├── fs2/                              Shared modified FastSpeech 2 implementation
│   ├── transformer/                  Encoder, decoder and postnet
│   ├── model/
│   │   ├── fastspeech2.py
│   │   ├── modules.py                Variance adaptor
│   │   ├── aligner.py                Unsupervised duration aligner
│   │   ├── loss.py
│   │   └── optimizer.py
│   ├── text/
│   │   ├── english.py                English normalization and G2P
│   │   └── telugu.py                 Telugu G2P and number expansion
│   ├── hifigan/                      HiFi-GAN vocoder implementation
│   ├── prepare_data.py
│   ├── preprocess_core.py
│   ├── train_core.py
│   └── infer_core.py
│
├── English/
│   ├── prepare_data.py
│   ├── preprocess.py
│   ├── train.py
│   ├── inference.py
│   ├── config.yaml
│   ├── checkpoints/
│   ├── checkpoints_v2/
│   └── checkpoints_v3/               Checkpoints hosted on Google Drive
│
├── Telugu/
│   ├── prepare_data.py
│   ├── preprocess.py
│   ├── train.py
│   ├── inference.py
│   ├── config.yaml
│   └── checkpoints/                  Checkpoints hosted on Google Drive
│
├── Results/
│   │
│   ├── English/                      Original English experiment
│   ├── English_best/                 Best-checkpoint experiment
│   ├── English_final/                Final experiment
│   ├── English_v2/                   English version 2
│   ├── English_v3/                   English version 3 (primary final results)
│   │
│   ├── Telugu/                       Main Telugu assignment results
│   ├── Telugu_reference_eval/        Telugu reference evaluation
│   └── Telugu_unseen/                Telugu unseen-sentence evaluation
│
├── Report/
│   ├── Technical_Report.md
│   └── build_report.py
│
├── colab_run.ipynb                   Google Colab notebook
├── run_all.sh                        Complete pipeline for one language
├── requirements.txt
├── .gitignore
└── README.md
```

**Checkpoint note:** Trained model checkpoints and pretrained HiFi-GAN weights are hosted on Google Drive instead of being committed to GitHub. Refer to Section 9 for exact download locations and checkpoint mapping.

---

## 1. Reference Implementation and Modifications

**Reference implementation:** [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)

The reference implementation was adapted to support independent English and Telugu training pipelines.

| Component | Implementation |
|---|---|
| Encoder | Transformer encoder from reference implementation |
| Decoder | Transformer decoder from reference implementation |
| Postnet | Mel-spectrogram refinement |
| Variance adaptor | Duration, pitch and energy prediction |
| Duration alignment | Attention + CTC forward-sum + Monotonic Alignment Search |
| Vocoder | Universal HiFi-GAN |
| English text processing | English normalization and G2P |
| Telugu text processing | Rule-based Telugu G2P and number expansion |

### Major modifications

- Removed the dependency on Montreal Forced Aligner.
- Introduced an unsupervised aligner for duration supervision.
- Added Telugu-specific grapheme-to-phoneme conversion.
- Added Telugu text normalization and number expansion.
- Adapted preprocessing for the Svarah and IndicTTS Telugu datasets.
- Maintained separate configurations and training pipelines for both languages.
- Added inference-time duration, pitch and energy visualizations.

The shared modifications are maintained in `fs2/`, while language-specific scripts and configurations are maintained separately in `English/` and `Telugu/`.

---

## 2. Datasets and Data Preparation

The models were trained independently using the following datasets.

| Language | Dataset | Hugging Face repository |
|---|---|---|
| English | Svarah | [ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) |
| Telugu | IndicTTS Telugu | [SPRINGLab/IndicTTS_Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu) |

### Dataset selection

- Approximately 5 hours of audio were selected for training.
- Approximately 1 hour of separate audio was selected for validation.
- Training and validation samples were kept disjoint.
- Dataset selection and subsampling statistics were generated during preparation.

### Preprocessing

The preprocessing pipeline includes:

1. Dataset downloading and selection.
2. Audio loading and validation.
3. Language-specific text normalization.
4. Grapheme-to-phoneme conversion.
5. Mel-spectrogram extraction.
6. Fundamental frequency (F0/pitch) extraction.
7. Energy extraction.
8. Preparation of the processed data required for training.

### Dataset access

**English — Svarah**

English Svarah is a gated Hugging Face dataset.

1. Open [ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah).
2. Log in and accept the dataset access terms.
3. Generate an access token from [Hugging Face Settings](https://huggingface.co/settings/tokens).
4. Use the token during English data preparation.

**Telugu — IndicTTS Telugu**

Telugu data is downloaded from [SPRINGLab/IndicTTS_Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu).

Dataset files and preprocessed audio are not included in GitHub because of their size. They are downloaded and prepared through the corresponding data preparation scripts.

---

## 3. Setup

### Requirements

- Python 3.9 or higher
- PyTorch
- CUDA-enabled GPU recommended for training
- Google Colab T4 or equivalent GPU

Install the dependencies:

```bash
pip install -r requirements.txt
```

### English authentication

Set the Hugging Face token as an environment variable:

```bash
export HF_TOKEN=hf_xxx
```

For Windows PowerShell:

```powershell
$env:HF_TOKEN="hf_xxx"
```

The token is required for accessing the gated English Svarah dataset.

---

## 4. Running the Pipeline

Each language has an independent data preparation, preprocessing, training and inference pipeline.

### English

```bash
cd English

python prepare_data.py --hf_token hf_xxx
python preprocess.py
python train.py
python inference.py
```

### Telugu

```bash
cd Telugu

python prepare_data.py
python preprocess.py
python train.py
python inference.py
```

Alternatively, run the complete pipeline from the project root:

```bash
HF_TOKEN=hf_xxx bash run_all.sh English
```

```bash
bash run_all.sh Telugu
```

Training configurations are maintained separately:

- `English/config.yaml`
- `Telugu/config.yaml`

The training scripts use the configurations defined in their respective YAML files.

### Google Colab

The project includes `colab_run.ipynb` for running the pipeline in Google Colab.

1. Upload `FastSpeech2_project.zip` to the root of Google Drive.
2. Open `colab_run.ipynb` in Google Colab.
3. Select a GPU runtime.
4. Set the language and Hugging Face token where required.
5. Execute the notebook.

English and Telugu can be trained in separate Colab sessions.

Training checkpoints can be maintained on Google Drive to support resuming after runtime disconnections.

### Approximate training time

On a Tesla T4 GPU:

- Data downloading and preprocessing: approximately 20–40 minutes.
- Training speed: approximately 0.25–0.35 seconds per step.
- 50,000 training steps: approximately 4 hours.

Intelligible speech may begin appearing around 15,000–20,000 steps. Alignment debug images are generated under the corresponding results/log directories.

---

## 5. Training Configuration and Evaluation

The two language models were trained separately.

| Language | Final training steps | Final checkpoint |
|---|---:|---|
| English | 75,000 | `English/checkpoints_v3/final_fp16.pth` |
| Telugu | 50,000 | `Telugu/checkpoints/final_fp16.pth` |

### English evaluation

- Final model: 75,000 training steps.
- WER on the held-out evaluation samples: 31.2%.
- Real-audio evaluation WER: 9.0%.
- Vocoded real-mel evaluation WER: 9.9%.

### Telugu evaluation

- Final model: 50,000 training steps.
- Average raw CER: 13.11%.
- Average normalized CER: 9.98%.
- Corpus normalized CER: 10.84%.

The English WER values and Telugu CER values are based on their respective evaluation procedures and should not be directly compared.


---

## 6. Test Sentences and Generated Audio

Speech synthesis was performed using the test sentences provided in the assignment.

### English test samples

| ID | Test sentence |
|---|---|
| E1 | The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings. |
| E2a | The orchestra's conduct during the ceremony was praised by the audience. |
| E2b | The researchers will conduct an experiment to evaluate the proposed method. |
| E3 | The rhythm of the orchestra changed unexpectedly. |
| E4 | The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time. |

### Telugu test samples

| ID | Test sentence |
|---|---|
| T1 | న్యూ రోఫార్మ కోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు. |
| T2a | తాడుపై అతని పట్టు చాలా బలంగా ఉంది. |
| T2b | ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది. |
| T3 | పులి అడవిలో నిశ్శబ్దంగా నడిచింది. |
| T4 | ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి. |

Each language model generates speech for the five assignment test samples.

The generated audio files are stored in the corresponding `test_generated_audios/` directory within the results folders described in Section 7.

---

## 7. Results and Generated Outputs

Results from different experiments and evaluation runs are maintained in separate directories to distinguish the final models from earlier experiments.

### 7.1 English Results

Five English result directories are maintained.

```text
Results/
├── English/
├── English_best/
├── English_final/
├── English_v2/
└── English_v3/
```

| Directory | Description |
|---|---|
| `Results/English/` | Original English experiment outputs |
| `Results/English_best/` | Best-checkpoint experiment outputs |
| `Results/English_final/` | Final experiment outputs |
| `Results/English_v2/` | English version 2 experimental outputs |
| `Results/English_v3/` | English version 3 outputs — primary final results |

**Primary English results: `Results/English_v3/`**

The `English_v3` results correspond to the final 75,000-step English model.

The primary assignment outputs are organized as follows:

```text
Results/English_v3/
├── training_loss.png
├── validation_loss.png
├── logs/
│   ├── train_log.csv
│   ├── val_log.csv
│   └── debug/
├── test_generated_audios/
│   ├── E1.wav
│   ├── E2a.wav
│   ├── E2b.wav
│   ├── E3.wav
│   └── E4.wav
├── variance_adaptor_plots/
│   ├── duration/
│   ├── pitch/
│   └── energy/
└── inference_summary.json
```

The remaining English directories preserve outputs from their respective experiments and checkpoint versions. They are kept separate to avoid mixing results from different training runs.

### 7.2 Telugu Results

Three separate Telugu results directories are maintained.

```text
Results/
├── Telugu/
├── Telugu_reference_eval/
└── Telugu_unseen/
```

**1. Main Telugu assignment results**

Location: `Results/Telugu/`

```text
Results/Telugu/
├── training_loss.png
├── validation_loss.png
├── inference_summary.json
├── logs/
│   ├── train_log.csv
│   ├── val_log.csv
│   └── debug/
├── test_generated_audios/
│   ├── T1.wav
│   ├── T2a.wav
│   ├── T2b.wav
│   ├── T3.wav
│   └── T4.wav
└── variance_adaptor_plots/
    ├── duration/
    │   ├── T1.png
    │   ├── T2a.png
    │   ├── T2b.png
    │   ├── T3.png
    │   └── T4.png
    ├── pitch/
    │   ├── T1.png
    │   ├── T2a.png
    │   ├── T2b.png
    │   ├── T3.png
    │   └── T4.png
    └── energy/
        ├── T1.png
        ├── T2a.png
        ├── T2b.png
        ├── T3.png
        └── T4.png
```

This directory contains the five assignment test audio samples, training and validation loss curves, logs and the corresponding duration, pitch and energy visualizations.

**2. Telugu reference evaluation**

Location: `Results/Telugu_reference_eval/`

Contains the generated outputs and evaluation artifacts associated with reference evaluation of the Telugu model.

**3. Telugu unseen evaluation**

Location: `Results/Telugu_unseen/`

Contains the generated outputs and evaluation artifacts associated with unseen-sentence evaluation of the Telugu model.

The reference and unseen evaluation directories are maintained separately from the main assignment results to preserve the additional evaluation experiments.

### 7.3 Result Summary

| Language | Results directory | Purpose |
|---|---|---|
| English | `Results/English/` | Original experiment |
| English | `Results/English_best/` | Best-checkpoint experiment |
| English | `Results/English_final/` | Final experiment |
| English | `Results/English_v2/` | Version 2 experiment |
| English | `Results/English_v3/` | Primary final English results |
| Telugu | `Results/Telugu/` | Main assignment results |
| Telugu | `Results/Telugu_reference_eval/` | Reference evaluation |
| Telugu | `Results/Telugu_unseen/` | Unseen-sentence evaluation |

---

## 8. Variance Adaptor Analysis

The variance adaptor predicts three important speech features:

**1. Duration**

Predicts the duration of each phoneme, representing how long it is spoken.

**2. Pitch (F0)**

Predicts the fundamental frequency, representing pitch variations in speech.

**3. Energy**

Predicts speech energy, representing variations in speech intensity.

### English variance-adaptor plots

Primary location:

```text
Results/English_v3/variance_adaptor_plots/
├── duration/
├── pitch/
└── energy/
```

Each directory contains the corresponding plots for the five English test samples.

### Telugu variance-adaptor plots

Main assignment location:

```text
Results/Telugu/variance_adaptor_plots/
├── duration/
├── pitch/
└── energy/
```

Each directory contains the corresponding plots for the five Telugu test samples.

Additional Telugu evaluation plots are maintained in:

```text
Results/Telugu_reference_eval/
Results/Telugu_unseen/
```

The technical report discusses the observed duration, pitch and energy patterns, including language-specific behaviour and limitations.

---

## 9. Model Checkpoints and HiFi-GAN Vocoder

The trained model checkpoints and pretrained HiFi-GAN weights are hosted on Google Drive because they are not committed to GitHub.

### Google Drive — Checkpoints and Vocoder

[Download FastSpeech 2 Checkpoints and HiFi-GAN Weights](https://drive.google.com/drive/u/0/folders/1VTgvtLR593rNAbip7GCTSoHMjQIJSV49)

The Drive folder contains the following checkpoint organization:

```text
FastSpeech2_Checkpoints/
│
├── English/
│   ├── English_v2/
│   │   ├── best_fp16.pth
│   │   └── final_fp16.pth
│   │
│   └── English_v3/
│       ├── best_fp16.pth
│       └── final_fp16.pth
│
├── Telugu/
│   ├── best_fp16.pth
│   └── final_fp16.pth
│
└── HiFi-GAN/
    └── generator_universal.pth.tar
```

### Exact checkpoint mapping

| Checkpoint | Purpose | Expected project location |
|---|---|---|
| English v2 `best_fp16.pth` | Best English v2 checkpoint | `English/checkpoints_v2/best_fp16.pth` |
| English v2 `final_fp16.pth` | Final English v2 checkpoint | `English/checkpoints_v2/final_fp16.pth` |
| English v3 `best_fp16.pth` | Best English v3 checkpoint | `English/checkpoints_v3/best_fp16.pth` |
| English v3 `final_fp16.pth` | Primary final English checkpoint (75k steps) | `English/checkpoints_v3/final_fp16.pth` |
| Telugu `best_fp16.pth` | Best Telugu checkpoint | `Telugu/checkpoints/best_fp16.pth` |
| Telugu `final_fp16.pth` | Primary final Telugu checkpoint (50k steps) | `Telugu/checkpoints/final_fp16.pth` |
| HiFi-GAN vocoder | Pretrained universal vocoder | `fs2/hifigan/generator_universal.pth.tar` |

**Official inference checkpoints:**

- English: `English/checkpoints_v3/final_fp16.pth`
- Telugu: `Telugu/checkpoints/final_fp16.pth`
- Vocoder: `fs2/hifigan/generator_universal.pth.tar`

The other English versions and best checkpoints are retained for experimental comparison and reference. They are not required for the primary inference pipeline.

### Download and use

1. Open the Google Drive checkpoint folder.
2. Download the required language checkpoint.
3. Download the HiFi-GAN vocoder weights.
4. Place the files at their corresponding project locations.
5. Run the language-specific inference script.


---

## 10. Technical Report

The complete technical report documents the implementation, architecture modifications, dataset preparation, training experiments, generated speech analysis, variance-adaptor visualizations, evaluation results and limitations.

**The technical report is hosted externally on Google Drive and is not included in the GitHub repository.**

[Access the Technical Report and Model Checkpoints on Google Drive](https://drive.google.com/drive/u/0/folders/1VTgvtLR593rNAbip7GCTSoHMjQIJSV49)

The report includes:
- FastSpeech 2 architecture and implementation approach.
- Modifications to the reference implementation for English and Telugu.
- Dataset selection, subsampling and preprocessing.
- Training configurations and convergence analysis.
- Generated speech observations.
- Duration, pitch and energy analysis.
- Comparative evaluation of English and Telugu.
- Limitations, improvements and reproducibility instructions.

---

## 11. Custom Text Inference

To synthesize speech for a custom sentence, use the language-specific inference script.

Example:

```bash
python inference.py \
    --text "Any sentence you like." \
    --id demo \
    --duration_control 1.1 \
    --pitch_control 1.0
```

The inference script uses the corresponding language configuration, trained checkpoint and HiFi-GAN vocoder to generate speech.

---

## 12. GitHub Submission and Reproducibility

The GitHub repository contains the complete implementation, language-specific scripts, configurations, results and technical report.

**Repository:** [Pixie7Mage/Multilingual-FastSpeech2](https://github.com/Pixie7Mage/Multilingual-FastSpeech2)

### Files maintained in GitHub

- Complete shared FastSpeech 2 implementation.
- English and Telugu preprocessing, training and inference scripts.
- Separate configuration files for both languages.
- English and Telugu generated audio samples.
- Training and validation loss plots where available.
- Duration, pitch and energy visualizations.
- Additional reference and unseen evaluation results.
- Technical report and report-generation script.
- Requirements and execution instructions.

### Files hosted externally

- English and Telugu trained model checkpoints.
- Pretrained HiFi-GAN vocoder weights.
- Large datasets and preprocessed data are downloaded separately through the preparation scripts.

Large intermediate training checkpoints, temporary files and dataset files are excluded from GitHub through `.gitignore`.

For reproducibility, download the appropriate trained checkpoint and vocoder weights from Google Drive, install the required dependencies, prepare the corresponding dataset and use the language-specific commands documented above.

**GitHub Repository:** [https://github.com/Pixie7Mage/Multilingual-FastSpeech2](https://github.com/Pixie7Mage/Multilingual-FastSpeech2)

**Google Drive — Model Checkpoints, Vocoder and Technical Report:** [Open Drive Folder](https://drive.google.com/drive/u/0/folders/1VTgvtLR593rNAbip7GCTSoHMjQIJSV49)

**Reference Paper:** [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)

**Reference Implementation:** [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)
