#!/usr/bin/env bash

# Usage:
# bash run_all.sh English
# bash run_all.sh English v3
# bash run_all.sh English original
# bash run_all.sh Telugu
#
# Optional:
# MAX_STEPS=30000 bash run_all.sh Telugu
# HF_TOKEN=your_token bash run_all.sh English v3

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

LANG_DIR="${1:?Usage: bash run_all.sh English|Telugu [v3|original]}"
VERSION="${2:-v3}"

PYTHON="${PYTHON:-python}"
WORKERS=2

export NLTK_ALLOW_PROXIED_URLOPEN=1

# Select configuration, checkpoint and result directories
case "$LANG_DIR" in

    English)
        case "$VERSION" in
            v3)
                CONFIG="config_v3.yaml"
                CKPT_DIR="checkpoints_v3"
                RESULT_DIR="../Results/English_v3"
                ;;
            original)
                CONFIG="config.yaml"
                CKPT_DIR="checkpoints"
                RESULT_DIR="../Results/English"
                ;;
            *)
                echo "Invalid English version: $VERSION"
                echo "Use: v3 or original"
                exit 1
                ;;
        esac
        ;;

    Telugu)
        CONFIG="config.yaml"
        CKPT_DIR="checkpoints"
        RESULT_DIR="../Results/Telugu"
        ;;

    *)
        echo "Invalid language: $LANG_DIR"
        echo "Use: English or Telugu"
        exit 1
        ;;
esac

cd "$PROJECT_DIR/$LANG_DIR"

echo "======================================"
echo " FastSpeech 2 - $LANG_DIR"
echo " Configuration: $CONFIG"
echo " Checkpoints: $CKPT_DIR"
echo " Results: $RESULT_DIR"
echo "======================================"

# 0. Download and select data
echo "=== 0/3 Download + select data ==="

if [ ! -f "../data/$LANG_DIR/raw/train.tsv" ]; then

    if [ "$LANG_DIR" = "English" ]; then
        if [ -z "${HF_TOKEN:-}" ]; then
            echo "Error: HF_TOKEN is required for English (Svarah)."
            exit 1
        fi

        "$PYTHON" prepare_data.py --hf_token "$HF_TOKEN"
    else
        "$PYTHON" prepare_data.py
    fi

else
    echo "Raw dataset already exists. Skipping download."
fi

# 1. Preprocessing
echo "=== 1/3 Preprocessing ==="

if [ ! -f "../data/$LANG_DIR/preprocessed/train.txt" ]; then
    "$PYTHON" preprocess.py --workers "$WORKERS"
else
    echo "Preprocessed data already exists. Skipping."
fi

# 2. Training
echo "=== 2/3 Training ==="

TRAIN_CMD=(
    "$PYTHON"
    train.py
    --config "$CONFIG"
    --workers "$WORKERS"
)

if [ -n "${MAX_STEPS:-}" ]; then
    TRAIN_CMD+=(--max_steps "$MAX_STEPS")
fi

"${TRAIN_CMD[@]}"

# 3. Inference
echo "=== 3/3 Inference + variance-adaptor plots ==="

CHECKPOINT="$CKPT_DIR/final_fp16.pth"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

"$PYTHON" inference.py \
    --ckpt "$CHECKPOINT" \
    --out "$RESULT_DIR"

echo "======================================"
echo "Pipeline completed successfully!"
echo "Results saved in: $RESULT_DIR"
echo "======================================"
