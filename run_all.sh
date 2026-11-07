#!/usr/bin/env bash
# Full pipeline for one language:  bash run_all.sh English   |   bash run_all.sh Telugu
# Optional: MAX_STEPS=30000 bash run_all.sh Telugu
set -euo pipefail
LANG_DIR=${1:?usage: bash run_all.sh English|Telugu}
cd "$(dirname "$0")/$LANG_DIR"
export NLTK_ALLOW_PROXIED_URLOPEN=1

echo "=== [$LANG_DIR] 0/3 download + select data ==="
[ -f ../data/$LANG_DIR/raw/train.tsv ] || python prepare_data.py ${HF_TOKEN:+--hf_token $HF_TOKEN}

echo "=== [$LANG_DIR] 1/3 preprocess ==="
[ -f ../data/$LANG_DIR/preprocessed/train.txt ] || python preprocess.py

echo "=== [$LANG_DIR] 2/3 train ==="
python train.py ${MAX_STEPS:+--max_steps $MAX_STEPS}

echo "=== [$LANG_DIR] 3/3 inference + variance-adaptor plots ==="
python inference.py
echo "Done. See ../Results/$LANG_DIR/"
