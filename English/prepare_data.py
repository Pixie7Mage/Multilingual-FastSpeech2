"""Step 0 - download the dataset and select ~5 h train / ~1 h validation.

    python prepare_data.py [--hf_token hf_xxx]

Writes ../data/English/raw/{wavs/, train.tsv, val.tsv, selection_stats.json}.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from fs2 import prepare_data  # noqa: E402
from fs2.utils import load_config  # noqa: E402

if __name__ == "__main__":
    cfg = load_config(os.path.join(HERE, "config.yaml"))
    if "--raw_dir" not in sys.argv:
        sys.argv += ["--raw_dir", cfg["path"]["raw_path"]]
    if "--dataset" not in sys.argv:
        sys.argv += ["--dataset", "svarah"]
    if "--max_dur" not in sys.argv:
        sys.argv += ["--max_dur", str(cfg["dataset"].get("max_duration_s", 15.0))]
    prepare_data.main()
