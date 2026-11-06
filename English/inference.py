"""Step 3 - synthesise the test sentences (English) + variance-adaptor plots.

    python inference.py                                  # all sentences in test_sentences.tsv
    python inference.py --text "your own sentence" --id demo
    python inference.py --ckpt checkpoints/best_fp16.pth

Outputs go to ../Results/English/{test_generated_audios, variance_adaptor_plots/{duration,pitch,energy}}.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from fs2.infer_core import read_sentences, synthesize  # noqa: E402
from fs2.utils import load_config  # noqa: E402

if __name__ == "__main__":
    cfg = load_config(os.path.join(HERE, "config.yaml"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "final_fp16.pth"))
    ap.add_argument("--sentences", default=os.path.join(HERE, "test_sentences.tsv"))
    ap.add_argument("--text", default=None)
    ap.add_argument("--id", default="custom")
    ap.add_argument("--out", default=cfg["path"]["result_path"])
    ap.add_argument("--speaker", default=None, help="speaker name from speakers.json (default: most data)")
    ap.add_argument("--pitch_control", type=float, default=1.0)
    ap.add_argument("--energy_control", type=float, default=1.0)
    ap.add_argument("--duration_control", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    ckpt = a.ckpt
    if not os.path.exists(ckpt):
        for alt in ["best_fp16.pth", "latest.pth", "best.pth"]:
            p = os.path.join(HERE, "checkpoints", alt)
            if os.path.exists(p):
                ckpt = p
                break
    sents = [(a.id, a.text)] if a.text else read_sentences(a.sentences)
    lex = cfg.get("text", {}).get("lexicon_override")
    synthesize(ckpt, sents, a.out, device=a.device, speaker=a.speaker, p_control=a.pitch_control,
               e_control=a.energy_control, d_control=a.duration_control, lexicon_override=lex)
