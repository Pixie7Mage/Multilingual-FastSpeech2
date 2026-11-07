"""Builds Report/Technical_Report.md + .html (+ .pdf if a Chrome/Chromium is found)
from the template, filling numbers from data/, Results/ and checkpoints.

    python Report/build_report.py --author "Your Name" --repo https://github.com/you/repo [--gpu "Tesla T4"]

If no PDF is produced automatically: open Report/Technical_Report.html in a
browser -> Print -> Save as PDF -> Report/Technical_Report.pdf
(the browser renders Telugu script correctly).
"""
import argparse
import base64
import csv
import json
import os
import re
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _json(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _csv_last_and_best(p):
    try:
        with open(p) as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return "n/a", "n/a", "n/a"
    if not rows:
        return "n/a", "n/a", "n/a"
    def fmt(r):
        return " / ".join(f"{float(r[k]):.3f}" for k in ["mel", "postnet_mel", "duration", "pitch", "energy"])
    score = lambda r: sum(float(r[k]) for k in ["mel", "postnet_mel", "duration", "pitch", "energy"])
    best = min(rows, key=score)
    return fmt(rows[-1]), best["step"], rows[-1]["step"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default="[[Your name]]")
    ap.add_argument("--repo", default="[[GitHub URL]]")
    ap.add_argument("--gpu", default="[[GPU used]]")
    a = ap.parse_args()

    t = open(os.path.join(HERE, "Technical_Report_template.md"), encoding="utf-8").read()
    v = {"AUTHOR": a.author, "REPO_URL": a.repo, "GPU": a.gpu, "RES": "../Results"}
    rows, figs = [], []
    for L, P in [("English", "EN"), ("Telugu", "TE")]:
        sel = _json(os.path.join(ROOT, "data", L, "raw", "selection_stats.json"))
        summ = _json(os.path.join(ROOT, "data", L, "preprocessed", "summary.json"))
        inf = _json(os.path.join(ROOT, "Results", L, "inference_summary.json"))
        v[f"{P}_TRAIN_N"] = sel.get("train", {}).get("utterances", "n/a")
        v[f"{P}_TRAIN_H"] = sel.get("train", {}).get("hours", "n/a")
        v[f"{P}_VAL_N"] = sel.get("val", {}).get("utterances", "n/a")
        v[f"{P}_VAL_H"] = sel.get("val", {}).get("hours", "n/a")
        v[f"{P}_SPK"] = sel.get("train", {}).get("speakers", "n/a")
        v[f"{P}_SKIP"] = summ.get("skipped", "n/a")
        f0 = summ.get("pitch_hz_mean_std")
        v[f"{P}_F0"] = f"{f0[0]} ± {f0[1]}" if f0 else "n/a"
        v[f"{P}_SPEAKER"] = inf.get("speaker", "n/a")
        last, best, steps = _csv_last_and_best(os.path.join(ROOT, "Results", L, "logs", "val_log.csv"))
        v[f"{P}_VAL_FINAL"], v[f"{P}_VAL_BEST"], v[f"{P}_STEPS"] = last, best, steps
        rates = [s.get("speech_rate_phones_per_s", 0) for s in inf.get("sentences", {}).values()]
        v[f"{P}_RATE"] = f"{sum(rates)/len(rates):.1f}" if rates else "n/a"
        for sid, s in inf.get("sentences", {}).items():
            rows.append(f"| {sid} | {s['duration_s']} | `{s['phonemes']}` | [[observation for {sid}]] |")
            figs.append(f"### {L} — {sid}: {s['text']}\n")
            for k in ["duration", "pitch", "energy"]:
                figs.append(f"![{sid} {k}](../Results/{L}/variance_adaptor_plots/{k}/{sid}.png)")
            figs.append(f"\n[[Analysis of {sid}: duration / pitch / energy patterns]]\n")
    v["SENTENCE_TABLE"] = "\n".join(rows) or "| | | | |"
    v["VARIANCE_FIGS"] = "\n".join(figs)
    v["CKPT_NOTE"] = "fp16 weights, < 100 MB each; if larger, see the download link in README.md"
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", str(val))

    md_path = os.path.join(HERE, "Technical_Report.md")
    open(md_path, "w", encoding="utf-8").write(t)

    import markdown
    body = markdown.markdown(t, extensions=["tables", "fenced_code"])

    def embed(m):
        src = m.group(1)
        p = os.path.normpath(os.path.join(HERE, src))
        if os.path.exists(p):
            b64 = base64.b64encode(open(p, "rb").read()).decode()
            return f'src="data:image/png;base64,{b64}"'
        return f'src="{src}"'
    body = re.sub(r'src="([^"]+)"', embed, body)
    body = body.replace("[[", '<span class="todo">[[').replace("]]", "]]</span>")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Technical Report</title>
<style>
@page {{ size: A4; margin: 16mm; }}
body {{ font-family: 'Noto Sans', 'Noto Sans Telugu', 'Nirmala UI', 'Gautami', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; max-width: 900px; margin: auto; color: #111; }}
h1 {{ font-size: 19pt; }} h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 3px; margin-top: 26px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9pt; margin: 8px 0; }}
td, th {{ border: 1px solid #bbb; padding: 4px 6px; vertical-align: top; }} th {{ background: #eef3f8; }}
img {{ max-width: 100%; display: block; margin: 6px auto; page-break-inside: avoid; }}
code, pre {{ font-size: 8.5pt; background: #f5f5f5; }} pre {{ padding: 8px; white-space: pre-wrap; }}
.todo {{ background: #fff3b0; }}
blockquote {{ color: #555; border-left: 3px solid #ccc; margin-left: 0; padding-left: 10px; }}
</style></head><body>{body}</body></html>"""
    html_path = os.path.join(HERE, "Technical_Report.html")
    open(html_path, "w", encoding="utf-8").write(html)
    print("wrote", md_path, "and", html_path)
    if "[[" in t:
        print("NOTE: the report still contains [[...]] placeholders (highlighted yellow) — fill them in the .md/.html.")

    pdf = os.path.join(HERE, "Technical_Report.pdf")
    for exe in ["google-chrome", "chromium", "chromium-browser", "msedge", "chrome"]:
        b = shutil.which(exe)
        if b:
            try:
                subprocess.run([b, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                                f"--print-to-pdf={pdf}", "file://" + html_path], check=True, timeout=120,
                               capture_output=True)
                print("wrote", pdf)
                return
            except Exception as e:  # noqa
                print("PDF export with", exe, "failed:", e)
    print("No Chrome found: open Technical_Report.html in a browser and Print -> Save as PDF -> Technical_Report.pdf")


if __name__ == "__main__":
    main()
