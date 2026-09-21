#!/usr/bin/env python3
"""Compute answer accuracy on the SOLVABLE split per model / language / prompt.

Two modes:

  metadata (default)
      Reads the per-combo metadata_*.json files under
      outputs/<Lang>/<model>/<style>/solvable/ and compares the model's
      predicted answer to the ground truth. It reuses the correctness_label
      already computed at extraction time (predicted_answer vs
      ground_truth_clean), and also recomputes from raw fields as a
      cross-check. Accuracy = correct / graded, where graded = correct +
      wrong (None/ungradable answers are reported separately, not counted in
      the denominator).

  records
      Scores a single JSON file of {"ground_truth", "prediction"} records
      (e.g. a human-corrected translation export) by substring match: a
      prediction is correct if any (normalized) ground-truth value appears
      in it. Ground truth may be a scalar, a string-encoded list, or a list.

Usage:
    python compute_solvable_accuracy.py metadata [--outputs DIR] [--csv PATH] [--recompute]
    python compute_solvable_accuracy.py records path/to/file.json
"""
import ast
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"


def normalize(s):
    if s is None:
        return ""
    s = str(s).strip().lower().replace("\\", "").replace("$", "")
    s = re.sub(r"\s+", "", s).rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except (ValueError, TypeError):
        return s


def is_correct(pred, gt):
    if pred is None or gt is None:
        return None
    p, g = normalize(pred), normalize(gt)
    if not p or not g:
        return None
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        return False


def load_records(combo_dir):
    recs = []
    for mp in sorted(combo_dir.glob("metadata_*.json")):
        try:
            recs.extend(json.load(open(mp)))
        except (json.JSONDecodeError, OSError):
            print(f"  WARN: could not read {mp}")
    return recs


def run_metadata_mode(args):
    root = Path(args.outputs)

    rows = []
    for lang_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        lang = lang_dir.name
        for model_dir in sorted(p for p in lang_dir.iterdir() if p.is_dir()):
            model = model_dir.name
            for style_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                style = style_dir.name
                combo = style_dir / "solvable"
                if not combo.is_dir():
                    continue
                recs = load_records(combo)
                if not recs:
                    continue
                c = w = u = 0
                for r in recs:
                    if args.recompute:
                        lab = is_correct(r.get("predicted_answer"), r.get("ground_truth_clean"))
                    else:
                        lab = r.get("correctness_label")
                    if lab is True:
                        c += 1
                    elif lab is False:
                        w += 1
                    else:
                        u += 1
                graded = c + w
                acc = (c / graded) if graded else float("nan")
                rows.append((lang, model, style, c, w, u, len(recs), acc))

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    hdr = f"{'LANG':<8} {'MODEL':<40} {'PROMPT':<38} {'acc':>7} {'corr':>5} {'wrong':>6} {'ungr':>5} {'n':>5}"
    print(hdr)
    print("-" * len(hdr))
    for lang, model, style, c, w, u, n, acc in rows:
        print(f"{lang:<8} {model:<40} {style:<38} {acc*100:6.1f}% {c:5d} {w:6d} {u:5d} {n:5d}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["language", "model", "prompt_style", "accuracy",
                         "correct", "wrong", "ungraded", "n"])
            for lang, model, style, c, w, u, n, acc in rows:
                wr.writerow([lang, model, style, f"{acc:.4f}", c, w, u, n])
        print(f"\nCSV written -> {args.csv}")


def parse_ground_truth(gt):
    if isinstance(gt, list):
        return gt
    if isinstance(gt, (int, float)):
        return [gt]
    if isinstance(gt, str):
        try:
            parsed = ast.literal_eval(gt)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            return [gt]
    return [gt]


def record_is_correct(record):
    gts = [normalize(x) for x in parse_ground_truth(record["ground_truth"])]
    pred = normalize(record["prediction"])
    return any(gt in pred for gt in gts)


def run_records_mode(args):
    records = json.loads(Path(args.records_file).read_text(encoding="utf-8"))
    n_correct = 0
    for r in records:
        if record_is_correct(r):
            n_correct += 1
    n = len(records)
    acc = n_correct / n if n else float("nan")
    print(f"Accuracy: {acc:.2%}")
    print(f"Correct: {n_correct}/{n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    meta_ap = sub.add_parser("metadata", help="Score from cot_generation extraction outputs.")
    meta_ap.add_argument("--outputs", default=str(OUTPUTS))
    meta_ap.add_argument("--csv", default=None, help="Optional path to write results as CSV.")
    meta_ap.add_argument("--recompute", action="store_true",
                          help="Recompute correctness from predicted_answer/ground_truth_clean "
                               "instead of trusting stored correctness_label.")

    rec_ap = sub.add_parser("records", help="Score a JSON file of ground_truth/prediction records.")
    rec_ap.add_argument("records_file", help="Path to a JSON array of "
                         "{ground_truth, prediction} records.")

    args = ap.parse_args()
    if args.mode == "metadata":
        run_metadata_mode(args)
    else:
        run_records_mode(args)


if __name__ == "__main__":
    main()