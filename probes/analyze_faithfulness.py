#!/usr/bin/env python3
"""
Faithfulness: agreement between the universal Knowledge (solvability) probe's
prediction and the model's own verbalized judgment (judge verbal_annotation),
per example, per language, per model.

Usage:
    python analyze_faithfulness.py --model gemma-4-31B-it
"""

import argparse
import json
from pathlib import Path

import numpy as np
import joblib

import train_probe as tp

HERE = Path(__file__).resolve().parent
JUDGE_PROMPT = "english_judge_prompt"


def best_solvability_layer(model: str) -> int:
    ls = tp.build_label_source("solvability", None)
    summary_f = HERE / ls.subdir / "results" / model / "probe_summary.json"
    d = json.load(open(summary_f))
    best_L, best_score = None, -1
    for L, layer_d in d.items():
        uni = layer_d["probes"].get("universal", {})
        score = (uni.get("val_by_split") or {}).get("all", {}).get("val_roc_auc")
        if score is None:
            keys = [k for k in layer_d["probes"] if k != "universal"]
            scores = [layer_d["probes"][k].get("val_roc_auc") for k in keys]
            scores = [s for s in scores if s is not None]
            score = sum(scores) / len(scores) if scores else None
        if score is not None and score > best_score:
            best_score, best_L = score, L
    return int(best_L)


def per_example_faithfulness(
    model: str, language: str, layer: int, tokens_per_problem: int, seed: int,
):
    know_ls = tp.build_label_source("solvability", None)
    verb_ls = tp.build_label_source("verbalization", JUDGE_PROMPT)

    ckpt = HERE / "solvability" / "checkpoints" / model / f"probe_universal_layer{layer}.joblib"
    if not ckpt.exists():
        raise FileNotFoundError(f"Universal Knowledge probe checkpoint not found: {ckpt}")
    pipe = joblib.load(ckpt)

    rows = []
    for style in tp.styles_for_language(language):
        for solvability in ("solvable", "unsolvable"):
            bucket_dir = tp.OUTPUTS_ROOT / language / model / style / solvability
            if not bucket_dir.exists():
                continue
            meta = tp.load_latest_metadata(bucket_dir)
            if meta is None:
                continue

            verbal_labels = verb_ls._labels_for_bucket(model, language, style, solvability)

            for ex in meta:
                ex_idx = ex["example_idx"]
                if ex_idx not in verbal_labels:
                    continue

                npz_path = bucket_dir / tp.npz_name(ex_idx, layer)
                if not npz_path.exists():
                    continue
                data = np.load(npz_path)
                h = data["hidden_states"].astype(np.float32)
                T = len(h)

                if tokens_per_problem > 0 and T > tokens_per_problem:
                    rng = np.random.default_rng(seed + ex_idx)
                    sel = tp.sample_tokens_from_problem(h, tokens_per_problem, rng)
                else:
                    sel = np.arange(T)

                token_preds = pipe.predict(h[sel])
                know_pred = int(round(float(np.mean(token_preds))))
                verbal_label = verbal_labels[ex_idx]

                rows.append({
                    "example_idx": ex_idx,
                    "style": style,
                    "true_solvability": solvability,
                    "knowledge_pred": know_pred,
                    "verbal_label": verbal_label,
                    "agree": int(know_pred == verbal_label),
                })
    return rows


def summarize(rows):
    n = len(rows)
    if n == 0:
        return None
    agree = sum(r["agree"] for r in rows)

    # rows=knowledge_pred, cols=verbal_label
    conf = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    for r in rows:
        conf[(r["knowledge_pred"], r["verbal_label"])] += 1

    n_know_solvable = conf[(0, 0)] + conf[(0, 1)]
    n_know_unsolvable = conf[(1, 0)] + conf[(1, 1)]

    return {
        "n": n,
        "agreement_rate": round(agree / n, 4),
        "confusion": {
            "know_solvable_verbal_solved": conf[(0, 0)],
            "know_solvable_verbal_unsolvable": conf[(0, 1)],
            "know_unsolvable_verbal_solved": conf[(1, 0)],
            "know_unsolvable_verbal_unsolvable": conf[(1, 1)],
        },
        "underclaim_rate": round(conf[(0, 1)] / n_know_solvable, 4) if n_know_solvable else None,
        "overclaim_rate": round(conf[(1, 0)] / n_know_unsolvable, 4) if n_know_unsolvable else None,
    }


def normalize_style(style: str, language: str) -> str:
    l = language.lower()
    if style == f"standard_{l}":
        return "standard"
    if style == f"unsolvable_aware_multilingual_{l}":
        return "unsolvable_aware"
    return style


def write_table(f, title: str, entries):
    f.write(f"\n{title}\n")
    f.write(f"{'':<28}{'N':>6}{'Agreement':>12}{'Underclaim':>13}{'Overclaim':>12}\n")
    f.write("-" * 71 + "\n")
    for label, summ in entries:
        if summ is None:
            f.write(f"{label:<28}{'N/A':>6}\n")
            continue
        uc = f"{summ['underclaim_rate']:.4f}" if summ["underclaim_rate"] is not None else "N/A"
        oc = f"{summ['overclaim_rate']:.4f}" if summ["overclaim_rate"] is not None else "N/A"
        f.write(f"{label:<28}{summ['n']:>6}{summ['agreement_rate']:>12.4f}{uc:>13}{oc:>12}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--tokens_per_problem", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    languages = tp.languages_for_model(args.model)
    layer = args.layer or best_solvability_layer(args.model)
    print(f"model={args.model} layer={layer} languages={languages}")

    result = {"model": args.model, "layer": layer, "languages": {},
              "by_language_style": {}, "by_language_truesolv": {},
              "by_language_style_truesolv": {}}
    rows_by_lang = {}
    SPLITS = ["solvable", "unsolvable"]

    for lang in languages:
        rows = per_example_faithfulness(args.model, lang, layer, args.tokens_per_problem, args.seed)
        rows_by_lang[lang] = rows

        summ = summarize(rows)
        result["languages"][lang] = summ
        if summ:
            print(f"  {lang} (pooled): n={summ['n']} agreement={summ['agreement_rate']:.4f} "
                  f"underclaim={summ['underclaim_rate']} overclaim={summ['overclaim_rate']}")
        else:
            print(f"  {lang} (pooled): no data")

        # by true solvability (pooled across styles)
        result["by_language_truesolv"][lang] = {}
        for split in SPLITS:
            split_rows = [r for r in rows if r["true_solvability"] == split]
            split_summ = summarize(split_rows)
            result["by_language_truesolv"][lang][split] = split_summ
            if split_summ:
                print(f"    {lang}/{split}: n={split_summ['n']} "
                      f"agreement={split_summ['agreement_rate']:.4f}")

        # per-style breakdown within this language
        styles_here = sorted(set(r["style"] for r in rows))
        result["by_language_style"][lang] = {}
        result["by_language_style_truesolv"][lang] = {}
        for style in styles_here:
            style_rows = [r for r in rows if r["style"] == style]
            style_summ = summarize(style_rows)
            norm_style = normalize_style(style, lang)
            result["by_language_style"][lang][norm_style] = style_summ
            if style_summ:
                print(f"    {lang}/{norm_style}: n={style_summ['n']} "
                      f"agreement={style_summ['agreement_rate']:.4f}")

            # per-style, per-true-solvability
            result["by_language_style_truesolv"][lang][norm_style] = {}
            for split in SPLITS:
                sub_rows = [r for r in style_rows if r["true_solvability"] == split]
                sub_summ = summarize(sub_rows)
                result["by_language_style_truesolv"][lang][norm_style][split] = sub_summ

    out_dir = HERE / "cross_lingual_analysis" / "faithfulness"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{args.model}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    txt_path = out_dir / f"{args.model}.txt"
    with open(txt_path, "w") as f:
        f.write(f"Faithfulness report — {args.model}  (Knowledge layer {layer})\n")
        f.write("=" * 71 + "\n")
        f.write("Faithfulness = agreement between the universal Knowledge probe's\n")
        f.write("predicted solvability and the model's verbalized judgment (judge label).\n")

        write_table(f, "PER LANGUAGE (pooled across prompt styles and true solvability)",
                    [(lang, result["languages"][lang]) for lang in languages])

        for lang in languages:
            split_entries = [(f"{lang} / true={split}", result["by_language_truesolv"][lang][split])
                              for split in SPLITS]
            write_table(f, f"PER TRUE-SOLVABILITY SPLIT (pooled across styles) — {lang}", split_entries)

        for lang in languages:
            style_entries = [(f"{lang} / {style}", summ)
                              for style, summ in result["by_language_style"][lang].items()]
            write_table(f, f"PER PROMPT STYLE (pooled across true solvability) — {lang}", style_entries)

        for lang in languages:
            for style in result["by_language_style_truesolv"][lang]:
                split_entries = [
                    (f"{lang} / {style} / true={split}",
                     result["by_language_style_truesolv"][lang][style][split])
                    for split in SPLITS
                ]
                write_table(f, f"PER PROMPT STYLE x TRUE-SOLVABILITY — {lang} / {style}", split_entries)

        f.write("\nunderclaim_rate: of examples where the probe reads 'solvable', fraction "
                "the model still verbalized as unsolvable (hedging).\n")
        f.write("overclaim_rate: of examples where the probe reads 'unsolvable', fraction "
                "the model still verbalized as solved (confabulating/over-claiming).\n\n")

        def write_confusion(label, summ):
            if summ is None:
                return
            c = summ["confusion"]
            f.write(f"\n{label}:\n")
            f.write(f"{'':20}{'solved':>10}{'unsolvable':>12}\n")
            f.write(f"{'probe: solvable':20}{c['know_solvable_verbal_solved']:>10}"
                    f"{c['know_solvable_verbal_unsolvable']:>12}\n")
            f.write(f"{'probe: unsolvable':20}{c['know_unsolvable_verbal_solved']:>10}"
                    f"{c['know_unsolvable_verbal_unsolvable']:>12}\n")

        f.write("\n--- Confusion matrices (rows=knowledge probe, cols=verbalized) ---\n")
        for lang in languages:
            write_confusion(f"{lang} (pooled)", result["languages"][lang])

        for lang in languages:
            for split in SPLITS:
                write_confusion(f"{lang} / true={split}", result["by_language_truesolv"][lang][split])

        for lang in languages:
            for style, summ in result["by_language_style"][lang].items():
                write_confusion(f"{lang} / {style}", summ)

        for lang in languages:
            for style in result["by_language_style_truesolv"][lang]:
                for split in SPLITS:
                    write_confusion(f"{lang} / {style} / true={split}",
                                     result["by_language_style_truesolv"][lang][style][split])

    print(f"Saved -> {json_path}")
    print(f"Saved -> {txt_path}")


if __name__ == "__main__":
    main()