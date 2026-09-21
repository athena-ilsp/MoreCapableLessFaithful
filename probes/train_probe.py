#!/usr/bin/env python3
"""
Solvability Belief (SB) linear probes on hidden states.

For a given --model this trains, per layer, one probe per language plus one
universal probe pooled over languages. Outputs go under
probes/solvability/{checkpoints,results}/<model>/.

This module also exposes VerbalizationLabels / build_label_source, used by
analyze_faithfulness.py to read (not train on) the LLM judge's verdict.

Usage:
    python train_probe.py --model qwen3_4b_instruct --all_layers
"""

import re
import json
import argparse
import logging
import random
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------

HERE          = Path(__file__).resolve().parent
COT_ROOT      = Path(os.environ.get("COT_ROOT", HERE.parent / "cot_generation"))
OUTPUTS_ROOT  = COT_ROOT / "outputs"
JUDGE_ROOT    = COT_ROOT / "judge_outputs"
PROBE_ROOT    = HERE

ALL_LANGUAGES = ["English", "French", "Greek"]
JUDGE_PROMPTS = ["english_judge_prompt", "multilingual_judge_prompts"]

MODEL_LANG_RESTRICTION = {
    "Llama-Krikri-8B-Instruct": ["English", "Greek"],
    "French-Alpaca-Llama3-8B-Instruct-v1.0": ["English", "French"],
}

VERBAL_LABEL_MAP = {"solved": 0, "unsolvable": 1}


def languages_for_model(model: str) -> List[str]:
    return MODEL_LANG_RESTRICTION.get(model, ALL_LANGUAGES)


def styles_for_language(lang: str) -> List[str]:
    """The two prompt styles present consistently across every language."""
    l = lang.lower()
    return [f"standard_{l}", f"unsolvable_aware_multilingual_{l}"]


def npz_name(ex_idx: int, layer: int) -> str:
    return f"ex{ex_idx:06d}_layer{layer}.npz"


# ---------------------------------------------------------------------------
# Label sources — the only thing that differs between probe types
# ---------------------------------------------------------------------------

class LabelSource:
    """Provides per-token labels for a (language, style, solvability) bucket."""

    subdir: str = ""
    class_names = ("class0", "class1")

    def token_labels(self, ex: Dict, bucket_dir: Path, n_tokens: int) -> Optional[np.ndarray]:
        raise NotImplementedError

    def bucket_context(self, model: str, language: str, style: str, solvability: str):
        return None


class SolvabilityLabels(LabelSource):
    subdir = "solvability"
    class_names = ("solvable", "unsolvable")

    def token_labels(self, ex, bucket_dir, n_tokens):
        return ex["_token_solvability"]


class VerbalizationLabels(LabelSource):
    class_names = ("solved", "verbalized_unsolvable")

    def __init__(self, judge_prompt: str):
        self.judge_prompt = judge_prompt
        self.subdir = f"verbalization/{judge_prompt}"
        self._label_cache: Dict[Path, Dict[int, int]] = {}

    def _labels_for_bucket(self, model, language, style, solvability) -> Dict[int, int]:
        jd = JUDGE_ROOT / self.judge_prompt / language / model / style / solvability
        if jd in self._label_cache:
            return self._label_cache[jd]
        result: Dict[int, int] = {}
        f = next(jd.glob("judge_metadata_0_*.jsonl"), None)
        if f is not None:
            for line in open(f):
                row = json.loads(line)
                ann = row.get("verbal_annotation")
                if ann in VERBAL_LABEL_MAP:
                    result[row["example_idx"]] = VERBAL_LABEL_MAP[ann]
        self._label_cache[jd] = result
        return result

    def bucket_context(self, model, language, style, solvability):
        return self._labels_for_bucket(model, language, style, solvability)

    def token_labels(self, ex, bucket_dir, n_tokens):
        labels = ex["_ctx"]
        lab = labels.get(ex["example_idx"])
        if lab is None:
            return None
        return np.full(n_tokens, lab, dtype=np.int8)


def build_label_source(probe_type: str, judge_prompt: Optional[str]) -> LabelSource:
    if probe_type == "solvability":
        return SolvabilityLabels()
    if probe_type == "verbalization":
        if judge_prompt not in JUDGE_PROMPTS:
            raise ValueError(f"--judge_prompt must be one of {JUDGE_PROMPTS}")
        return VerbalizationLabels(judge_prompt)
    raise ValueError(f"unknown probe_type {probe_type}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_latest_metadata(d: Path) -> Optional[List[Dict]]:
    candidates = list(d.glob("metadata_0_*.json"))
    if not candidates:
        return None
    def end_idx(p):
        m = re.search(r"metadata_0_(\d+)\.json", p.name)
        return int(m.group(1)) if m else 0
    return json.load(open(max(candidates, key=end_idx)))


def problem_level_split(meta: List[Dict], val_frac: float, seed: int) -> Tuple[set, set]:
    all_idxs = [ex["example_idx"] for ex in meta]
    rng = random.Random(seed)
    shuffled = all_idxs[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    return set(shuffled[n_val:]), set(shuffled[:n_val])


def sample_tokens_from_problem(h: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n token positions spread across sequence depth via stratified binning."""
    T = len(h)
    if T <= n:
        return np.arange(T)
    bin_edges = np.linspace(0, T, n + 1).astype(int)
    return np.array([rng.integers(bin_edges[i], bin_edges[i + 1]) for i in range(n)])


def _concat_and_cap(Xs, ys, max_tok, rng_seed):
    if not Xs:
        return np.empty((0, 1)), np.empty(0, dtype=np.int8)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if max_tok > 0 and len(X) > max_tok:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(X), size=max_tok, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def load_tokens_for_bucket(
    label_source: LabelSource,
    model: str, language: str, style: str, solvability: str,
    layer: int, val_frac: float, seed: int,
    tokens_per_problem: int, tokens_per_style: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load hidden states + labels for one (language, style, solvability) bucket."""
    bucket_dir = OUTPUTS_ROOT / language / model / style / solvability
    meta = load_latest_metadata(bucket_dir)
    if meta is None:
        return np.empty((0, 1)), np.empty(0), np.empty((0, 1)), np.empty(0)

    ctx = label_source.bucket_context(model, language, style, solvability)
    train_idxs, _ = problem_level_split(meta, val_frac, seed)

    X_train, y_train, X_val, y_val = [], [], [], []
    for ex in meta:
        ex_idx = ex["example_idx"]
        npz_path = bucket_dir / npz_name(ex_idx, layer)
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        h = data["hidden_states"].astype(np.float32)  # [n_tokens, hidden_dim]

        ex["_ctx"] = ctx
        ex["_token_solvability"] = data["token_solvability"].astype(np.int8)
        lab = label_source.token_labels(ex, bucket_dir, len(h))
        if lab is None:
            continue

        if tokens_per_problem > 0:
            rng = np.random.default_rng(seed + ex_idx)
            sel = sample_tokens_from_problem(h, tokens_per_problem, rng)
            h, lab = h[sel], lab[sel]

        if ex_idx in train_idxs:
            X_train.append(h); y_train.append(lab)
        else:
            X_val.append(h);   y_val.append(lab)

    X_tr, y_tr = _concat_and_cap(X_train, y_train, tokens_per_style, seed)
    X_v,  y_v  = _concat_and_cap(X_val,   y_val,   tokens_per_style, seed + 1)
    return X_tr, y_tr, X_v, y_v


def collect_language(
    label_source: LabelSource, model: str, language: str, layer: int,
    tokens_per_problem: int, tokens_per_style: int, val_frac: int, seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool train/val over both styles and both solvability splits for one language."""
    all_X_tr, all_y_tr, all_X_v, all_y_v = [], [], [], []

    for style in styles_for_language(language):
        for solvability in ("solvable", "unsolvable"):
            if not (OUTPUTS_ROOT / language / model / style / solvability).exists():
                continue
            X_tr, y_tr, X_v, y_v = load_tokens_for_bucket(
                label_source, model, language, style, solvability, layer,
                val_frac, seed, tokens_per_problem, tokens_per_style,
            )
            if len(X_tr) > 0:
                all_X_tr.append(X_tr); all_y_tr.append(y_tr)
            if len(X_v) > 0:
                all_X_v.append(X_v);   all_y_v.append(y_v)
            logger.info(f"    {language}/{style}/{solvability}: "
                        f"train={len(X_tr):,} val={len(X_v):,}")

    dim = 1
    for arr in all_X_tr + all_X_v:
        if arr.ndim == 2 and arr.shape[1] > 1:
            dim = arr.shape[1]; break

    def cat(Xs, ys):
        if not Xs:
            return np.empty((0, dim)), np.empty(0, dtype=np.int8)
        return np.concatenate(Xs, 0), np.concatenate(ys, 0)

    X_tr, y_tr = cat(all_X_tr, all_y_tr)
    X_v,  y_v  = cat(all_X_v,  all_y_v)
    return X_tr, y_tr, X_v, y_v


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def build_pipe(C: float, max_iter: int, seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C, max_iter=max_iter, solver="lbfgs",
            class_weight="balanced", random_state=seed,
        )),
    ])


def eval_split(pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> Optional[Dict]:
    if len(X) < 10 or len(np.unique(y)) < 2:
        return None
    y_pred = pipe.predict(X)
    y_prob = pipe.predict_proba(X)[:, 1]
    return {
        "val_accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "val_roc_auc":  round(float(roc_auc_score(y, y_prob)), 4),
        "n_val": int(len(X)),
        "val_class_balance": round(float(y.mean()), 4),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }


def train_one_probe(
    name: str,
    X_train: np.ndarray, y_train: np.ndarray,
    val_sets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    C: float, max_iter: int, seed: int,
    primary_key: Optional[str] = None,
) -> Tuple[Optional[Pipeline], Dict]:
    """Train a probe and evaluate on every named val split (val_sets)."""
    if len(X_train) == 0 or len(np.unique(y_train)) < 2:
        logger.warning(f"  [{name}] insufficient training data — skipped.")
        return None, {"skipped": True, "n_train": int(len(X_train))}

    logger.info(f"  [{name}] Train: {X_train.shape}  pos_frac={y_train.mean():.3f}")
    pipe = build_pipe(C, max_iter, seed)
    pipe.fit(X_train, y_train)

    val_metrics = {}
    for key, (Xv, yv) in val_sets.items():
        m = eval_split(pipe, Xv, yv)
        if m is None:
            logger.info(f"    val[{key}]: (not evaluable, n={len(Xv)})")
            continue
        val_metrics[key] = m
        tag = " (train lang)" if key == name else ""
        logger.info(f"    val[{key}]{tag}: acc={m['val_accuracy']:.4f} "
                    f"auc={m['val_roc_auc']:.4f} n={m['n_val']}")

    key = primary_key or "all"
    primary = val_metrics.get(key) or next(iter(val_metrics.values()), {})
    metrics = {
        "n_train": int(len(X_train)),
        "train_class_balance": round(float(y_train.mean()), 4),
        "val_accuracy": primary.get("val_accuracy"),
        "val_roc_auc":  primary.get("val_roc_auc"),
        "val_by_split": val_metrics,
    }
    return pipe, metrics


# ---------------------------------------------------------------------------
# Per-layer driver
# ---------------------------------------------------------------------------

def run_layer(args, label_source: LabelSource, languages: List[str], layer: int) -> Dict:
    logger.info("=" * 64)
    logger.info(f"Model: {args.model} | Layer: {layer} | probe={label_source.subdir}")
    logger.info("=" * 64)

    out_base = PROBE_ROOT / label_source.subdir
    ckpt_dir = out_base / "checkpoints" / args.model
    res_dir  = out_base / "results" / args.model
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    per_lang_train: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    per_lang_val:   Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for lang in languages:
        if not (OUTPUTS_ROOT / lang / args.model).exists():
            logger.info(f"  (no data for {lang}/{args.model}, skipping)")
            continue
        logger.info(f"  Collecting {lang} ...")
        X_tr, y_tr, X_v, y_v = collect_language(
            label_source, args.model, lang, layer,
            tokens_per_problem=args.tokens_per_problem,
            tokens_per_style=args.tokens_per_style,
            val_frac=args.val_frac, seed=args.seed,
        )
        if len(X_tr) > 0:
            per_lang_train[lang] = (X_tr, y_tr)
        if len(X_v) > 0:
            per_lang_val[lang] = (X_v, y_v)

    layer_report: Dict[str, Dict] = {}

    # Per-language probes, each evaluated on every language's val set.
    for lang in languages:
        if lang not in per_lang_train:
            continue
        X_tr, y_tr = per_lang_train[lang]
        val_sets = {vl: per_lang_val[vl] for vl in per_lang_val}
        pipe, metrics = train_one_probe(
            lang, X_tr, y_tr, val_sets,
            C=args.C, max_iter=args.max_iter, seed=args.seed,
            primary_key=lang,
        )
        layer_report[lang] = metrics
        if pipe is not None:
            joblib.dump(pipe, ckpt_dir / f"probe_{lang}_layer{layer}.joblib")

    if len(per_lang_train) > 1:
        X_tr = np.concatenate([v[0] for v in per_lang_train.values()], 0)
        y_tr = np.concatenate([v[1] for v in per_lang_train.values()], 0)
        val_sets = {lang: per_lang_val[lang] for lang in per_lang_val}
        if per_lang_val:
            X_all = np.concatenate([v[0] for v in per_lang_val.values()], 0)
            y_all = np.concatenate([v[1] for v in per_lang_val.values()], 0)
            val_sets["all"] = (X_all, y_all)
        pipe, metrics = train_one_probe(
            "universal", X_tr, y_tr, val_sets,
            C=args.C, max_iter=args.max_iter, seed=args.seed,
        )
        layer_report["universal"] = metrics
        if pipe is not None:
            joblib.dump(pipe, ckpt_dir / f"probe_universal_layer{layer}.joblib")
    elif per_lang_train:
        logger.info("  (single-language model — universal probe == language probe, skipped)")

    out = {
        "model": args.model,
        "layer": layer,
        "languages": languages,
        "class_names": list(label_source.class_names),
        "tokens_per_problem": args.tokens_per_problem,
        "tokens_per_style": args.tokens_per_style,
        "val_frac": args.val_frac,
        "C": args.C,
        "probes": layer_report,
    }
    with open(res_dir / f"probe_layer{layer}_results.json", "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"  Saved -> {res_dir / f'probe_layer{layer}_results.json'}")
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_summary(model: str, res_dir: Path, languages: List[str], label_source: LabelSource):
    layer_files = sorted(
        res_dir.glob("probe_layer*_results.json"),
        key=lambda p: int(re.search(r"layer(\d+)", p.name).group(1)),
    )
    all_results = {json.load(open(f))["layer"]: json.load(open(f)) for f in layer_files}
    if not all_results:
        return

    with open(res_dir / "probe_summary.json", "w") as f:
        json.dump({str(l): m for l, m in all_results.items()}, f, indent=2)

    layers = sorted(all_results)
    has_universal = any("universal" in all_results[L]["probes"] for L in layers)
    txt = res_dir / "probe_summary.txt"
    with open(txt, "w") as f:
        title = f"{label_source.subdir}  probe results — {model}"
        f.write(title + "\n" + "=" * max(72, len(title)) + "\n")
        f.write(f"classes: 0={label_source.class_names[0]}  "
                f"1={label_source.class_names[1]}\n\n")

        f.write("PER-LANGUAGE PROBES — in-language val (acc / auc)\n")
        f.write("-" * 72 + "\n")
        f.write(f"{'Layer':>6}")
        for lang in languages:
            f.write(f"{lang+' acc':>16}{lang+' auc':>16}")
        f.write("\n")
        for L in layers:
            probes = all_results[L]["probes"]
            f.write(f"{L:>6}")
            for lang in languages:
                v = (probes.get(lang, {}).get("val_by_split", {}) or {}).get(lang, {})
                acc, auc = v.get("val_accuracy"), v.get("val_roc_auc")
                f.write(f"{acc if acc is not None else 'N/A':>16}"
                        f"{auc if auc is not None else 'N/A':>16}")
            f.write("\n")

        if len(languages) > 1:
            f.write("\n\nCROSS-LINGUAL TRANSFER (AUC) — rows=trained on, cols=evaluated on\n")
            f.write("-" * 72 + "\n")
            for L in layers:
                probes = all_results[L]["probes"]
                f.write(f"Layer {L}\n")
                corner = "trained/eval"
                header = f"{corner:<14}" + "".join(f"{c:>12}" for c in languages)
                f.write(header + "\n")
                for train_lang in languages:
                    vbs = (probes.get(train_lang, {}).get("val_by_split", {}) or {})
                    row = f"{train_lang:<14}"
                    for eval_lang in languages:
                        auc = vbs.get(eval_lang, {}).get("val_roc_auc")
                        row += f"{auc if auc is not None else 'N/A':>12}"
                    f.write(row + "\n")
                f.write("\n")

        if has_universal:
            f.write("\n\nUNIVERSAL PROBE  (trained on all languages pooled)\n")
            f.write("  val acc / auc per language and on the pooled mix\n")
            f.write("-" * 72 + "\n")
            split_order = languages + ["all"]
            f.write(f"{'Layer':>6}")
            for s in split_order:
                f.write(f"{s+' acc':>16}{s+' auc':>16}")
            f.write("\n")
            for L in layers:
                vbs = all_results[L]["probes"].get("universal", {}).get("val_by_split", {}) or {}
                f.write(f"{L:>6}")
                for s in split_order:
                    v = vbs.get(s, {})
                    acc, auc = v.get("val_accuracy"), v.get("val_roc_auc")
                    f.write(f"{acc if acc is not None else 'N/A':>16}"
                            f"{auc if auc is not None else 'N/A':>16}")
                f.write("\n")

    logger.info(f"Summary -> {res_dir / 'probe_summary.json'}")
    logger.info(f"Summary -> {txt}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_layers(model: str, languages: List[str]) -> List[int]:
    layers = set()
    for lang in languages:
        d = OUTPUTS_ROOT / lang / model
        if not d.exists():
            continue
        for p in d.rglob("ex000000_layer*.npz"):
            m = re.search(r"layer(\d+)\.npz", p.name)
            if m:
                layers.add(int(m.group(1)))
    return sorted(layers)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--all_layers", action="store_true")
    parser.add_argument("--tokens_per_problem", type=int, default=20)
    parser.add_argument("--tokens_per_style", type=int, default=50_000)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--C", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    label_source = SolvabilityLabels()
    languages = languages_for_model(args.model)

    if not any((OUTPUTS_ROOT / lang / args.model).exists() for lang in languages):
        raise FileNotFoundError(
            f"No data for model '{args.model}' under {OUTPUTS_ROOT}/<language>/")

    logger.info(f"probe={label_source.subdir}  model={args.model}  "
                f"languages={languages}")

    available = discover_layers(args.model, languages)
    logger.info(f"Available layers: {available}")

    if args.all_layers:
        layers = available
    elif args.layer is not None:
        if args.layer not in available:
            raise ValueError(f"Layer {args.layer} not in {available}")
        layers = [args.layer]
    else:
        raise ValueError("Specify --layer <N> or --all_layers")

    for layer in layers:
        run_layer(args, label_source, languages, layer)

    res_dir = PROBE_ROOT / label_source.subdir / "results" / args.model
    write_summary(args.model, res_dir, languages, label_source)
    logger.info("Done.")


if __name__ == "__main__":
    main()
