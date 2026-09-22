#!/usr/bin/env python3
"""
English-to-target-language dataset translation via OpenRouter (Claude Sonnet 4.6).
"""

import argparse
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("translate_dataset_openrouter")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# --------------------------------------------------------------------------
# I/O helpers (unchanged from the local-model version)
# --------------------------------------------------------------------------

def infer_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix in {"jsonl", "json", "csv", "parquet"}:
        return suffix
    raise ValueError(f"Unsupported extension: {path.suffix}. Use jsonl/json/csv/parquet or pass a supported file.")


def load_dataset(path: Path, fmt: str) -> Any:
    if fmt == "jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if fmt == "json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    if fmt == "csv":
        return pd.read_csv(path)
    if fmt == "parquet":
        return pd.read_parquet(path)
    raise ValueError(fmt)


def save_dataset(data: Any, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    if fmt == "json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return
    if fmt == "csv":
        data.to_csv(path, index=False)
        return
    if fmt == "parquet":
        data.to_parquet(path, index=False)
        return
    raise ValueError(fmt)


def flatten_json_records(obj: Any) -> List[Dict[str, Any]]:
    """Return row-like dicts for JSON/JSONL. Keeps references, so edits preserve structure."""
    if isinstance(obj, list):
        if not all(isinstance(x, dict) for x in obj):
            raise ValueError("For JSON lists, every item must be an object/dict.")
        return obj
    if isinstance(obj, dict):
        for key in ("data", "examples", "records", "rows"):
            if isinstance(obj.get(key), list) and all(isinstance(x, dict) for x in obj[key]):
                return obj[key]
    raise ValueError("JSON must be either a list of objects or a dict containing data/examples/records/rows.")


def guess_text_fields(records_or_df: Any) -> List[str]:
    if isinstance(records_or_df, pd.DataFrame):
        cols = list(records_or_df.columns)
        sample = records_or_df.head(50)
        return [c for c in cols if sample[c].map(lambda x: isinstance(x, str) and len(x.strip()) > 0).any()]
    records = flatten_json_records(records_or_df)
    fields = set()
    for row in records[:50]:
        for k, v in row.items():
            if isinstance(v, str) and v.strip():
                fields.add(k)
    return sorted(fields)


def batched(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def get_cell(data: Any, row_idx: int, field: str) -> Optional[str]:
    if isinstance(data, pd.DataFrame):
        value = data.at[row_idx, field]
        return value if isinstance(value, str) else None
    records = flatten_json_records(data)
    value = records[row_idx].get(field)
    return value if isinstance(value, str) else None


def collect_jobs(data: Any, fields: List[str]) -> List[Dict[str, Any]]:
    jobs = []
    if isinstance(data, pd.DataFrame):
        for row_idx in range(len(data)):
            for field in fields:
                value = data.at[row_idx, field]
                if isinstance(value, str) and value.strip():
                    jobs.append({"row_idx": row_idx, "field": field, "text": value})
        return jobs

    records = flatten_json_records(data)
    for row_idx, row in enumerate(records):
        for field in fields:
            value = row.get(field)
            if isinstance(value, str) and value.strip():
                jobs.append({"row_idx": row_idx, "field": field, "text": value})
    return jobs


def apply_translations(data: Any, jobs: List[Dict[str, Any]], translations: List[str]) -> None:
    if isinstance(data, pd.DataFrame):
        for job, translated in zip(jobs, translations):
            data.at[job["row_idx"], job["field"]] = translated
        return

    records = flatten_json_records(data)
    for job, translated in zip(jobs, translations):
        records[job["row_idx"]][job["field"]] = translated


def write_checkpoint(data: Any, output_path: Path, output_format: str, done_jobs: int) -> None:
    ckpt_path = output_path.with_suffix(output_path.suffix + f".checkpoint_{done_jobs}")
    save_dataset(data, ckpt_path, output_format)
    logger.info("Wrote checkpoint: %s", ckpt_path)


def progress_path_for(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".progress.jsonl")


def load_progress(path: Path) -> Dict[tuple, str]:
    """Load previously completed (row_idx, field) -> translated_text entries."""
    done: Dict[tuple, str] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[(rec["row_idx"], rec["field"])] = rec["text"]
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate a truncated last line from a killed process
    return done


def append_progress(path: Path, jobs: List[Dict[str, Any]], translations: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for job, translated in zip(jobs, translations):
            if not translated:
                continue  # don't record failed/empty translations as done
            f.write(json.dumps({"row_idx": job["row_idx"], "field": job["field"], "text": translated}, ensure_ascii=False) + "\n")


def find_latest_legacy_checkpoint(output_path: Path) -> Optional[Path]:
    """Find the highest-numbered output_path.checkpoint_N file from a pre-resume run."""
    pattern = output_path.name + ".checkpoint_*"
    candidates = []
    for p in output_path.parent.glob(pattern):
        m = re.search(r"\.checkpoint_(\d+)$", p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


def bootstrap_progress_from_checkpoint(
    ckpt_path: Path, output_format: str, original_data: Any, jobs: List[Dict[str, Any]]
) -> Dict[tuple, str]:
    """Diff a legacy checkpoint against the freshly loaded original data to recover
    which (row_idx, field) cells were already translated, for runs predating the
    progress-file resume mechanism."""
    ckpt_data = load_dataset(ckpt_path, output_format)
    completed: Dict[tuple, str] = {}
    for job in jobs:
        row_idx, field = job["row_idx"], job["field"]
        ckpt_value = get_cell(ckpt_data, row_idx, field)
        if ckpt_value and ckpt_value.strip() and ckpt_value != job["text"]:
            completed[(row_idx, field)] = ckpt_value
    return completed


# --------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------

def build_messages(text: str, source_lang: str, target_lang: str) -> List[Dict[str, str]]:
    system = (
        f"You are a professional {source_lang}-to-{target_lang} translator specializing in mathematical problems' translation. "
        f"Your goal is to accurately convey the meaning and nuances of the original {source_lang} text while adhering to {target_lang} grammar and vocabulary. "
        f"Produce only the {target_lang} translation, without any additional explanations or commentary."
    )

    user = f"""Translate the following {source_lang} mathematical problem into fluent, natural {target_lang}.

Requirements:
- Preserve the exact mathematical meaning.
- Do not solve the problem.
- Do not simplify, rewrite, or paraphrase the problem.
- Preserve all mathematical notation, LaTeX commands, equations, symbols, variable names, numbers, and units exactly as they appear.
- Preserve the original formatting whenever possible.
- Translate only the natural language text.
- Return only the {target_lang} translation. Do not include explanations or comments.

Problem:
{text}
"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def clean_translation(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:text)?\s*", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s*```$", "", s).strip()
    return s


# --------------------------------------------------------------------------
# OpenRouter call
# --------------------------------------------------------------------------

def call_openrouter(
    session: requests.Session,
    api_key: str,
    model: str,
    text: str,
    source_lang: str,
    target_lang: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    max_retries: int = 5,
    timeout: int = 120,
) -> str:
    messages = build_messages(text, source_lang, target_lang)
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional but recommended by OpenRouter for routing/analytics:
        "HTTP-Referer": "https://localhost",
        "X-Title": "dataset-translation",
    }

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise RuntimeError(f"Retryable HTTP {resp.status_code}: {resp.text[:300]}")
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            content = choice["message"]["content"]
            if isinstance(content, list):
                # Some providers return a list of content blocks.
                content = "".join(
                    block.get("text", "") for block in content if isinstance(block, dict)
                )
            return clean_translation(content)
        except Exception as e:  # noqa: BLE001
            last_err = e
            sleep_s = min(2 ** attempt, 30)
            logger.warning("Request failed (attempt %d/%d): %s. Retrying in %ds.", attempt, max_retries, e, sleep_s)
            time.sleep(sleep_s)

    raise RuntimeError(f"Exceeded max retries calling OpenRouter: {last_err}")


def translate_batch_concurrent(
    api_key: str,
    model: str,
    texts: List[str],
    source_lang: str,
    target_lang: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    max_workers: int,
    max_retries: int,
) -> List[str]:
    results: List[Optional[str]] = [None] * len(texts)
    with requests.Session() as session:
        adapter = requests.adapters.HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    call_openrouter,
                    session,
                    api_key,
                    model,
                    text,
                    source_lang,
                    target_lang,
                    max_tokens,
                    temperature,
                    top_p,
                    max_retries,
                ): idx
                for idx, text in enumerate(texts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:  # noqa: BLE001
                    logger.error("Translation failed for item %d: %s", idx, e)
                    results[idx] = ""  # leave empty on irrecoverable failure
    return [r if r is not None else "" for r in results]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default="anthropic/claude-sonnet-4.6",
        help="OpenRouter model slug to use (default: anthropic/claude-sonnet-4.6).",
    )
    p.add_argument(
        "--api_key",
        default=None,
        help="OpenRouter API key. If omitted, read from the OPENROUTER_API_KEY env var.",
    )
    p.add_argument("--input_path", required=True, help="Input dataset path: jsonl/json/csv/parquet.")
    p.add_argument("--output_path", default=None, help="Translated dataset path. If omitted, derived from --output_dir + input filename + target language.")
    p.add_argument("--output_dir", default=None, help="Folder to save the translated dataset into (created if missing). Used to auto-build --output_path when it's not given, e.g. /ReliableMath_greek_claude.")
    p.add_argument("--fields", nargs="+", default=None, help="Columns/keys to translate. Default: all non-empty string fields guessed from samples.")
    p.add_argument("--source_lang", default="English")
    p.add_argument("--target_lang", default="Greek")
    p.add_argument("--input_format", default=None, choices=["jsonl", "json", "csv", "parquet"])
    p.add_argument("--output_format", default=None, choices=["jsonl", "json", "csv", "parquet"])
    p.add_argument("--batch_size", type=int, default=16, help="Number of items submitted concurrently per batch.")
    p.add_argument("--max_workers", type=int, default=8, help="Concurrent in-flight requests to OpenRouter.")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_retries", type=int, default=5, help="Retries per request on transient failures.")
    p.add_argument("--save_every", type=int, default=500, help="Checkpoint every N translated text cells. 0 disables checkpoints.")
    p.add_argument("--no_resume", action="store_true", help="Ignore any existing progress file and translate everything from scratch.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)

    if not args.output_path and not args.output_dir:
        raise ValueError("Pass either --output_path (exact file) or --output_dir (folder, filename is auto-built).")

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "No OpenRouter API key provided. Pass --api_key or set the OPENROUTER_API_KEY env var."
        )
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    input_format = args.input_format or infer_format(input_path)

    if args.output_path:
        output_path = Path(args.output_path)
        output_format = args.output_format or infer_format(output_path)
    else:
        output_format = args.output_format or input_format
        lang_slug = re.sub(r"[^a-z0-9]+", "", args.target_lang.lower())
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{input_path.stem}_{lang_slug}.{output_format}"

    data = load_dataset(input_path, input_format)
    fields = args.fields or guess_text_fields(data)
    if not fields:
        raise ValueError("No text fields found. Pass --fields explicitly.")

    logger.info("Input: %s", input_path)
    logger.info("Output: %s", output_path)
    logger.info("Model: %s", args.model)
    logger.info("Fields to translate: %s", fields)

    jobs = collect_jobs(data, fields)
    total_jobs = len(jobs)
    logger.info("Text cells to translate: %d", total_jobs)

    prog_path = progress_path_for(output_path)
    completed: Dict[tuple, str] = {}
    if not args.no_resume:
        completed = load_progress(prog_path)
        if not completed:
            ckpt_path = find_latest_legacy_checkpoint(output_path)
            if ckpt_path is not None:
                logger.info("No progress file found; bootstrapping resume from legacy checkpoint: %s", ckpt_path)
                completed = bootstrap_progress_from_checkpoint(ckpt_path, output_format, data, jobs)
                if completed:
                    # Persist as a proper progress file so future resumes don't need to re-diff.
                    append_progress(
                        prog_path,
                        [{"row_idx": k[0], "field": k[1]} for k in completed.keys()],
                        list(completed.values()),
                    )
        if completed:
            apply_translations(
                data,
                [{"row_idx": k[0], "field": k[1]} for k in completed.keys()],
                list(completed.values()),
            )
            jobs = [j for j in jobs if (j["row_idx"], j["field"]) not in completed]
            logger.info(
                "Resuming: %d/%d cells already translated, %d remaining.",
                len(completed), total_jobs, len(jobs),
            )
    elif prog_path.exists():
        prog_path.unlink()
        logger.info("--no_resume set: deleted existing progress file %s", prog_path)

    processed = len(completed)
    for batch_jobs in tqdm(list(batched(jobs, args.batch_size)), desc="Translating batches"):
        batch_translations = translate_batch_concurrent(
            api_key=api_key,
            model=args.model,
            texts=[j["text"] for j in batch_jobs],
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
        )
        processed += len(batch_jobs)

        # Mutate data as we go so checkpoints contain progress.
        apply_translations(data, batch_jobs, batch_translations)
        append_progress(prog_path, batch_jobs, batch_translations)
        if args.save_every and processed % args.save_every < len(batch_jobs):
            write_checkpoint(data, output_path, output_format, processed)

    save_dataset(data, output_path, output_format)
    logger.info("Done. Saved translated dataset with identical structure to: %s", output_path)
    if prog_path.exists():
        prog_path.unlink()
        logger.info("Removed progress file (run completed): %s", prog_path)


if __name__ == "__main__":
    main()
