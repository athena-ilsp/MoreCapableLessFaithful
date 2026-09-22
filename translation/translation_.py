#!/usr/bin/env python3
"""
English-to-target-language dataset translation.
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("translate_dataset_gemma")


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
        # Common HF-style shape: {"data": [{...}, ...]}.
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

def make_prompts(processor: Any, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
    prompts = []
    for text in texts:
        messages = build_messages(text, source_lang, target_lang)
        system = messages[0]["content"]
        user = messages[1]["content"]

        prompt = (
            "<start_of_turn>user\n"
            f"{system}\n\n{user}"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        prompts.append(prompt)
    return prompts

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True, help="Local path to Gemma 4 31B model directory.")
    p.add_argument("--input_path", required=True, help="Input dataset path: jsonl/json/csv/parquet.")
    p.add_argument("--output_path", required=True, help="Translated dataset path. Structure/format is preserved.")
    p.add_argument("--fields", nargs="+", default=None, help="Columns/keys to translate. Default: all non-empty string fields guessed from samples.")
    p.add_argument("--source_lang", default="English")
    p.add_argument("--target_lang", default="Greek")
    p.add_argument("--input_format", default=None, choices=["jsonl", "json", "csv", "parquet"])
    p.add_argument("--output_format", default=None, choices=["jsonl", "json", "csv", "parquet"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--tensor_parallel_size", type=int, default=4)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.70)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--save_every", type=int, default=500, help="Checkpoint every N translated text cells. 0 disables checkpoints.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    input_format = args.input_format or infer_format(input_path)
    output_format = args.output_format or infer_format(output_path)

    data = load_dataset(input_path, input_format)
    fields = args.fields or guess_text_fields(data)
    if not fields:
        raise ValueError("No text fields found. Pass --fields explicitly.")

    logger.info("Input: %s", input_path)
    logger.info("Output: %s", output_path)
    logger.info("Fields to translate: %s", fields)

    jobs = collect_jobs(data, fields)
    logger.info("Text cells to translate: %d", len(jobs))
    processor = None
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    translations: List[str] = []
    processed = 0
    for batch_jobs in tqdm(list(batched(jobs, args.batch_size)), desc="Translating batches"):
        prompts = make_prompts(processor, [j["text"] for j in batch_jobs], args.source_lang, args.target_lang)
        outputs = llm.generate(prompts, sampling)
        batch_translations = [clean_translation(out.outputs[0].text) for out in outputs]
        translations.extend(batch_translations)
        processed += len(batch_jobs)

        # Mutate data as we go so checkpoints contain progress.
        apply_translations(data, batch_jobs, batch_translations)
        if args.save_every and processed % args.save_every < len(batch_jobs):
            write_checkpoint(data, output_path, output_format, processed)

    save_dataset(data, output_path, output_format)
    logger.info("Done. Saved translated dataset with identical structure to: %s", output_path)


if __name__ == "__main__":
    main()
