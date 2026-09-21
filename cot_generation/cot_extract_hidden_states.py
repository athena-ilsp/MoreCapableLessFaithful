#!/usr/bin/env python3
"""
Two-pass CoT generation + hidden state extraction over the ReliableMath
solvable/unsolvable splits (solve.parquet / unsol.parquet).

Pass 1 generates responses with vLLM; pass 2 runs the same prompts through HF
with teacher forcing to extract per-token hidden states. Model is selected via
--model (must match a subfolder in MODELS_ROOT). Output layout:
<output_dir>/<model_name>/<prompt_style>/<solvable|unsolvable>/, with one npz
per example (hidden states [n_tokens, hidden_dim] per layer) and a metadata
JSON (question, response, per-token solvability label).
"""

import os

if os.environ.get("HF_OFFLINE_OK", "1") == "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import re
import gc
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", HERE.parent / "models"))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", HERE.parent / "data" / "ReliableMath_parquet"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", HERE / "outputs"))

_EXCLUDED_PATTERNS = ["70b", "70B"]

MODEL_LAYERS: Dict[str, List[int]] = {
    # 32 transformer blocks, hidden_size=4096
    "llama_31_8b_instruct":        [1, 8, 16, 24, 32],
    # 32 transformer blocks, hidden_size=4096
    # "deepseek_r1_distill_llama8b": [1, 8, 16, 24, 32],
    # 36 transformer blocks, hidden_size=2560
    "qwen3_4b_instruct":           [1, 9, 18, 27, 36],
    # 48 transformer blocks, hidden_size=2048
    "qwen3_30b_instruct":          [1, 12, 24, 36, 48],
    # 60 transformer blocks, hidden_size=5376, multimodal (text decoder layers)
    "gemma-4-31B-it":              [1, 12, 24, 36, 48, 60],
    # 32 transformer blocks, hidden_size=4096 (Llama 3.1 8B base, Greek)
    "Llama-Krikri-8B-Instruct":    [1, 8, 16, 24, 32],
    # 32 transformer blocks, hidden_size=4096 (Llama 3 8B base, French)
    "French-Alpaca-Llama3-8B-Instruct-v1.0": [1, 8, 16, 24, 32],
    # 52 blocks, hidden_size=2688, hybrid Mamba/attention (nemotron_h)
    "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16":   [1, 13, 26, 39, 52],
}


def available_models() -> List[str]:
    return [
        d.name for d in sorted(MODELS_ROOT.iterdir())
        if d.is_dir() and not any(pat in d.name for pat in _EXCLUDED_PATTERNS)
    ]


SAMPLE_FRAC = 0.10
SAMPLE_SEED = 0

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args():
    models = available_models()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    # Model selection
    parser.add_argument(
        "--model", type=str, required=True, choices=models,
        help=f"Model name (subfolder of {MODELS_ROOT}). Available: {models}",
    )

    # Dataset
    parser.add_argument(
        "--solvability", type=str, required=True, nargs="+",
        choices=["solvable", "unsolvable"],
        help="One or more subsets to process in a SINGLE model load: solvable "
             "(solve*.parquet) and/or unsolvable (unsol*.parquet).",
    )
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Language BASE directory containing 'sol/' and 'unsol/' subfolders, "
             "each with a 'solve*.parquet' / 'unsol*.parquet' (e.g. "
             ".../ReliableMath_greek_claude). If the path itself contains the "
             "parquet directly, it is used as-is for all solvabilities. "
             "Defaults to the built-in DATA_ROOT if not set.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Base output directory. Per-combo outputs go to "
             "<output_dir>/<model>/<prompt_style>/<solvability>. "
             "Defaults to the built-in OUTPUT_ROOT if not set.",
    )

    parser.add_argument(
        "--phase", type=str, default="both", choices=["gen", "extract", "both"],
        help="gen: vLLM generate + cache only (no HF). extract: load cache + HF "
             "extraction only (no vLLM). both: legacy single-process (may OOM on "
             "large models). Recommended: run gen then extract as separate jobs/steps.",
    )

    # Processing
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layer indices to extract. Defaults to MODEL_LAYERS[model] if not set.")
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_thinking_tokens", type=int, default=None,
                        help="Cap on thinking tokens. Passed to vLLM SamplingParams if supported.")
    parser.add_argument("--n_tokens_per_trace", type=int, default=None,
                        help="If set, save exactly this many hidden-state vectors per trace "
                             "(thinking and response separately), evenly spaced by depth/position "
                             "instead of saving every token. Overrides --sample_frac behavior.")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_implementation", type=str, default="sdpa")

    # Generation
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--top_p", type=float, default=0.95)

    # Thinking model
    parser.add_argument("--thinking_model", action="store_true")
    parser.add_argument("--force_think_prefix", action="store_true",
                        help="Append '<think>\\n' after the chat generation prompt.")

    # Prompt style
    parser.add_argument("--use_chat_template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--prompt_style", type=str, default=["standard"], nargs="+",
        choices=[
            "standard",
            "standard_english",
            "standard_greek",
            "standard_french",
            "unsolvable_aware_multilingual",
            "unsolvable_aware_multilingual_english",
            "unsolvable_aware_multilingual_greek",
            "unsolvable_aware_multilingual_french",
        ],
        help=(
            "One or more prompt styles processed in a SINGLE model load.\n"
            "standard[_english/_greek/_french]:                 CoT only, answer in \\boxed{}.\n"
            "unsolvable_aware_multilingual[_english/_greek/_french]: solvable-or-unsolvable, "
            "answer in \\boxed{} or state unsolvable.\n"
            "The _english / _greek / _french suffixes give the same instructions in that language."
        ),
    )

    # vLLM
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_max_model_len", type=int, default=None)
    parser.add_argument("--vllm_enforce_eager", action="store_true",
                        help="Disable CUDA graph capture (eager mode). Needed for "
                             "gemma-4 under TP>1, where a worker dies during CUDA "
                             "graph capture causing RuntimeError: cancelled.")

    # Saving
    parser.add_argument("--save_float16", action="store_true",
                        help="Store hidden states as float16 instead of float32.")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Checkpoint metadata JSON every N examples.")

    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--num_examples", type=int, default=None)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_torch_dtype(dtype_str: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_str]

def sample_token_indices(n_tokens: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Pick a sorted, deduplicated subset of token positions (at least 1 if n_tokens > 0)."""
    if n_tokens <= 0:
        return np.empty(0, dtype=np.int64)
    k = min(n_tokens, max(1, int(round(n_tokens * frac))))
    idx = rng.choice(n_tokens, size=k, replace=False)
    idx.sort()  # keep chronological order
    return idx


def evenly_spaced_indices(n_tokens: int, n_target: int) -> np.ndarray:
    """Pick up to n_target token positions evenly spaced across [0, n_tokens).

    Deterministic (no RNG) — same trace length always yields the same depth
    coverage. Used to get a fixed-size, depth-uniform sample per trace
    regardless of trace length (short traces yield fewer than n_target points).
    """
    if n_tokens <= 0:
        return np.empty(0, dtype=np.int64)
    if n_tokens <= n_target:
        return np.arange(n_tokens, dtype=np.int64)
    idx = np.linspace(0, n_tokens - 1, num=n_target)
    idx = np.unique(np.round(idx).astype(np.int64))
    return idx

class ThinkLogitsProcessor:
    """Force </think> after a fixed number of thinking tokens.

    Works correctly with both:
      - force_think_prefix=True: <think> is in the prompt (not in past_tokens_ids),
        so we count all generated tokens until </think> appears.
      - force_think_prefix=False: <think> is generated by the model and appears
        in past_tokens_ids; we count tokens after the last <think>.

    vLLM passes only the *generated* tokens as past_tokens_ids, not the prompt.
    """

    def __init__(self, think_start_token: int, think_end_token: int,
                 num_think_tokens: int, force_think_prefix: bool = False):
        self.think_start_token = int(think_start_token)
        self.think_end_token = int(think_end_token)
        self.num_think_tokens = int(num_think_tokens)
        self.force_think_prefix = bool(force_think_prefix)

    def __call__(self, past_tokens_ids, logits: torch.Tensor) -> torch.Tensor:
        ids = past_tokens_ids.tolist() if hasattr(past_tokens_ids, "tolist") else list(past_tokens_ids)

        if self.think_end_token in ids:
            logits[self.think_start_token] = float("-inf")
            return logits

        if self.force_think_prefix:
            tokens_since_think = len(ids)
        else:
            if self.think_start_token not in ids:
                return logits
            think_start_pos = len(ids) - 1 - ids[::-1].index(self.think_start_token)
            tokens_since_think = len(ids) - think_start_pos - 1

        if tokens_since_think >= self.num_think_tokens:
            logits[:] = float("-inf")
            logits[self.think_end_token] = 0.0

        return logits

class ForceAfterThinkProcessor:
    """After </think>, force a fixed token sequence, e.g. '\n\nResponse:\n'."""

    def __init__(self, think_end_token: int, forced_ids: List[int], think_start_token: Optional[int] = None):
        self.think_end_token = int(think_end_token)
        self.forced_ids = list(map(int, forced_ids))
        self.think_start_token = int(think_start_token) if think_start_token is not None else None

    def __call__(self, past_tokens_ids, logits: torch.Tensor) -> torch.Tensor:
        ids = past_tokens_ids.tolist() if hasattr(past_tokens_ids, "tolist") else list(past_tokens_ids)

        if self.think_end_token in ids and self.think_start_token is not None:
            logits[self.think_start_token] = float("-inf")

        if self.think_end_token not in ids:
            return logits

        end_pos = len(ids) - 1 - ids[::-1].index(self.think_end_token)
        after_end = ids[end_pos + 1:]

        if len(after_end) >= len(self.forced_ids):
            return logits

        if after_end == self.forced_ids[:len(after_end)]:
            next_id = self.forced_ids[len(after_end)]
            logits[:] = float("-inf")
            logits[next_id] = 0.0

        return logits

def trim_generation(text: str) -> str:
    """Trim runaway model output after the final answer."""
    if not text:
        return text
    text = text.strip()

    # cut after the last \boxed{...}
    boxed_matches = list(re.finditer(r"\\boxed\{", text))
    if boxed_matches:
        m = boxed_matches[-1]
        i = m.end()
        depth = 1
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        return text[:i].strip()

    # else cut after answer/unsolvable/solvable sentence patterns
    answer_patterns = [
        r"(?is)(.*?the\s+answer\s+is\s+[^.\n]+\.)",
        r"(?is)(.*?the\s+problem\s+is\s+unsolvable\.)",
        r"(?is)(.*?the\s+problem\s+is\s+solvable\.)",
        r"(?is)(.*?the\s+problem\s+is\s+\{unsolvable\}\.)",
        r"(?is)(.*?the\s+problem\s+is\s+\{solvable\}\.)",
    ]
    for pat in answer_patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()

    # else cut at new-example continuation markers
    stop_patterns = [
        r"\n\s*Problem:\s*",
        r"\n\s*Question:\s*",
        r"\n\s*Q:\s*",
        r"\n\s*###\s*Problem",
        r"\n\s*User:\s*",
        r"\n\s*Assistant:\s*",
    ]
    cut = len(text)
    for pat in stop_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            cut = min(cut, m.start())
    return text[:cut].strip()


def parse_thinking_and_response(
    generated_text: str,
    thinking_model: bool,
    force_think_prefix: bool,
) -> Tuple[str, int, str, int, str]:
    """Return (thinking_text, thinking_offset, response_text, response_offset, parse_status)."""
    if not thinking_model:
        return "", 0, trim_generation(generated_text), 0, "non_thinking_model"

    open_tag, close_tag = "<think>", "</think>"
    open_idx = generated_text.find(open_tag)
    close_idx = generated_text.find(close_tag)

    if open_idx != -1 and close_idx != -1 and close_idx > open_idx:
        t_start = open_idx + len(open_tag)
        r_start = close_idx + len(close_tag)
        thinking_raw = generated_text[t_start:close_idx]
        response_raw = generated_text[r_start:]
        t_off = t_start + (len(thinking_raw) - len(thinking_raw.lstrip()))
        r_off = r_start + (len(response_raw) - len(response_raw.lstrip()))
        return thinking_raw.strip(), t_off, trim_generation(response_raw), r_off, "closed_think"

    if force_think_prefix and close_idx != -1:
        thinking_raw = generated_text[:close_idx]
        response_raw = generated_text[close_idx + len(close_tag):]
        t_off = len(thinking_raw) - len(thinking_raw.lstrip())
        r_off = close_idx + len(close_tag) + (len(response_raw) - len(response_raw.lstrip()))
        return thinking_raw.strip(), t_off, trim_generation(response_raw), r_off, "forced_prefix_closed_think"

    if force_think_prefix and close_idx == -1:
        logger.warning("Forced thinking prefix but no closing </think>; treating full output as thinking.")
        t_off = len(generated_text) - len(generated_text.lstrip())
        return generated_text.strip(), t_off, "", len(generated_text), "unclosed_think"

    logger.warning("thinking_model=True but no usable think tags. Treating output as response.")
    return "", 0, trim_generation(generated_text), 0, "no_think_tags_response"


def parse_edit_type(data_id: str) -> Optional[str]:
    """Extract edit type from data_id suffix, e.g. 'aime_9_remove_1' -> 'remove'."""
    if not data_id:
        return None
    m = re.search(r'_(remove|contradict)_\d+$', data_id)
    return m.group(1) if m else None


def find_response_token_start(
    tokenizer,
    generated_ids: List[int],
    force_think_prefix: bool,
) -> int:
    """Return the index into generated_ids where the response begins (after </think>)."""
    close_tag = "</think>"
    running = ""
    for i, tok_id in enumerate(generated_ids):
        running = tokenizer.decode(generated_ids[: i + 1], skip_special_tokens=False)
        idx = running.find(close_tag)
        if idx != -1:
            return i + 1
    return len(generated_ids)


# ---------------------------------------------------------------------------
# Answer extraction & correctness
# ---------------------------------------------------------------------------

def extract_final_answer(text: str) -> Optional[str]:
    if not text:
        return None
    s = trim_generation(text).strip()
    if "\\boxed{" in s:
        idx = s.rfind("\\boxed{")
        i = idx + len("\\boxed{")
        depth = 1
        buf = []
        while i < len(s) and depth > 0:
            if s[i] == "{":
                depth += 1
                buf.append(s[i])
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    break
                buf.append(s[i])
            else:
                buf.append(s[i])
            i += 1
        if buf:
            return "".join(buf).strip()
    m = re.search(r"(?i)(?:final\s+answer|answer)\s*(?:is|=|:)\s*([^\n\.]+)", s)
    if m:
        return m.group(1).strip().rstrip(".").strip()
    m = re.search(r"####\s*([^\n]+)", s)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"-?\d+(?:[\.,/]\d+)*", s)
    if nums:
        return nums[-1].replace(",", "")
    return None


def normalize_for_comparison(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower().replace("\\", "").replace("$", "")
    s = re.sub(r"\s+", "", s).rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except (ValueError, TypeError):
        return s


def check_correctness(predicted: Optional[str], ground_truth: Optional[str]) -> Optional[bool]:
    if predicted is None or ground_truth is None:
        return None
    p = normalize_for_comparison(predicted)
    g = normalize_for_comparison(ground_truth)
    if not p or not g:
        return None
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

_BOXED_ANSWER_FORMATS = (
    "- If the problem is solvable:   \\boxed{{<answer>}}\n"
    "- If the problem cannot be solved with the information provided: \\boxed{{unsolvable}}"
)

# Greek / French answer-format blocks. These are inserted into prompts that are
# NOT passed through str.format(), so braces are literal — write single braces
# and the literal token \boxed{...} that we want the model to emit.
_BOXED_ANSWER_FORMATS_GREEK = (
    "- Αν το πρόβλημα είναι επιλύσιμο:   \\boxed{<απάντηση>}\n"
    "- Αν το πρόβλημα δεν μπορεί να λυθεί με τις δοθείσες πληροφορίες: \\boxed{unsolvable}"
)

_BOXED_ANSWER_FORMATS_FRENCH = (
    "- Si le problème est résoluble :   \\boxed{<réponse>}\n"
    "- Si le problème ne peut pas être résolu avec les informations fournies : \\boxed{unsolvable}"
)

def _user_content(question: str, prompt_style: str, edit_type: Optional[str]) -> str:
    # English is the canonical (unsuffixed) style; "_english" is an alias so
    # the launcher can use a uniform "<base>_<lang>" naming across all langs.
    if prompt_style.endswith("_english"):
        prompt_style = prompt_style[: -len("_english")]

    if prompt_style == "standard":
        return (
            "Solve the following math problem step by step.\n"
            "Put your final answer within \\boxed{{}}.\n"
            f"Problem: {question}"
        )

    if prompt_style == "standard_greek":
        return (
            "Λύσε το παρακάτω μαθηματικό πρόβλημα βήμα προς βήμα.\n"
            "Γράψε την τελική σου απάντηση μέσα σε \\boxed{}.\n"
            f"Πρόβλημα: {question}"
        )

    if prompt_style == "standard_french":
        return (
            "Résous le problème de mathématiques suivant étape par étape.\n"
            "Indique ta réponse finale dans \\boxed{}.\n"
            f"Problème : {question}"
        )

    if prompt_style == "unsolvable_aware_multilingual":
        return (
            "Examine whether the following math problem is solvable with the information provided.\n"
            "If it is solvable, solve it step by step. Otherwise, state that it is unsolvable.\n"
            "Put your final answer in exactly one of these formats:\n"
            f"{_BOXED_ANSWER_FORMATS}\n"
            f"Problem: {question}"
        )

    if prompt_style == "unsolvable_aware_multilingual_greek":
        return (
            "Εξέτασε αν το παρακάτω μαθηματικό πρόβλημα είναι επιλύσιμο με τις πληροφορίες που δίνονται.\n"
            "Αν είναι επιλύσιμο, λύσε το βήμα προς βήμα. Διαφορετικά, δήλωσε ότι είναι μη επιλύσιμο.\n"
            "Γράψε την τελική σου απάντηση σε ακριβώς μία από αυτές τις μορφές:\n"
            f"{_BOXED_ANSWER_FORMATS_GREEK}\n"
            f"Πρόβλημα: {question}"
        )

    if prompt_style == "unsolvable_aware_multilingual_french":
        return (
            "Examine si le problème de mathématiques suivant est résoluble avec les informations fournies.\n"
            "S'il est résoluble, résous-le étape par étape. Sinon, indique qu'il est insoluble.\n"
            "Indique ta réponse finale dans exactement l'un de ces formats :\n"
            f"{_BOXED_ANSWER_FORMATS_FRENCH}\n"
            f"Problème : {question}"
        )

    raise ValueError(f"Unknown prompt_style: {prompt_style!r}")


def build_prompt(tokenizer, question: str, edit_type: Optional[str], args, prompt_style: str) -> str:
    user_content = _user_content(question, prompt_style, edit_type)

    if args.use_chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            add_generation_prompt=True,
            tokenize=False,
        )
    else:
        prompt = user_content + "\n"

    if args.thinking_model and args.force_think_prefix:
        prompt += "<think>\n"
    return prompt


# ---------------------------------------------------------------------------
# Pass 1: vLLM batched generation
# ---------------------------------------------------------------------------

def vllm_load(
    model_path: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    dtype: str,
    max_model_len: Optional[int] = None,
    enforce_eager: bool = False,
):
    """Construct and return a vLLM engine. Caller frees it via vllm_free().
    Kept separate from generation so one engine can serve many
    (solvability x prompt_style) combinations without reloading the model."""
    from vllm import LLM

    logger.info(f"Loading vLLM model: {model_path}")
    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        trust_remote_code=True,
    )
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    if enforce_eager:
        llm_kwargs["enforce_eager"] = True
    return LLM(**llm_kwargs)


def vllm_free(llm) -> None:
    """Release a vLLM engine and free GPU memory."""
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def vllm_generate(
    llm,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    top_p: float,
    max_thinking_tokens: Optional[int] = None,
    force_think_prefix: bool = False,
) -> List[List[int]]:
    """Run generation on an already-loaded vLLM engine. Does NOT free it."""
    from vllm import SamplingParams

    think_processors = None
    if max_thinking_tokens is not None:
        tok = llm.get_tokenizer()
        think_start_ids = tok.encode("<think>", add_special_tokens=False)
        think_end_ids = tok.encode("</think>", add_special_tokens=False)

        if len(think_start_ids) != 1 or len(think_end_ids) != 1:
            raise ValueError(
                "The logits-processor thinking cap requires <think> and </think> "
                f"to each be a single token, got <think>={think_start_ids}, "
                f"</think>={think_end_ids}."
            )
        response_prefix_ids = tok.encode("\n\nResponse:\n", add_special_tokens=False)
        think_processors = [
            ThinkLogitsProcessor(
                think_start_token=think_start_ids[0],
                think_end_token=think_end_ids[0],
                num_think_tokens=max_thinking_tokens,
                force_think_prefix=force_think_prefix,
            ),
            ForceAfterThinkProcessor(
                think_end_token=think_end_ids[0],
                forced_ids=response_prefix_ids,
                think_start_token=think_start_ids[0],
            ),
        ]
        logger.info(
            f"Using ThinkLogitsProcessor: max_thinking_tokens={max_thinking_tokens}, "
            f"force_think_prefix={force_think_prefix}, "
            f"<think> token={think_start_ids[0]}, </think> token={think_end_ids[0]}"
        )

    common_kwargs = dict(
        max_tokens=max_new_tokens,
        stop=["\nProblem:", "\nQuestion:", "\nQ:", "\n### Problem", "\nUser:", "\nAssistant:"],
    )
    if think_processors is not None:
        common_kwargs["logits_processors"] = think_processors

    if do_sample:
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, **common_kwargs)
    else:
        sampling_params = SamplingParams(temperature=0.0, **common_kwargs)

    logger.info(f"Generating for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)
    return [list(out.outputs[0].token_ids) for out in outputs]


# ---------------------------------------------------------------------------
# Pass 2: HF teacher-forced hidden state extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def hf_extract_hidden_states(
    model,
    tokenizer,
    prompt_text: str,
    generated_ids: List[int],
    layers: List[int],
    max_input_length: int,
) -> Tuple[str, Dict[int, torch.Tensor]]:
    """Return (generated_text, {layer: tensor[n_gen_tokens, hidden_dim]})."""
    prompt_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
        add_special_tokens=True,
    )["input_ids"][0].tolist()

    gen_token_start = len(prompt_ids)
    full_ids = prompt_ids + list(generated_ids)
    full_tensor = torch.tensor([full_ids], device=model.device, dtype=torch.long)
    attention_mask = torch.ones_like(full_tensor)

    outputs = model(
        input_ids=full_tensor,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )

    per_layer: Dict[int, torch.Tensor] = {}
    for layer in layers:
        if layer < 0 or layer >= len(outputs.hidden_states):
            raise ValueError(
                f"Invalid layer {layer}; model has {len(outputs.hidden_states)} hidden state tensors."
            )
        per_layer[layer] = outputs.hidden_states[layer][0, gen_token_start:, :].detach().cpu()

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, per_layer


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_reliablemath(solvability: str, data_root: Path) -> Any:
    """Load the solvable/unsolvable parquet from data_root and return a list of dicts.

    Accepts either the canonical names (solve.parquet / unsol.parquet) or the
    multilingual variants (solve_greek.parquet, unsol_french.parquet, ...).
    """
    import pandas as pd

    prefix = "solve" if solvability == "solvable" else "unsol"

    exact = data_root / f"{prefix}.parquet"
    if exact.exists():
        path = exact
    else:
        candidates = sorted(
            p for p in data_root.glob(f"{prefix}*.parquet")
            if ".checkpoint" not in p.name
        )
        if not candidates:
            raise FileNotFoundError(
                f"No '{prefix}*.parquet' found in {data_root}. "
                f"Expected e.g. {prefix}.parquet or {prefix}_<lang>.parquet."
            )
        if len(candidates) > 1:
            logger.warning(f"Multiple {prefix} parquets in {data_root}: "
                           f"{[c.name for c in candidates]}; using {candidates[0].name}")
        path = candidates[0]

    logger.info(f"Loading {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} examples from {path.name}")
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def save_metadata_checkpoint(records: List[Dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info(f"Checkpoint metadata -> {path} ({len(records)} records)")


# ---------------------------------------------------------------------------
# Pass-1 generation cache
# ---------------------------------------------------------------------------
# Generations live only in RAM otherwise; if Pass 2 (HF) crashes, the
# expensive vLLM generation would be lost. We persist token-ids per combo to
# disk right after generating, and reload them on restart so a crashed job
# resumes from Pass 2 without regenerating.

def gen_cache_path(out_dir: Path) -> Path:
    return out_dir / "gen_cache.json"


def save_gen_cache(out_dir: Path, all_data: List[Dict], generated_ids: List[List[int]],
                   start_idx: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "start_idx": start_idx,
        "n": len(all_data),
        "example_idx": [d["example_idx"] for d in all_data],
        "generated_ids": generated_ids,
    }
    tmp = gen_cache_path(out_dir).with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f)
    tmp.replace(gen_cache_path(out_dir))  # atomic, so a partial write never looks valid
    logger.info(f"Cached {len(generated_ids)} generations -> {gen_cache_path(out_dir)}")


def load_gen_cache(out_dir: Path, all_data: List[Dict], start_idx: int):
    """Return cached generated_ids if a valid cache matches this combo's
    examples, else None."""
    path = gen_cache_path(out_dir)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning(f"Ignoring unreadable gen cache {path}")
        return None
    expected_idx = [d["example_idx"] for d in all_data]
    if (payload.get("start_idx") != start_idx
            or payload.get("n") != len(all_data)
            or payload.get("example_idx") != expected_idx):
        logger.warning(f"Gen cache {path} does not match current example set; ignoring.")
        return None
    gen = payload.get("generated_ids")
    if not isinstance(gen, list) or len(gen) != len(all_data):
        logger.warning(f"Gen cache {path} malformed; ignoring.")
        return None
    logger.info(f"Loaded {len(gen)} cached generations <- {path} (skipping regeneration)")
    return gen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_input_dir(data_root: Path, solvability: str) -> Path:
    """Return the directory that actually holds the parquet for this solvability.

    Supports both a language BASE dir with sol/ + unsol/ subfolders, and a dir
    that already contains the parquet directly.
    """
    sub = "sol" if solvability == "solvable" else "unsol"
    if (data_root / sub).is_dir():
        return data_root / sub
    return data_root


def _build_examples(tokenizer, args, solvability: str, prompt_style: str, data_root: Path):
    """Load dataset for (solvability) and build per-example dicts with prompts
    for the given prompt_style. Returns the (sliced) list of example dicts."""
    is_unsolvable = solvability == "unsolvable"
    input_dir = _resolve_input_dir(data_root, solvability)
    all_records = load_reliablemath(solvability, input_dir)

    end_idx = args.end_idx if args.end_idx is not None else len(all_records)
    if args.num_examples:
        end_idx = min(end_idx, args.start_idx + args.num_examples)
    all_records = all_records[args.start_idx:end_idx]
    logger.info(f"[{solvability}/{prompt_style}] examples "
                f"{args.start_idx}..{args.start_idx + len(all_records) - 1} "
                f"({len(all_records)} total) from {input_dir}")

    all_data = []
    for local_idx, rec in enumerate(all_records):
        example_idx = args.start_idx + local_idx
        if is_unsolvable:
            question = rec.get("rewritten_question", rec.get("question", ""))
            original_question = rec.get("question", "")
            edit_type = parse_edit_type(rec.get("data_id", ""))
        else:
            question = rec.get("question", rec.get("problem", ""))
            original_question = question
            edit_type = None

        ground_truth_raw = rec.get("ground_truth", rec.get("answer", ""))
        ground_truth_clean = extract_final_answer(str(ground_truth_raw)) if ground_truth_raw else None
        if ground_truth_clean is None and ground_truth_raw:
            ground_truth_clean = str(ground_truth_raw).strip()

        prompt = build_prompt(tokenizer, question, edit_type, args, prompt_style)
        all_data.append({
            "example_idx": example_idx,
            "question": question,
            "original_question": original_question,
            "ground_truth_raw": str(ground_truth_raw) if ground_truth_raw is not None else None,
            "ground_truth_clean": ground_truth_clean,
            "prompt": prompt,
            "solvability_label": solvability,
            "edit_type": edit_type,
            "data_id": rec.get("data_id", None),
        })
    return all_data


def _extract_and_save(model, tokenizer, args, solvability, prompt_style,
                      all_data, all_generated_ids, out_dir, out_dtype, data_root, model_path):
    """Pass-2 extraction + per-example save for one (solvability, prompt_style)
    combo. Logic is identical to the single-combo path of the original script."""
    is_unsolvable = solvability == "unsolvable"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / f"metadata_{args.start_idx}_{args.start_idx + len(all_data)}.json"
    problems_metadata: List[Dict] = []

    for i, item in enumerate(tqdm(all_data, desc=f"Pass 2 [{solvability}/{prompt_style}]")):
        example_idx = item["example_idx"]
        generated_ids = all_generated_ids[i]

        if not generated_ids:
            logger.warning(f"Empty generation for example {example_idx}, skipping.")
            continue

        try:
            generated_text, per_layer_hiddens = hf_extract_hidden_states(
                model=model,
                tokenizer=tokenizer,
                prompt_text=item["prompt"],
                generated_ids=generated_ids,
                layers=args.layers,
                max_input_length=args.max_input_length,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM at example {example_idx}, skipping.")
                torch.cuda.empty_cache()
                continue
            raise

        thinking_text, _t_off, response_text, _r_off, parse_status = parse_thinking_and_response(
            generated_text,
            thinking_model=args.thinking_model,
            force_think_prefix=args.force_think_prefix,
        )

        if args.thinking_model and not response_text:
            logger.warning(
                f"Example {example_idx}: empty response after thinking "
                f"({len(generated_ids)} tokens generated). "
                f"Try increasing --max_new_tokens or setting --max_thinking_tokens."
            )
            continue

        if args.thinking_model:
            response_tok_start = find_response_token_start(
                tokenizer, generated_ids, args.force_think_prefix
            )
        else:
            response_tok_start = 0

        n_tokens = len(generated_ids)
        n_thinking_tokens = response_tok_start
        n_response_tokens = n_tokens - response_tok_start

        if is_unsolvable:
            predicted_answer = None
            correctness_label = None
        else:
            answer_source = response_text if response_text else generated_text
            predicted_answer = extract_final_answer(answer_source)
            correctness_label = check_correctness(predicted_answer, item["ground_truth_clean"])

        sol_value = np.int8(1 if is_unsolvable else 0)
        example_stem = f"ex{example_idx:06d}"

        if args.n_tokens_per_trace is not None:
            if args.thinking_model:
                think_sel = evenly_spaced_indices(n_thinking_tokens, args.n_tokens_per_trace)
                resp_sel = evenly_spaced_indices(n_response_tokens, args.n_tokens_per_trace)
                full_sel = None
            else:
                think_sel = resp_sel = None
                full_sel = evenly_spaced_indices(n_tokens, args.n_tokens_per_trace)
        else:
            rng = np.random.default_rng(SAMPLE_SEED + example_idx)
            if args.thinking_model:
                think_sel = sample_token_indices(n_thinking_tokens, SAMPLE_FRAC, rng)
                resp_sel = sample_token_indices(n_response_tokens, SAMPLE_FRAC, rng)
                full_sel = None
            else:
                think_sel = resp_sel = None
                full_sel = sample_token_indices(n_tokens, SAMPLE_FRAC, rng)

        for layer in args.layers:
            h = per_layer_hiddens[layer]
            if args.thinking_model:
                if think_sel.size > 0:
                    h_thinking = h[:response_tok_start][think_sel].float().numpy().astype(out_dtype)
                    np.savez(
                        str(out_dir / f"{example_stem}_thinking_layer{layer}.npz"),
                        hidden_states=h_thinking,
                        token_solvability=np.full(think_sel.size, sol_value, dtype=np.int8),
                        token_positions=think_sel.astype(np.int32),
                    )
                if resp_sel.size > 0:
                    h_response = h[response_tok_start:][resp_sel].float().numpy().astype(out_dtype)
                    np.savez(
                        str(out_dir / f"{example_stem}_response_layer{layer}.npz"),
                        hidden_states=h_response,
                        token_solvability=np.full(resp_sel.size, sol_value, dtype=np.int8),
                        token_positions=(resp_sel + response_tok_start).astype(np.int32),
                    )
            else:
                if full_sel.size > 0:
                    np.savez(
                        str(out_dir / f"{example_stem}_layer{layer}.npz"),
                        hidden_states=h[full_sel].float().numpy().astype(out_dtype),
                        token_solvability=np.full(full_sel.size, sol_value, dtype=np.int8),
                        token_positions=full_sel.astype(np.int32),
                    )

        meta = {
            "example_idx": example_idx,
            "solvability_label": solvability,
            "question": item["question"],
            "original_question": item["original_question"],
            "ground_truth_raw": item["ground_truth_raw"],
            "ground_truth_clean": item["ground_truth_clean"],
            "predicted_answer": predicted_answer,
            "correctness_label": correctness_label,
            "full_generated_text": generated_text,
            "thinking_text": thinking_text,
            "response_text": response_text,
            "has_thinking": bool(thinking_text),
            "has_response": bool(response_text),
            "thinking_parse_status": parse_status,
            "closed_think_tag": "</think>" in generated_text,
            "n_generated_tokens": n_tokens,
            "n_thinking_tokens": n_thinking_tokens,
            "n_response_tokens": n_response_tokens,
            "response_token_start": response_tok_start,
            "layers_saved": args.layers,
            "token_solvability_value": int(sol_value),
            "edit_type": item["edit_type"],
            "data_id": item["data_id"],
            "prompt_style": prompt_style,
        }
        problems_metadata.append(meta)

        if (i + 1) % args.save_every == 0:
            save_metadata_checkpoint(problems_metadata, meta_path)

    save_metadata_checkpoint(problems_metadata, meta_path)

    config_path = out_dir / "run_config.json"
    cfg = vars(args).copy()
    cfg["solvability"] = solvability
    cfg["prompt_style"] = prompt_style
    cfg["model_path"] = str(model_path)
    cfg["data_root"] = str(data_root)
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info(f"Config saved -> {config_path}")

    n_correct = sum(1 for p in problems_metadata if p["correctness_label"] is True)
    n_incorrect = sum(1 for p in problems_metadata if p["correctness_label"] is False)
    n_unscored = sum(1 for p in problems_metadata if p["correctness_label"] is None)
    logger.info(f"=== Done [{solvability}/{prompt_style}] === "
                f"examples={len(problems_metadata)} correct={n_correct} "
                f"wrong={n_incorrect} unscored={n_unscored} -> {out_dir}")


def main():
    args = parse_args()

    model_path = MODELS_ROOT / args.model
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    if args.layers is None:
        if args.model not in MODEL_LAYERS:
            raise ValueError(
                f"No default layers defined for '{args.model}' in MODEL_LAYERS. "
                f"Pass --layers explicitly or add an entry to MODEL_LAYERS."
            )
        args.layers = MODEL_LAYERS[args.model]
        logger.info(f"Using default layers for {args.model}: {args.layers}")

    solvabilities = list(dict.fromkeys(args.solvability))
    prompt_styles = list(dict.fromkeys(args.prompt_style))
    combos = [(s, p) for s in solvabilities for p in prompt_styles]
    logger.info(f"Combos in this load: {combos}")

    data_root = Path(args.input_dir) if args.input_dir else DATA_ROOT
    output_root = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT
    if not data_root.exists():
        raise FileNotFoundError(f"Input directory not found: {data_root}")
    logger.info(f"Input directory (base): {data_root}")

    torch_dtype = get_torch_dtype(args.dtype)
    out_dtype = np.float16 if args.save_float16 else np.float32

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    combo_data = {combo: _build_examples(tokenizer, args, combo[0], combo[1], data_root)
                  for combo in combos}

    combo_out_dir = {combo: output_root / args.model / combo[1] / combo[0] for combo in combos}

    # =======================================================================
    # PHASE: gen  -- vLLM generate + cache only. No HF load in this process.
    # =======================================================================
    if args.phase in ("gen", "both"):
        logger.info("=== Phase 'gen': vLLM generation (single model load) ===")
        todo = []
        for combo in combos:
            cached = load_gen_cache(combo_out_dir[combo], combo_data[combo], args.start_idx)
            if cached is not None:
                logger.info(f"Combo {combo} already cached; skipping generation.")
            else:
                todo.append(combo)

        if todo:
            llm = vllm_load(
                model_path=str(model_path),
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                dtype=args.dtype,
                max_model_len=args.vllm_max_model_len,
                enforce_eager=args.vllm_enforce_eager,
            )
            try:
                for combo in todo:
                    all_data = combo_data[combo]
                    logger.info(f"Generating combo {combo} ({len(all_data)} prompts)...")
                    gen = vllm_generate(
                        llm,
                        prompts=[d["prompt"] for d in all_data],
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        do_sample=args.do_sample,
                        top_p=args.top_p,
                        max_thinking_tokens=args.max_thinking_tokens,
                        force_think_prefix=args.force_think_prefix,
                    )
                    assert len(gen) == len(all_data)
                    save_gen_cache(combo_out_dir[combo], all_data, gen, args.start_idx)
            finally:
                vllm_free(llm)
        else:
            logger.info("All combos already cached; nothing to generate.")

        if args.phase == "gen":
            logger.info("Phase 'gen' complete; generations cached. Run --phase extract next.")
            return

    # =======================================================================
    # PHASE: extract  -- load cache + HF, extract hidden states, save.
    # Runs in a FRESH process (when phase==extract) so no vLLM memory lingers.
    # =======================================================================
    logger.info("=== Phase 'extract': HF hidden state extraction (single model load) ===")
    combo_generated = {}
    missing = []
    for combo in combos:
        cached = load_gen_cache(combo_out_dir[combo], combo_data[combo], args.start_idx)
        if cached is None:
            missing.append(combo)
        else:
            combo_generated[combo] = cached
    if missing:
        raise FileNotFoundError(
            f"Phase 'extract' needs cached generations but these combos are missing "
            f"a valid gen_cache.json: {missing}. Run --phase gen first."
        )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    model.eval()

    num_model_layers = getattr(model.config, "num_hidden_layers", None)
    logger.info(f"Model has {num_model_layers} transformer layers. Extracting layers: {args.layers}")
    if num_model_layers is not None:
        invalid = [l for l in args.layers if l < 0 or l > num_model_layers]
        if invalid:
            raise ValueError(
                f"Invalid layers {invalid}. Valid range: 0..{num_model_layers} "
                f"(0 = embedding output, {num_model_layers} = after final transformer block)."
            )

    for combo in combos:
        solvability, prompt_style = combo
        _extract_and_save(
            model, tokenizer, args, solvability, prompt_style,
            combo_data[combo], combo_generated[combo],
            combo_out_dir[combo], out_dtype, data_root, model_path,
        )

    logger.info("\n=== All combos complete ===")
    logger.info(f"Model:  {args.model}")
    logger.info(f"Combos: {combos}")


if __name__ == "__main__":
    main()
