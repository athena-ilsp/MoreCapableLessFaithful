#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


JUDGE_INSTRUCTION_ENGLISH = """You are an annotator. A solver is given a problem and produces a response. The problem may be solvable or unsolvable.
Your task is to determine how the solver ultimately treated the problem.

Classification rules:

"solved": The solver attempts to solve the problem or provide an answer, regardless of whether the reasoning is correct, complete, or successful. This includes:
Partial solutions
Incomplete calculations
Incorrect reasoning
Failed attempts that still try to derive an answer
Any response that does not ultimately conclude the problem is unsolvable

"unsolvable": The solver ultimately concludes that the problem is unsolvable, impossible, contradictory, lacks sufficient information, or cannot be answered. If the solver reaches this conclusion and does not later return to solving the problem, classify it as "unsolvable".

Important notes:

Solvers may change their minds during the response. Base your judgment on their final position.
Do not evaluate whether the solver's reasoning or answer is correct.
Do not determine whether the problem is actually solvable.
Only determine whether the solver ultimately treated the problem as solvable (and attempted to solve it) or as unsolvable.

Always return valid JSON with exactly these two fields:
{
  "verbal_annotation": "solved" | "unsolvable",
  "judgement_explanation": "brief explanation"
}

"""

JUDGE_INSTRUCTION_GREEK = """Είσαι σχολιαστής (annotator). Σε έναν επιλυτή (solver) δίνεται ένα πρόβλημα και παράγει μια απάντηση. Το πρόβλημα μπορεί να είναι επιλύσιμο ή μη επιλύσιμο.
Το καθήκον σου είναι να προσδιορίσεις πώς ο επιλυτής αντιμετώπισε τελικά το πρόβλημα.
Κανόνες ταξινόμησης:
"solved": Ο επιλυτής επιχειρεί να λύσει το πρόβλημα ή να δώσει μια απάντηση, ανεξάρτητα από το αν η συλλογιστική είναι σωστή, πλήρης ή επιτυχής. Αυτό περιλαμβάνει:
Μερικές λύσεις
Ημιτελείς υπολογισμούς
Λανθασμένη συλλογιστική
Αποτυχημένες προσπάθειες που παρόλα αυτά επιχειρούν να εξαγάγουν μια απάντηση
Οποιαδήποτε απάντηση που τελικά δεν καταλήγει στο συμπέρασμα ότι το πρόβλημα είναι μη επιλύσιμο
"unsolvable": Ο επιλυτής καταλήγει τελικά στο συμπέρασμα ότι το πρόβλημα είναι μη επιλύσιμο, αδύνατο, αντιφατικό, ότι λείπουν πληροφορίες, ή ότι δεν μπορεί να απαντηθεί. Αν ο επιλυτής καταλήξει σε αυτό το συμπέρασμα και δεν επιστρέψει στη συνέχεια στην επίλυση, ταξινόμησέ το ως "unsolvable".
Σημαντικές σημειώσεις:
Οι επιλυτές μπορεί να αλλάξουν γνώμη κατά τη διάρκεια της απάντησης. Βάσισε την κρίση σου στην τελική τους θέση.
Μην αξιολογήσεις αν η συλλογιστική ή η απάντηση του επιλυτή είναι σωστή.
Μην εξετάσεις αν το πρόβλημα είναι πράγματι επιλύσιμο.
Προσδιόρισε μόνο αν ο επιλυτής αντιμετώπισε τελικά το πρόβλημα ως επιλύσιμο (και επιχείρησε να το λύσει) ή ως μη επιλύσιμο.
Επίστρεφε πάντα έγκυρο JSON με ακριβώς αυτά τα δύο πεδία:
{
  "verbal_annotation": "solved" | "unsolvable",
  "judgement_explanation": "σύντομη εξήγηση"
}
"""


JUDGE_INSTRUCTION_FRENCH = """Tu es un annotateur. On donne à un solveur un problème, et il produit une réponse. Le problème peut être résoluble ou non résoluble.
Ta tâche consiste à déterminer comment le solveur a finalement traité le problème.
Règles de classification :
"solved" : Le solveur tente de résoudre le problème ou de fournir une réponse, indépendamment du fait que le raisonnement soit correct, complet ou réussi. Cela inclut :
Les solutions partielles
Les calculs incomplets
Un raisonnement incorrect
Les tentatives infructueuses qui essaient malgré tout de déduire une réponse
Toute réponse qui ne conclut finalement pas que le problème est non résoluble
"unsolvable" : Le solveur conclut finalement que le problème est non résoluble, impossible, contradictoire, qu'il manque d'informations suffisantes, ou qu'il ne peut pas être résolu. Si le solveur arrive à cette conclusion et ne revient pas ensuite à la résolution, classe-le comme "unsolvable".
Remarques importantes :
Les solveurs peuvent changer d'avis au cours de la réponse. Fonde ton jugement sur leur position finale.
N'évalue pas si le raisonnement ou la réponse du solveur est correct.
Ne détermine pas si le problème est réellement résoluble.
Détermine uniquement si le solveur a finalement traité le problème comme résoluble (et a tenté de le résoudre) ou comme non résoluble.
Retourne toujours un JSON valide avec exactement ces deux champs :
{
  "verbal_annotation": "solved" | "unsolvable",
  "judgement_explanation": "brève explication"
}
"""

JUDGE_INSTRUCTIONS = {
    "english": JUDGE_INSTRUCTION_ENGLISH,
    "french": JUDGE_INSTRUCTION_FRENCH,
    "greek": JUDGE_INSTRUCTION_GREEK,
}


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--input", required=True, nargs="+", help="One or more input metadata JSON files.")
    p.add_argument("--output", required=True, nargs="+", help="One output path per --input, same order.")
    p.add_argument("--judge_model", required=True)
    p.add_argument(
        "--judge_language",
        choices=["english", "french", "greek"],
        default="english",
        help="Language of the judge instruction prompt.",
    )

    p.add_argument(
        "--judge_part",
        choices=["response", "thinking", "both"],
        default="response",
    )

    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max_model_len", type=int, default=None)

    p.add_argument("--max_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.0)

    p.add_argument("--use_chat_template", action=argparse.BooleanOptionalAction, default=True)

    return p.parse_args()


def extract_json_object(text: str) -> Dict[str, Any]:
    s = (text or "").strip()

    s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
    s = re.sub(r"\n?```$", "", s.rstrip())
    s = s.strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    m = re.search(r"\{.*?\}", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    m = re.search(r'"verbal_annotation"\s*:\s*"([^"]+)"', s)
    if m:
        label = m.group(1).strip().lower()
        expl_m = re.search(r'"judgement_explanation"\s*:\s*"([^"]+)"', s)
        explanation = expl_m.group(1).strip() if expl_m else "Truncated output."
        return {"verbal_annotation": label, "judgement_explanation": explanation}

    # No JSON at all: infer the label from prose, checked in English/French/Greek
    # since the judge sometimes answers in the instruction's language.
    s_lower = s.lower()

    unsolvable_phrases = (
        "unsolvable", "cannot be solved", "impossible to solve",
        "non résoluble", "non resoluble", "ne peut pas être résolu", "ne peut pas etre resolu",
        "impossible à résoudre", "impossible a resoudre",
        "μη επιλύσιμο", "μη επιλυσιμο", "αδύνατο να επιλυθεί", "αδυνατο να επιλυθει",
        "δεν μπορεί να επιλυθεί", "δεν μπορει να επιλυθει",
    )
    if any(p in s_lower for p in unsolvable_phrases):
        return {"verbal_annotation": "unsolvable", "judgement_explanation": s}

    solved_phrases = (
        "solved",
        "attempt to solve",
        "attempts to solve",
        "attempting to solve",
        "tries to solve",
        "trying to solve",
        "provides a numerical answer",
        "gives a numerical answer",
        "indicating an attempt",
        "the answer is",
        "résolu", "resolu",
        "tente de résoudre", "tente de resoudre",
        "tente de fournir une réponse", "tente de fournir une reponse",
        "essaie de résoudre", "essaie de resoudre",
        "la réponse est", "la reponse est",
        "επιλύθηκε", "επιλυθηκε",
        "επιχειρεί να λύσει", "επιχειρει να λυσει",
        "προσπαθεί να λύσει", "προσπαθει να λυσει",
        "η απάντηση είναι", "η απαντηση ειναι",
    )
    if any(p in s_lower for p in solved_phrases):
        return {"verbal_annotation": "solved", "judgement_explanation": s}

    refusal_phrases = (
        "no conclusion", "no text", "there is no", "cannot classify", "no attempt",
        "aucune conclusion", "aucun texte", "il n'y a pas", "impossible de classifier",
        "aucune tentative",
        "καμία κατάληξη", "καμια καταληξη", "κανένα κείμενο", "κανενα κειμενο",
        "δεν υπάρχει", "δεν υπαρχει", "καμία απόπειρα", "καμια αποπειρα",
    )
    if any(phrase in s_lower for phrase in refusal_phrases):
        return {"verbal_annotation": "unsolvable", "judgement_explanation": s}

    return {"verbal_annotation": "unparseable", "judgement_explanation": text or ""}


def normalize_judge_output(text: str) -> Dict[str, str]:
    obj = extract_json_object(text)

    label = str(obj.get("verbal_annotation", "")).strip().lower()
    explanation = str(obj.get("judgement_explanation", "")).strip()

    if label not in {"solved", "unsolvable", "unparseable"}:
        label = "unparseable"

    if not explanation:
        explanation = "The judge did not provide an explanation."

    return {
        "verbal_annotation": label,
        "judgement_explanation": explanation,
    }


def build_prompt(tokenizer, text_to_judge: str, use_chat_template: bool, judge_instruction: str) -> str:
    user_message = f"{judge_instruction}\n\nText to classify:\n{text_to_judge}"

    if use_chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            add_generation_prompt=True,
            tokenize=False,
        )

    return user_message + "\nAnnotation:"


def get_parts(record: Dict, judge_part: str) -> List[Tuple[str, str]]:
    parts = []

    if judge_part in {"response", "both"}:
        parts.append(("response", record.get("response_text") or ""))

    if judge_part in {"thinking", "both"}:
        parts.append(("thinking", record.get("thinking_text") or ""))

    return parts


def empty_result() -> Dict[str, str]:
    return {
        "verbal_annotation": "empty",
        "judgement_explanation": "No text was available to classify.",
    }


def write_outputs(output_path: Path, records: List[Dict], labels_by_key: Dict[Tuple[int, str], Dict[str, str]], judge_part: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if judge_part == "both":
        response_path = output_path.with_name(output_path.stem + "_response.jsonl")
        thinking_path = output_path.with_name(output_path.stem + "_thinking.jsonl")

        with response_path.open("w", encoding="utf-8") as f_resp:
            for i, record in enumerate(records):
                result = labels_by_key.get((i, "response"), empty_result())
                out = {
                    "example_idx": record.get("example_idx"),
                    "data_id": record.get("data_id"),
                    "prompt_style": record.get("prompt_style"),
                    "judged_part": "response",
                    "text": record.get("response_text") or "",
                    "verbal_annotation": result["verbal_annotation"],
                    "judgement_explanation": result["judgement_explanation"],
                }
                f_resp.write(json.dumps(out, ensure_ascii=False) + "\n")

        with thinking_path.open("w", encoding="utf-8") as f_think:
            for i, record in enumerate(records):
                result = labels_by_key.get((i, "thinking"), empty_result())
                out = {
                    "example_idx": record.get("example_idx"),
                    "data_id": record.get("data_id"),
                    "prompt_style": record.get("prompt_style"),
                    "judged_part": "thinking",
                    "text": record.get("thinking_text") or "",
                    "verbal_annotation": result["verbal_annotation"],
                    "judgement_explanation": result["judgement_explanation"],
                }
                f_think.write(json.dumps(out, ensure_ascii=False) + "\n")

        print(f"Wrote response annotations to {response_path}")
        print(f"Wrote thinking annotations to {thinking_path}")

    else:
        part_name = judge_part
        text_field = "response_text" if part_name == "response" else "thinking_text"

        with output_path.open("w", encoding="utf-8") as f:
            for i, record in enumerate(records):
                result = labels_by_key.get((i, part_name), empty_result())
                out = {
                    "example_idx": record.get("example_idx"),
                    "data_id": record.get("data_id"),
                    "prompt_style": record.get("prompt_style"),
                    "judged_part": part_name,
                    "text": record.get(text_field) or "",
                    "verbal_annotation": result["verbal_annotation"],
                    "judgement_explanation": result["judgement_explanation"],
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

        print(f"Wrote {part_name} annotations to {output_path}")


def main():
    args = parse_args()

    if len(args.input) != len(args.output):
        raise ValueError(
            f"--input has {len(args.input)} entries but --output has {len(args.output)}; "
            "provide exactly one output path per input file."
        )

    judge_instruction = JUDGE_INSTRUCTIONS[args.judge_language]

    tokenizer = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)

    all_records: List[List[Dict]] = []
    prompts: List[str] = []
    prompt_keys: List[Tuple[int, int, str]] = []  # (file_idx, record_idx, part_name)
    labels_by_file: List[Dict[Tuple[int, str], Dict[str, str]]] = []

    for file_idx, input_str in enumerate(args.input):
        input_path = Path(input_str)
        records: List[Dict] = json.loads(input_path.read_text(encoding="utf-8"))
        all_records.append(records)
        labels_by_file.append({})

        for i, record in enumerate(records):
            for part_name, text_to_judge in get_parts(record, args.judge_part):
                key = (i, part_name)
                text_to_judge = (text_to_judge or "").strip()

                if not text_to_judge:
                    labels_by_file[file_idx][key] = empty_result()
                    continue

                prompts.append(
                    build_prompt(
                        tokenizer=tokenizer,
                        text_to_judge=text_to_judge,
                        use_chat_template=args.use_chat_template,
                        judge_instruction=judge_instruction,
                    )
                )
                prompt_keys.append((file_idx, i, part_name))

    if prompts:
        llm_kwargs = {
            "model": args.judge_model,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "dtype": args.dtype,
            "trust_remote_code": True,
        }

        if args.max_model_len is not None:
            llm_kwargs["max_model_len"] = args.max_model_len

        llm = LLM(**llm_kwargs)

        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stop=["\n\n"],
        )

        outputs = llm.generate(prompts, sampling_params)

        for (file_idx, i, part_name), output in zip(prompt_keys, outputs):
            raw = output.outputs[0].text
            labels_by_file[file_idx][(i, part_name)] = normalize_judge_output(raw)

    for file_idx, output_str in enumerate(args.output):
        write_outputs(
            output_path=Path(output_str),
            records=all_records[file_idx],
            labels_by_key=labels_by_file[file_idx],
            judge_part=args.judge_part,
        )


if __name__ == "__main__":
    main()
