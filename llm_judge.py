"""
LLM Judge for ReAct Benchmark
=============================
Loads the benchmark CSV (from react_rag_benchmark.py) containing predicted
answers, matches each row against ground-truth in agentic_rag_gt.json, and
scores:
  A. Retrieval Accuracy   — semantic Retrieval Precision, Retrieval Recall
  B. Generation Accuracy  — Faithfulness, Correctness

All judge scores are kept as raw 1-5 values.  Output is CSV only.

Usage:
  python llm_judge.py
  python llm_judge.py --benchmark-file path/to/raw.csv
  python llm_judge.py --qa-file path/to/agentic_rag_gt.json
  python llm_judge.py --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
MODELS_DIR = PROJECT_DIR / "models"
_LOCAL_MODEL_PATH = MODELS_DIR / "Qwen--Qwen3-14B"
MODEL_PATH = str(_LOCAL_MODEL_PATH) if _LOCAL_MODEL_PATH.exists() else "Qwen/Qwen3-14B"


# ---------------------------------------------------------------------------
# Judge model
# ---------------------------------------------------------------------------
_judge_model = None
_judge_tokenizer = None


def _get_judge(device: str = "cuda"):
    global _judge_model, _judge_tokenizer
    if _judge_model is None:
        print("Loading Qwen3-14B judge model...")
        dtype = torch.float16 if device == "cuda" else torch.float32
        _judge_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
        if _judge_tokenizer.pad_token is None:
            _judge_tokenizer.pad_token = _judge_tokenizer.eos_token
        _judge_model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            torch_dtype=dtype,
            device_map=device,
            attn_implementation="eager",
        )
        _judge_model.eval()
        print("Judge model ready.")
    return _judge_model, _judge_tokenizer


def _judge_generate(
    system_msg: str,
    user_msg: str,
    device: str = "cuda",
    max_new_tokens: int = 64,
) -> str:
    model, tokenizer = _get_judge(device)
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()


def _parse_score(response: str) -> float:
    """Parse a raw 1-5 score from the judge response."""
    match = re.search(r"\b([1-5])\b", response)
    if match:
        return float(match.group(1))
    return float("nan")


def _clip(text: str, limit: int = 6000) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def _safe_mean(values: list[float]) -> float:
    clean = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    return sum(clean) / len(clean) if clean else float("nan")


def _fmt(val: float) -> str:
    if isinstance(val, float) and math.isnan(val):
        return "N/A"
    return f"{val:.3f}"


DEFAULT_QA_FILE = DATASET_DIR / "agentic_rag_gt.json"


def _score_retrieval_precision(
    question: str,
    reference_context: str,
    reference_answer: str,
    retrieved_context: str,
    device: str,
) -> tuple[float, str]:
    if not retrieved_context:
        return float("nan"), ""

    response = _judge_generate(
        system_msg=(
            "You are an impartial evaluator for retrieval precision.\n"
            "Judge the RETRIEVED CONTENT by semantic meaning, not exact wording.\n"
            "Precision here means: how much of the retrieved content is relevant "
            "and useful for answering the question, compared with the reference meaning.\n\n"
            "Score on a scale of 1-5:\n"
            "  1 = Mostly irrelevant or misleading\n"
            "  2 = Limited relevance; mostly noisy or off-topic\n"
            "  3 = Mixed relevance; some useful content, substantial noise\n"
            "  4 = Mostly relevant with minor extra noise\n"
            "  5 = Highly relevant throughout and directly useful\n\n"
            "Respond with ONLY the score as a single integer (1-5)."
        ),
        user_msg=(
            f"QUESTION:\n{question}\n\n"
            f"REFERENCE CONTEXT:\n{_clip(reference_context, 2000)}\n\n"
            f"REFERENCE ANSWER:\n{_clip(reference_answer, 1200)}\n\n"
            f"RETRIEVED CONTENT:\n{_clip(retrieved_context)}"
        ),
        device=device,
    )
    return _parse_score(response), response


def _score_retrieval_recall(
    question: str,
    reference_context: str,
    reference_answer: str,
    retrieved_context: str,
    device: str,
) -> tuple[float, str]:
    if not retrieved_context:
        return float("nan"), ""

    response = _judge_generate(
        system_msg=(
            "You are an impartial evaluator for retrieval recall.\n"
            "Judge the RETRIEVED CONTENT by semantic meaning, not exact wording.\n"
            "Recall here means: how completely the retrieved content covers the key "
            "information needed to answer the question according to the reference meaning.\n\n"
            "Score on a scale of 1-5:\n"
            "  1 = Misses almost all needed information\n"
            "  2 = Covers only a small part of the needed information\n"
            "  3 = Covers the main idea but misses important details\n"
            "  4 = Covers most needed information with minor gaps\n"
            "  5 = Covers essentially all needed information\n\n"
            "Respond with ONLY the score as a single integer (1-5)."
        ),
        user_msg=(
            f"QUESTION:\n{question}\n\n"
            f"REFERENCE CONTEXT:\n{_clip(reference_context, 2000)}\n\n"
            f"REFERENCE ANSWER:\n{_clip(reference_answer, 1200)}\n\n"
            f"RETRIEVED CONTENT:\n{_clip(retrieved_context)}"
        ),
        device=device,
    )
    return _parse_score(response), response


def _score_faithfulness(
    predicted_answer: str,
    context: str,
    device: str,
) -> tuple[float, str]:
    if not context or not predicted_answer:
        return float("nan"), ""

    response = _judge_generate(
        system_msg=(
            "You are an impartial evaluator for answer faithfulness.\n"
            "Judge whether the ANSWER is supported by the CONTEXT.\n\n"
            "Score on a scale of 1-5:\n"
            "  1 = Completely unfaithful; contradicts or fabricates\n"
            "  2 = Mostly unfaithful; major unsupported claims\n"
            "  3 = Partially faithful; mixed support\n"
            "  4 = Mostly faithful; minor unsupported details\n"
            "  5 = Fully faithful; all key claims are supported\n\n"
            "Respond with ONLY the score as a single integer (1-5)."
        ),
        user_msg=(
            f"CONTEXT:\n{_clip(context)}\n\n"
            f"ANSWER:\n{_clip(predicted_answer, 2000)}"
        ),
        device=device,
    )
    return _parse_score(response), response


def _score_correctness(
    question: str,
    reference_answer: str,
    predicted_answer: str,
    device: str,
) -> tuple[float, str]:
    if not predicted_answer:
        return float("nan"), ""

    response = _judge_generate(
        system_msg=(
            "You are an impartial evaluator for answer correctness.\n"
            "Judge whether the PREDICTED ANSWER correctly answers the QUESTION, "
            "compared with the REFERENCE ANSWER. Use semantic meaning, not exact wording.\n\n"
            "Score on a scale of 1-5:\n"
            "  1 = Completely incorrect or irrelevant\n"
            "  2 = Mostly incorrect; misses the key point\n"
            "  3 = Partially correct; captures some important parts\n"
            "  4 = Mostly correct; minor inaccuracies or omissions\n"
            "  5 = Fully correct and complete\n\n"
            "Respond with ONLY the score as a single integer (1-5)."
        ),
        user_msg=(
            f"QUESTION:\n{question}\n\n"
            f"REFERENCE ANSWER:\n{_clip(reference_answer, 2000)}\n\n"
            f"PREDICTED ANSWER:\n{_clip(predicted_answer, 2000)}"
        ),
        device=device,
    )
    return _parse_score(response), response


def _load_ground_truth(qa_path: Path) -> dict[int, dict]:
    """Load ground-truth QA pairs and index by id."""
    qa_list = json.loads(qa_path.read_text())
    return {q["id"]: q for q in qa_list}


def _load_benchmark_csv(csv_path: Path) -> list[dict]:
    """Load benchmark raw CSV rows."""
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Judge pipeline — reads benchmark CSV + ground-truth JSON, outputs scored CSV
# ---------------------------------------------------------------------------

JUDGED_CSV_COLUMNS = [
    "question_id", "run", "question", "expected_tool",
    "reference_answer", "predicted_answer",
    "retrieval_precision", "retrieval_recall",
    "faithfulness", "correctness",
    "tool_correct", "tools_called",
    "pass_at_k", "pass_pow_k",
    "total_steps", "latency_ms",
    "retrieved_sources",
]


def judge_benchmark(
    benchmark_rows: list[dict],
    ground_truth: dict[int, dict],
    device: str = "cuda",
) -> list[dict]:
    """Score each benchmark row and return enriched rows."""
    scored_rows: list[dict] = []
    total = len(benchmark_rows)

    for idx, row in enumerate(benchmark_rows, 1):
        qid = int(row["question_id"])
        gt = ground_truth.get(qid, {})

        question = row.get("question", "")
        predicted_answer = row.get("predicted_answer", "")
        reference_answer = gt.get("reference_answer", row.get("reference_answer", ""))
        reference_context = gt.get("reference_context", "")
        expected_tool = row.get("expected_tool", "")
        actual_context = row.get("actual_context", "")

        print(f"[{idx}/{total}] Judging question_id={qid} run={row.get('run')}")

        # --- Retrieval scores (only for retrieve questions) ---
        if expected_tool == "retrieve" and actual_context:
            retrieval_precision, _ = _score_retrieval_precision(
                question=question,
                reference_context=reference_context,
                reference_answer=reference_answer,
                retrieved_context=actual_context,
                device=device,
            )
            retrieval_recall, _ = _score_retrieval_recall(
                question=question,
                reference_context=reference_context,
                reference_answer=reference_answer,
                retrieved_context=actual_context,
                device=device,
            )
        else:
            retrieval_precision = float("nan")
            retrieval_recall = float("nan")

        # --- Generation scores ---
        faithfulness, _ = _score_faithfulness(
            predicted_answer=predicted_answer,
            context=actual_context,
            device=device,
        )
        correctness, _ = _score_correctness(
            question=question,
            reference_answer=reference_answer,
            predicted_answer=predicted_answer,
            device=device,
        )

        scored_rows.append({
            "question_id": qid,
            "run": row.get("run", ""),
            "question": question,
            "expected_tool": expected_tool,
            "reference_answer": reference_answer,
            "predicted_answer": predicted_answer,
            "retrieval_precision": retrieval_precision,
            "retrieval_recall": retrieval_recall,
            "faithfulness": faithfulness,
            "correctness": correctness,
            "tool_correct": row.get("tool_correct", ""),
            "tools_called": row.get("tools_called", ""),
            "pass_at_k": row.get("pass_at_k", ""),
            "pass_pow_k": row.get("pass_pow_k", ""),
            "total_steps": row.get("total_steps", ""),
            "latency_ms": row.get("latency_ms", ""),
            "retrieved_sources": row.get("retrieved_sources", ""),
        })

    return scored_rows


def _compute_and_print_summary(rows: list[dict]):
    retrieve_rows = [r for r in rows if r["expected_tool"] == "retrieve"]
    r_precisions = [r["retrieval_precision"] for r in retrieve_rows]
    r_recalls = [r["retrieval_recall"] for r in retrieve_rows]
    faiths = [r["faithfulness"] for r in rows]
    corrects = [r["correctness"] for r in rows]

    print(f"\n{'='*60}")
    print("LLM JUDGE SUMMARY")
    print(f"{'='*60}")
    print(f"\n--- Retrieval Accuracy (raw 1-5, n={len(retrieve_rows)} runs) ---")
    print(f"  Precision:            {_fmt(_safe_mean(r_precisions))}")
    print(f"  Recall:               {_fmt(_safe_mean(r_recalls))}")
    print(f"\n--- Generation Accuracy (raw 1-5, n={len(rows)} runs) ---")
    print(f"  Faithfulness:         {_fmt(_safe_mean(faiths))}")
    print(f"  Correctness:          {_fmt(_safe_mean(corrects))}")
    print(f"{'='*60}\n")


def _default_benchmark_csv() -> Path | None:
    candidates = sorted(DATASET_DIR.glob("react_benchmark_raw_*.csv"))
    return candidates[-1] if candidates else None


def _default_output_path(benchmark_file: Path) -> Path:
    stem = benchmark_file.stem
    if stem.startswith("react_benchmark_raw_"):
        suffix = stem.removeprefix("react_benchmark_raw_")
        return benchmark_file.with_name(f"react_benchmark_judged_{suffix}.csv")
    return benchmark_file.with_name(f"{stem}_judged.csv")


def main():
    parser = argparse.ArgumentParser(description="LLM judge for ReAct benchmark results")
    parser.add_argument(
        "--benchmark-file",
        default=None,
        help="Path to the benchmark raw CSV from react_rag_benchmark.py",
    )
    parser.add_argument(
        "--qa-file",
        default=DEFAULT_QA_FILE,
        help=f"Path to ground-truth QA JSON (default: {DEFAULT_QA_FILE})",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional output path for the judged CSV file",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run the judge on (default: cuda)",
    )
    args = parser.parse_args()

    # --- Load benchmark CSV ---
    benchmark_file = Path(args.benchmark_file) if args.benchmark_file else _default_benchmark_csv()
    if benchmark_file is None or not benchmark_file.exists():
        print("Error: benchmark CSV not found.")
        print("Run react_rag_benchmark.py first or pass --benchmark-file.")
        return

    benchmark_rows = _load_benchmark_csv(benchmark_file)
    print(f"Loaded {len(benchmark_rows)} rows from {benchmark_file}")

    # --- Load ground-truth ---
    qa_path = Path(args.qa_file) if args.qa_file else DEFAULT_QA_FILE
    if not qa_path.exists():
        print(f"Error: ground-truth file not found at {qa_path}")
        return

    ground_truth = _load_ground_truth(qa_path)
    print(f"Loaded {len(ground_truth)} ground-truth QA pairs from {qa_path}")

    # --- Judge ---
    scored_rows = judge_benchmark(benchmark_rows, ground_truth, device=args.device)

    # --- Write output CSV ---
    output_csv = Path(args.output_csv) if args.output_csv else _default_output_path(benchmark_file)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JUDGED_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(scored_rows)

    print(f"\nJudged CSV saved to {output_csv}")
    _compute_and_print_summary(scored_rows)


if __name__ == "__main__":
    main()
