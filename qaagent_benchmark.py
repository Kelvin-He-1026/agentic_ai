"""
QAAgent Benchmark — Retrieval + Answer Quality Scoring
=======================================================
Benchmarks a QAAgent-style retrieve/validate/retrieve/answer pipeline
against a ground-truth dataset.

Models (all local, no API keys needed for LLM):
  Agent LLM : Qwen3-8B  (validation + answering)  — loaded via model_loader.py
  Judge LLM : Qwen3-14B (scoring)                  — loaded via model_loader.py
  Retrieval : Tavily search (requires TAVILY_API_KEY env var)

Metrics scored:
  1. Retrieval Recall@5       — gold source found in top-5 retrieved results
  2. Retrieval Context Recall — LLM-judged (1-5): coverage of needed info
  3. Retrieval Context Precision — LLM-judged (1-5): relevance of retrieved content
  4. Answer Truthfulness      — LLM-judged (1-5): correctness vs gold answer
  5. Answer Faithfulness      — LLM-judged (1-5): grounded in retrieved context

Supplementary lexical metrics:
  - Token-level F1, ROUGE-L, exact match
  - Lexical groundedness (token overlap proxy)

Agent source: github.com/astordu/r1-reasoning-rag/tree/main/src
  QAAgent flow: retrieve (Tavily) -> validate (LLM) -> [find_missing] -> answer (LLM)

Usage:
  # Sanity check with gold oracle (no external calls)
  python qaagent_benchmark.py --gold-sanity --no-judge

  # Run with local Qwen3-8B agent + Qwen3-14B judge (requires TAVILY_API_KEY)
  python qaagent_benchmark.py --agent-spec qaagent

  # Use a different local agent model
  python qaagent_benchmark.py --agent-spec qaagent --agent-model qwen3-14b

  # Skip LLM judge (lexical metrics only)
  python qaagent_benchmark.py --gold-sanity --no-judge

  # Save full results
  python qaagent_benchmark.py --agent-spec qaagent --output results.json
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import torch

# Allow importing model_loader from langGraph/
sys.path.insert(0, str(Path(__file__).resolve().parent / "langGraph"))
from model_loader import load_llm, load_embedding, MODELS_DIR


# ============================================================
# Paths
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
DEFAULT_DATASET = DATASET_DIR / "qaagent_ground_truth.json"

# Default model presets (local, via model_loader)
DEFAULT_AGENT_MODEL = "qwen3-8b"   # for validation + answering
DEFAULT_JUDGE_MODEL = "qwen3-14b"  # for scoring
DEFAULT_EMBED_MODEL = "bge-large"  # for semantic similarity


# ============================================================
# Data model
# ============================================================

@dataclass
class GroundTruthItem:
    id: int
    question: str
    tool: str
    context_reference: dict[str, Any]
    answer_reference: dict[str, Any]
    # Optional fields for richer QAAgent-style benchmarking
    round1_status: str | None = None
    gold_missing_information: str | None = None
    gold_followup_query: str | None = None
    gold_useful_information: str | None = None
    round2_status: str | None = None
    question_type: str | None = None
    difficulty: str | None = None

    @property
    def gold_answer(self) -> str:
        return self.answer_reference.get("answer", "")

    @property
    def gold_context(self) -> str:
        return self.context_reference.get("context", "")

    @property
    def gold_title(self) -> str:
        return self.context_reference.get("metadata", {}).get("title", "")

    @property
    def gold_file_name(self) -> str:
        return self.context_reference.get("metadata", {}).get("file_name", "")


@dataclass
class RetrievalResult:
    retrieved_context: str
    retrieved_sources: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    status: str
    missing_information: str = ""
    useful_information: str = ""
    reasoning: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerResult:
    answer: str
    reasoning: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationStats:
    """Token counts and timing from a single LLM generation call."""
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0       # tokens inside <think>...</think>
    response_tokens: int = 0       # tokens after </think> (the actual answer)
    total_time_ms: float = 0.0     # wall time for full generate()
    think_time_ms: float = 0.0     # estimated: total * (thinking_tokens / output_tokens)
    generate_time_ms: float = 0.0  # estimated: total * (response_tokens / output_tokens)


@dataclass
class StepTrace:
    """Detailed trace for one step of the QAAgent pipeline."""
    step_name: str                          # e.g. "retrieve", "validate", "find_missing", "answer"
    step_number: int
    wall_time_ms: float                     # total step wall time
    # LLM generation stats (None for retrieval-only steps)
    generation: GenerationStats | None = None
    # Input/output content
    input_text: str = ""                    # what was sent to the LLM / search
    output_text: str = ""                   # raw LLM output / retrieved context
    thinking_text: str = ""                 # content of <think> block (if any)
    # Retrieval-specific
    retrieved_context: str = ""
    retrieved_sources: list[str] = field(default_factory=list)


@dataclass
class ItemResult:
    id: int
    question: str
    rounds_used: int
    # --- Step-by-step traces ---
    step_traces: list[dict] = field(default_factory=list)
    # --- Retrieval Recall@5 ---
    retrieval_recall_at_5: bool = False
    # --- Semantic retrieval metrics (cosine similarity via embeddings) ---
    retrieval_context_similarity: float = 0.0  # cos_sim(retrieved_ctx, gold_ctx)
    retrieval_source_match: bool = False
    # --- LLM-judged retrieval metrics (1-5 scale, NaN if skipped) ---
    retrieval_context_recall_judge: float = float("nan")
    retrieval_context_precision_judge: float = float("nan")
    # --- LLM-judged answer metrics (1-5 scale, NaN if skipped) ---
    answer_truthfulness_judge: float = float("nan")
    answer_faithfulness_judge: float = float("nan")
    # --- Semantic answer metrics (cosine similarity via embeddings) ---
    answer_similarity: float = 0.0       # cos_sim(predicted_answer, gold_answer)
    answer_groundedness: float = 0.0     # cos_sim(predicted_answer, retrieved_context)
    # --- Validator tracking ---
    validator_scored: bool = False
    validator_correct: bool | None = None
    predicted_round1_status: str | None = None
    gold_round1_status: str | None = None
    followup_scored: bool = False
    followup_missing_info_similarity: float | None = None
    # --- Raw text ---
    predicted_answer: str = ""
    gold_answer: str = ""
    retrieved_context: str = ""
    retrieved_sources: list[str] = field(default_factory=list)


class QAFlowAgent(Protocol):
    """Minimal agent interface matching the QAAgent flow.

    Every method returns (result, StepTrace) so the benchmark runner can
    capture per-step token counts, timing, and context.
    """

    def retrieve(self, question: str) -> tuple[RetrievalResult, StepTrace]: ...

    def validate_retrieval(
        self, question: str, retrieved_context: str,
    ) -> tuple[ValidationResult, StepTrace]: ...

    def find_missing_information(
        self, question: str, missing_information: str, useful_information: str,
    ) -> tuple[RetrievalResult, StepTrace]: ...

    def answer(
        self, question: str, retrieved_context: str,
    ) -> tuple[AnswerResult, StepTrace]: ...


# ============================================================
# Shared LLM generation helper
# ============================================================

_loaded_models: dict[str, tuple] = {}  # cache: preset_name -> (model, tokenizer)


def _get_model(preset: str, device: str = "cuda") -> tuple:
    """Load a model via model_loader (cached across calls)."""
    key = f"{preset}@{device}"
    if key not in _loaded_models:
        dtype = torch.float16 if "cuda" in device else torch.float32
        model, tokenizer = load_llm(preset, device=device, torch_dtype=dtype)
        _loaded_models[key] = (model, tokenizer)
    return _loaded_models[key]


def _generate(
    model_preset: str,
    messages: list[dict[str, str]],
    device: str = "cuda",
    max_new_tokens: int = 256,
    enable_thinking: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> tuple[str, GenerationStats]:
    """Generate text from a local model. Returns (text, stats)."""
    model, tokenizer = _get_model(model_preset, device)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_token_count = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    total_ms = (time.perf_counter() - t0) * 1000.0

    new_tokens = output[0][input_token_count:]
    output_token_count = len(new_tokens)

    # Decode with special tokens to find <think> boundaries for token counting
    raw_decoded = tokenizer.decode(new_tokens, skip_special_tokens=False)
    result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Count thinking vs response tokens
    thinking_tokens = 0
    response_tokens = output_token_count
    if "<think>" in raw_decoded and "</think>" in raw_decoded:
        think_end_text = raw_decoded.split("</think>")[0] + "</think>"
        thinking_tokens = len(tokenizer.encode(think_end_text, add_special_tokens=False))
        response_tokens = max(0, output_token_count - thinking_tokens)

    # Estimate think vs generate time proportionally
    if output_token_count > 0:
        think_ms = total_ms * (thinking_tokens / output_token_count)
        gen_ms = total_ms * (response_tokens / output_token_count)
    else:
        think_ms = 0.0
        gen_ms = total_ms

    stats = GenerationStats(
        input_tokens=input_token_count,
        output_tokens=output_token_count,
        thinking_tokens=thinking_tokens,
        response_tokens=response_tokens,
        total_time_ms=total_ms,
        think_time_ms=think_ms,
        generate_time_ms=gen_ms,
    )
    return result, stats


def _strip_think_tags(text: str) -> tuple[str, str]:
    """Strip <think>...</think> reasoning tags (Qwen3 / DeepSeek R1 style)."""
    if "<think>" in text and "</think>" in text:
        reasoning = text.split("<think>")[1].split("</think>")[0].strip()
        response = text.split("</think>")[1].strip()
        return response, reasoning
    return text.strip(), ""


# ============================================================
# LLM Judge (Qwen3-14B via model_loader)
# ============================================================

def _judge_generate(
    system_msg: str,
    user_msg: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    device: str = "cuda",
    max_new_tokens: int = 64,
) -> str:
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    result, _stats = _generate(
        judge_model, messages, device=device,
        max_new_tokens=max_new_tokens, enable_thinking=False,
    )
    # Strip any thinking tags that leak through
    result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
    return result


def _parse_score(response: str) -> float:
    match = re.search(r"\b([1-5])\b", response)
    return float(match.group(1)) if match else float("nan")


def _clip(text: str, limit: int = 6000) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]..."


# --- Individual judge scoring functions ---

def judge_retrieval_context_recall(
    question: str, reference_context: str, reference_answer: str,
    retrieved_context: str, device: str, judge_model: str = DEFAULT_JUDGE_MODEL,
) -> float:
    if not retrieved_context:
        return float("nan")
    response = _judge_generate(
        system_msg=(
            "You are an impartial evaluator for retrieval recall.\n"
            "Judge the RETRIEVED CONTENT by semantic meaning, not exact wording.\n"
            "Recall means: how completely the retrieved content covers the key "
            "information needed to answer the question according to the reference.\n\n"
            "Score 1-5:\n"
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
        judge_model=judge_model,
        device=device,
    )
    return _parse_score(response)


def judge_retrieval_context_precision(
    question: str, reference_context: str, reference_answer: str,
    retrieved_context: str, device: str, judge_model: str = DEFAULT_JUDGE_MODEL,
) -> float:
    if not retrieved_context:
        return float("nan")
    response = _judge_generate(
        system_msg=(
            "You are an impartial evaluator for retrieval precision.\n"
            "Judge the RETRIEVED CONTENT by semantic meaning, not exact wording.\n"
            "Precision means: how much of the retrieved content is relevant and "
            "useful for answering the question, compared with the reference.\n\n"
            "Score 1-5:\n"
            "  1 = Mostly irrelevant or misleading\n"
            "  2 = Limited relevance; mostly noisy or off-topic\n"
            "  3 = Mixed relevance; some useful, substantial noise\n"
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
        judge_model=judge_model,
        device=device,
    )
    return _parse_score(response)


def judge_answer_truthfulness(
    question: str, reference_answer: str, predicted_answer: str,
    device: str, judge_model: str = DEFAULT_JUDGE_MODEL,
) -> float:
    if not predicted_answer:
        return float("nan")
    response = _judge_generate(
        system_msg=(
            "You are an impartial evaluator for answer truthfulness.\n"
            "Judge whether the PREDICTED ANSWER correctly and truthfully answers "
            "the QUESTION, compared with the REFERENCE ANSWER.\n"
            "Use semantic meaning, not exact wording.\n\n"
            "Score 1-5:\n"
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
        judge_model=judge_model,
        device=device,
    )
    return _parse_score(response)


def judge_answer_faithfulness(
    predicted_answer: str, context: str,
    device: str, judge_model: str = DEFAULT_JUDGE_MODEL,
) -> float:
    if not context or not predicted_answer:
        return float("nan")
    response = _judge_generate(
        system_msg=(
            "You are an impartial evaluator for answer faithfulness.\n"
            "Judge whether the ANSWER is supported by and grounded in the CONTEXT.\n"
            "A faithful answer makes no claims beyond what the context supports.\n\n"
            "Score 1-5:\n"
            "  1 = Completely unfaithful; contradicts or fabricates\n"
            "  2 = Mostly unfaithful; major unsupported claims\n"
            "  3 = Partially faithful; mixed support\n"
            "  4 = Mostly faithful; minor unsupported details\n"
            "  5 = Fully faithful; all key claims are supported by context\n\n"
            "Respond with ONLY the score as a single integer (1-5)."
        ),
        user_msg=(
            f"CONTEXT:\n{_clip(context)}\n\n"
            f"ANSWER:\n{_clip(predicted_answer, 2000)}"
        ),
        judge_model=judge_model,
        device=device,
    )
    return _parse_score(response)


# ============================================================
# Semantic similarity (embedding-based)
# ============================================================

import numpy as np

_embed_model = None


def _get_embeddings(embed_preset: str = DEFAULT_EMBED_MODEL, device: str = "cuda"):
    """Load or return cached embedding model."""
    global _embed_model
    if _embed_model is None:
        print(f"Loading embedding model: {embed_preset}")
        _embed_model = load_embedding(embed_preset, device=device)
    return _embed_model


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def semantic_similarity(
    text_a: str, text_b: str,
    embed_preset: str = DEFAULT_EMBED_MODEL, device: str = "cuda",
) -> float:
    """Cosine similarity between two texts via embedding model."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    emb = _get_embeddings(embed_preset, device)
    vecs = emb.embed_documents([text_a, text_b])
    return cosine_similarity(vecs[0], vecs[1])


# ============================================================
# Lightweight text helpers (still used for source matching)
# ============================================================

_WORD_RE = re.compile(r"\w+")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(normalize_text(text))


def jaccard_similarity(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ============================================================
# Retrieval scoring helpers
# ============================================================

def has_source_match(pred_sources: list[str], item: GroundTruthItem) -> bool:
    gold_title = normalize_text(item.gold_title)
    gold_file = normalize_text(item.gold_file_name)
    for source in pred_sources:
        s = normalize_text(source)
        if gold_title and gold_title in s:
            return True
        if gold_file and gold_file in s:
            return True
    return False


def retrieval_recall_at_k(
    pred_sources: list[str], item: GroundTruthItem, k: int = 5,
) -> bool:
    """Recall@K: is the gold document in the top-K retrieved sources?

    Checks both title and file_name against the first K sources.
    Falls back to lexical context overlap if no source metadata available.
    """
    top_k = pred_sources[:k]
    return has_source_match(top_k, item)


def compute_retrieval_metrics(
    pred_context: str,
    pred_sources: list[str],
    item: GroundTruthItem,
    embed_preset: str = DEFAULT_EMBED_MODEL,
    device: str = "cuda",
) -> dict[str, Any]:
    """Compute retrieval metrics using semantic similarity."""
    source_match = has_source_match(pred_sources, item)
    ctx_similarity = semantic_similarity(
        pred_context, item.gold_context, embed_preset=embed_preset, device=device,
    )
    recall_at_5 = retrieval_recall_at_k(pred_sources, item, k=5)
    return {
        "ctx_similarity": ctx_similarity,
        "source_match": source_match,
        "recall_at_5": recall_at_5,
    }


# ============================================================
# Dataset loading
# ============================================================

def load_dataset(path: str | Path) -> list[GroundTruthItem]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [GroundTruthItem(**item) for item in data]


# ============================================================
# Benchmark runner
# ============================================================

class BenchmarkRunner:
    def __init__(
        self,
        agent: QAFlowAgent,
        max_rounds: int = 2,
        use_judge: bool = True,
        judge_device: str = "cuda",
        judge_model: str = DEFAULT_JUDGE_MODEL,
        embed_model: str = DEFAULT_EMBED_MODEL,
    ):
        self.agent = agent
        self.max_rounds = max_rounds
        self.use_judge = use_judge
        self.judge_device = judge_device
        self.judge_model = judge_model
        self.embed_model = embed_model

    @staticmethod
    def _print_trace(trace: StepTrace) -> None:
        """Print a compact summary of a step trace."""
        gen = trace.generation
        if gen:
            print(f"    {trace.step_name}: {trace.wall_time_ms:.0f}ms total | "
                  f"in={gen.input_tokens} out={gen.output_tokens} "
                  f"(think={gen.thinking_tokens} resp={gen.response_tokens}) | "
                  f"think={gen.think_time_ms:.0f}ms gen={gen.generate_time_ms:.0f}ms")
        else:
            ctx_len = len(trace.retrieved_context) if trace.retrieved_context else 0
            print(f"    {trace.step_name}: {trace.wall_time_ms:.0f}ms | "
                  f"context_chars={ctx_len} sources={len(trace.retrieved_sources)}")

    def run_item(self, item: GroundTruthItem, idx: int = 0, total: int = 0) -> ItemResult:
        label = f"[{idx}/{total}]" if total else ""
        print(f"\n{label} Question {item.id}: {item.question[:80]}...")

        rounds_used = 0
        traces: list[StepTrace] = []

        # --- Step 1: Retrieve ---
        print("  Step 1: Retrieving...")
        retrieval, trace_ret = self.agent.retrieve(item.question)
        traces.append(trace_ret)
        self._print_trace(trace_ret)
        rounds_used += 1

        ret_metrics = compute_retrieval_metrics(
            retrieval.retrieved_context, retrieval.retrieved_sources, item,
            embed_preset=self.embed_model, device=self.judge_device,
        )

        # --- Step 2: Validate ---
        print("  Step 2: Validating retrieval...")
        validation, trace_val = self.agent.validate_retrieval(
            item.question, retrieval.retrieved_context,
        )
        traces.append(trace_val)
        self._print_trace(trace_val)
        predicted_status = (validation.status or "").upper() or None
        validator_scored = item.round1_status is not None
        validator_correct = None
        if validator_scored:
            validator_correct = predicted_status == item.round1_status.upper()

        current_context = retrieval.retrieved_context
        current_sources = retrieval.retrieved_sources
        followup_scored = item.gold_missing_information is not None
        followup_similarity: float | None = None

        # --- Step 2b: Follow-up if incomplete ---
        if predicted_status == "INCOMPLETE" and rounds_used < self.max_rounds:
            print("  Step 2b: Retrieving missing information...")
            if followup_scored:
                followup_similarity = jaccard_similarity(
                    validation.missing_information,
                    item.gold_missing_information or "",
                )
            second, trace_miss = self.agent.find_missing_information(
                item.question, validation.missing_information,
                validation.useful_information,
            )
            traces.append(trace_miss)
            self._print_trace(trace_miss)
            rounds_used += 1
            current_context = second.retrieved_context
            current_sources = second.retrieved_sources or current_sources

        # --- Step 3: Answer ---
        print("  Step 3: Generating answer...")
        answer_result, trace_ans = self.agent.answer(item.question, current_context)
        traces.append(trace_ans)
        self._print_trace(trace_ans)

        # --- Semantic answer metrics ---
        print("  Computing semantic similarity scores...")
        answer_sim = semantic_similarity(
            answer_result.answer, item.gold_answer,
            embed_preset=self.embed_model, device=self.judge_device,
        )
        answer_ground = semantic_similarity(
            answer_result.answer, current_context,
            embed_preset=self.embed_model, device=self.judge_device,
        )

        # --- LLM-judge metrics ---
        judge_ctx_recall = float("nan")
        judge_ctx_precision = float("nan")
        judge_truthfulness = float("nan")
        judge_faithfulness = float("nan")

        if self.use_judge:
            print(f"  Scoring with LLM judge ({self.judge_model})...")
            judge_ctx_recall = judge_retrieval_context_recall(
                question=item.question,
                reference_context=item.gold_context,
                reference_answer=item.gold_answer,
                retrieved_context=current_context,
                device=self.judge_device,
                judge_model=self.judge_model,
            )
            judge_ctx_precision = judge_retrieval_context_precision(
                question=item.question,
                reference_context=item.gold_context,
                reference_answer=item.gold_answer,
                retrieved_context=current_context,
                device=self.judge_device,
                judge_model=self.judge_model,
            )
            judge_truthfulness = judge_answer_truthfulness(
                question=item.question,
                reference_answer=item.gold_answer,
                predicted_answer=answer_result.answer,
                device=self.judge_device,
                judge_model=self.judge_model,
            )
            judge_faithfulness = judge_answer_faithfulness(
                predicted_answer=answer_result.answer,
                context=current_context,
                device=self.judge_device,
                judge_model=self.judge_model,
            )
            print(f"    Recall={judge_ctx_recall} Precision={judge_ctx_precision} "
                  f"Truthfulness={judge_truthfulness} Faithfulness={judge_faithfulness}")

        return ItemResult(
            id=item.id,
            question=item.question,
            rounds_used=rounds_used,
            step_traces=[asdict(t) for t in traces],
            retrieval_recall_at_5=ret_metrics["recall_at_5"],
            retrieval_context_similarity=ret_metrics["ctx_similarity"],
            retrieval_source_match=ret_metrics["source_match"],
            retrieval_context_recall_judge=judge_ctx_recall,
            retrieval_context_precision_judge=judge_ctx_precision,
            answer_truthfulness_judge=judge_truthfulness,
            answer_faithfulness_judge=judge_faithfulness,
            answer_similarity=answer_sim,
            answer_groundedness=answer_ground,
            validator_scored=validator_scored,
            validator_correct=validator_correct,
            predicted_round1_status=predicted_status,
            gold_round1_status=item.round1_status,
            followup_scored=followup_scored,
            followup_missing_info_similarity=followup_similarity,
            predicted_answer=answer_result.answer,
            gold_answer=item.gold_answer,
            retrieved_context=current_context,
            retrieved_sources=current_sources,
        )

    def run(self, dataset: list[GroundTruthItem]) -> dict[str, Any]:
        total = len(dataset)
        results = [self.run_item(item, idx=i + 1, total=total) for i, item in enumerate(dataset)]
        return {
            "summary": summarize_results(results),
            "per_item": [asdict(r) for r in results],
        }


# ============================================================
# Summary metrics
# ============================================================

def _safe_mean(values: list[float]) -> float:
    clean = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    return statistics.fmean(clean) if clean else math.nan


def _fmt(val: float) -> str:
    if isinstance(val, float) and math.isnan(val):
        return "N/A"
    return f"{val:.3f}"


def summarize_results(results: list[ItemResult]) -> dict[str, Any]:
    n = len(results)

    # --- Primary metrics (the 4 requested + Recall@5) ---
    recall_at_5_hits = [1.0 if r.retrieval_recall_at_5 else 0.0 for r in results]
    judge_ctx_recalls = [r.retrieval_context_recall_judge for r in results]
    judge_ctx_precisions = [r.retrieval_context_precision_judge for r in results]
    judge_truthfulness = [r.answer_truthfulness_judge for r in results]
    judge_faithfulness = [r.answer_faithfulness_judge for r in results]

    summary: dict[str, Any] = {
        "num_questions": n,
        # --- Primary scored metrics ---
        "retrieval_recall_at_5": _safe_mean(recall_at_5_hits),
        "retrieval_context_recall_judge": _safe_mean(judge_ctx_recalls),
        "retrieval_context_precision_judge": _safe_mean(judge_ctx_precisions),
        "answer_truthfulness_judge": _safe_mean(judge_truthfulness),
        "answer_faithfulness_judge": _safe_mean(judge_faithfulness),
        # --- Semantic similarity metrics (embedding-based) ---
        "retrieval_context_similarity": _safe_mean(
            [r.retrieval_context_similarity for r in results]
        ),
        "retrieval_source_match_rate": _safe_mean(
            [1.0 if r.retrieval_source_match else 0.0 for r in results]
        ),
        "answer_similarity": _safe_mean([r.answer_similarity for r in results]),
        "answer_groundedness": _safe_mean([r.answer_groundedness for r in results]),
        "avg_rounds_used": _safe_mean([float(r.rounds_used) for r in results]),
    }

    # Validator accuracy (only if ground-truth has round1_status)
    validator_subset = [r for r in results if r.validator_scored]
    if validator_subset:
        summary["validator_accuracy"] = _safe_mean(
            [1.0 if r.validator_correct else 0.0 for r in validator_subset]
        )
        summary["validator_num_scored"] = len(validator_subset)

    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print(f"\n{'=' * 64}")
    print("QAAgent BENCHMARK RESULTS")
    print(f"{'=' * 64}")
    print(f"  Questions evaluated: {summary['num_questions']}")

    print(f"\n--- Retrieval Metrics ---")
    print(f"  Recall@5:                      {_fmt(summary['retrieval_recall_at_5'])}")
    print(f"  Context Similarity (semantic): {_fmt(summary['retrieval_context_similarity'])}")
    print(f"  Context Recall  (judge 1-5):   {_fmt(summary['retrieval_context_recall_judge'])}")
    print(f"  Context Precision (judge 1-5): {_fmt(summary['retrieval_context_precision_judge'])}")
    print(f"  Source Match Rate:             {_fmt(summary['retrieval_source_match_rate'])}")

    print(f"\n--- Answer Metrics ---")
    print(f"  Truthfulness  (judge 1-5):     {_fmt(summary['answer_truthfulness_judge'])}")
    print(f"  Faithfulness  (judge 1-5):     {_fmt(summary['answer_faithfulness_judge'])}")
    print(f"  Answer Similarity (semantic):  {_fmt(summary['answer_similarity'])}")
    print(f"  Groundedness (semantic):       {_fmt(summary['answer_groundedness'])}")

    print(f"\n--- Workflow ---")
    print(f"  Avg Rounds Used:               {_fmt(summary['avg_rounds_used'])}")
    if "validator_accuracy" in summary:
        print(f"  Validator Accuracy:            {_fmt(summary['validator_accuracy'])} "
              f"(n={summary['validator_num_scored']})")
    print(f"{'=' * 64}\n")


# ============================================================
# Example agents / adapters
# ============================================================

class GoldContextSanityAgent:
    """Oracle agent that returns gold answers. For verifying the benchmark pipeline."""

    def __init__(self, dataset: list[GroundTruthItem]):
        self.by_question = {item.question: item for item in dataset}

    def retrieve(self, question: str) -> tuple[RetrievalResult, StepTrace]:
        item = self.by_question[question]
        ctx = item.gold_context
        sources = [item.gold_title, item.gold_file_name]
        trace = StepTrace(
            step_name="retrieve", step_number=1, wall_time_ms=0.0,
            input_text=question, output_text=ctx,
            retrieved_context=ctx, retrieved_sources=sources,
        )
        return RetrievalResult(retrieved_context=ctx, retrieved_sources=sources), trace

    def validate_retrieval(
        self, question: str, retrieved_context: str,
    ) -> tuple[ValidationResult, StepTrace]:
        item = self.by_question[question]
        status = item.round1_status or "COMPLETE"
        trace = StepTrace(
            step_name="validate", step_number=2, wall_time_ms=0.0,
            input_text=question, output_text=status,
        )
        return ValidationResult(
            status=status,
            useful_information=retrieved_context,
            missing_information=item.gold_missing_information or "",
        ), trace

    def find_missing_information(
        self, question: str, missing_information: str, useful_information: str,
    ) -> tuple[RetrievalResult, StepTrace]:
        item = self.by_question[question]
        ctx = (useful_information + "\n" + item.gold_context).strip()
        sources = [item.gold_title, item.gold_file_name]
        trace = StepTrace(
            step_name="find_missing", step_number=3, wall_time_ms=0.0,
            input_text=missing_information, output_text=ctx,
            retrieved_context=ctx, retrieved_sources=sources,
        )
        return RetrievalResult(retrieved_context=ctx, retrieved_sources=sources), trace

    def answer(
        self, question: str, retrieved_context: str,
    ) -> tuple[AnswerResult, StepTrace]:
        item = self.by_question[question]
        trace = StepTrace(
            step_name="answer", step_number=4, wall_time_ms=0.0,
            input_text=question, output_text=item.gold_answer,
        )
        return AnswerResult(answer=item.gold_answer), trace


class R1ReasoningRAGAdapter:
    """Adapter wrapping the r1-reasoning-rag QAAgent flow for benchmarking.

    Mirrors github.com/astordu/r1-reasoning-rag/src/agent.py but uses:
      - Tavily for retrieval (top 3 results)
      - Local Qwen3-8B (via model_loader) for validation + answering
        instead of DeepSeek R1 via OpenRouter

    Prompts are from r1-reasoning-rag/src/prompts.py (verbatim).
    """

    # --- Prompts from r1-reasoning-rag/src/prompts.py ---
    VALIDATE_PROMPT = (
        "You are a retrieval validator.\n"
        "You will be provided with a question and chunks of text that may or may not "
        "contain the answer to the question.\n"
        "Your role is to carefully look through the chunks of text and provide a JSON "
        "response with three fields:\n"
        "1. status: whether the retrieved chunks contain the answer to the question.\n"
        "- 'COMPLETE' if the retrieved chunks contain the answer to the question, "
        "'INCOMPLETE' otherwise. Nothing else.\n\n"
        "2. useful_information: the useful information from the retrieved chunks. "
        "Be concise and direct.\n"
        "- if there is no useful information, set this to an empty string.\n\n"
        "3. missing_information: the missing information that is needed to answer "
        "the question in full. Be concise and direct.\n"
        "- if there is no missing information, set this to an empty string.\n\n"
        "Please provide your response as dictionary in the following format.\n\n"
        '{{"status": "<status>",\n'
        '"useful_information": "<useful_information>",\n'
        '"missing_information": "<missing_information>"}}\n\n'
        "Do not include any other text."
    )

    ANSWER_PROMPT = (
        "You are a question answering agent.\n"
        "You will be provided with a question and chunks of text that contain the "
        "answer to the question.\n"
        "Your role is to carefully look through the chunks of text and answer the "
        "question.\n"
        "Provide a direct and concise answer based on the information provided.\n"
        "Do not include any additional information or commentary."
    )

    def __init__(self, agent_model: str = DEFAULT_AGENT_MODEL, device: str = "cuda"):
        from dotenv import load_dotenv
        load_dotenv()
        from tavily import TavilyClient

        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.agent_model = agent_model
        self.device = device

        # Pre-load the agent model so first call isn't slow
        print(f"Loading agent model: {agent_model}")
        _get_model(agent_model, device)

    def _agent_generate(
        self, system_msg: str, user_msg: str, max_new_tokens: int = 512,
    ) -> tuple[str, GenerationStats]:
        """Generate using the local agent model. Returns (text, stats)."""
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        return _generate(
            self.agent_model, messages, device=self.device,
            max_new_tokens=max_new_tokens, enable_thinking=True,
            temperature=0.6, top_p=0.7,
        )

    def retrieve(self, question: str) -> tuple[RetrievalResult, StepTrace]:
        t0 = time.perf_counter()
        result = self.tavily.search(question, max_results=3)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        docs = result.get("results", [])
        context = "\n".join(r["content"] for r in docs)
        sources = [r.get("title", r.get("url", "")) for r in docs]
        trace = StepTrace(
            step_name="retrieve", step_number=1, wall_time_ms=wall_ms,
            input_text=question, output_text=context,
            retrieved_context=context, retrieved_sources=sources,
        )
        return RetrievalResult(
            retrieved_context=context, retrieved_sources=sources, raw=result,
        ), trace

    def validate_retrieval(
        self, question: str, retrieved_context: str,
    ) -> tuple[ValidationResult, StepTrace]:
        user_msg = f"Context: {retrieved_context}\n\nThe Question: {question}\nResponse:"
        t0 = time.perf_counter()
        llm_output, gen_stats = self._agent_generate(self.VALIDATE_PROMPT, user_msg)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        response, reasoning = _strip_think_tags(llm_output)
        trace = StepTrace(
            step_name="validate", step_number=2, wall_time_ms=wall_ms,
            generation=gen_stats,
            input_text=user_msg, output_text=response, thinking_text=reasoning,
        )
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            return ValidationResult(
                status="COMPLETE", reasoning=reasoning, raw={"raw": response},
            ), trace
        return ValidationResult(
            status=parsed.get("status", "COMPLETE"),
            missing_information=parsed.get("missing_information", ""),
            useful_information=parsed.get("useful_information", ""),
            reasoning=reasoning, raw=parsed,
        ), trace

    def find_missing_information(
        self, question: str, missing_information: str, useful_information: str,
    ) -> tuple[RetrievalResult, StepTrace]:
        t0 = time.perf_counter()
        result = self.tavily.search(missing_information, max_results=3)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        docs = result.get("results", [])
        new_context = "\n".join(r["content"] for r in docs)
        combined = f"{useful_information}\n{new_context}"
        sources = [r.get("title", r.get("url", "")) for r in docs]
        trace = StepTrace(
            step_name="find_missing", step_number=3, wall_time_ms=wall_ms,
            input_text=missing_information, output_text=combined,
            retrieved_context=combined, retrieved_sources=sources,
        )
        return RetrievalResult(
            retrieved_context=combined, retrieved_sources=sources, raw=result,
        ), trace

    def answer(
        self, question: str, retrieved_context: str,
    ) -> tuple[AnswerResult, StepTrace]:
        user_msg = f"The Question: {question}\nContext: {retrieved_context}\nAnswer:"
        t0 = time.perf_counter()
        llm_output, gen_stats = self._agent_generate(self.ANSWER_PROMPT, user_msg)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        response, reasoning = _strip_think_tags(llm_output)
        trace = StepTrace(
            step_name="answer", step_number=4, wall_time_ms=wall_ms,
            generation=gen_stats,
            input_text=user_msg, output_text=response, thinking_text=reasoning,
        )
        return AnswerResult(answer=response, reasoning=reasoning), trace


class GenericAgentAdapter:
    """Generic adapter for any agent with retrieve/validate/find/answer methods."""

    def __init__(self, agent: Any):
        self.agent = agent

    def retrieve(self, question: str) -> tuple[RetrievalResult, StepTrace]:
        t0 = time.perf_counter()
        raw = self.agent.retrieve(question)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        if isinstance(raw, str):
            result = RetrievalResult(retrieved_context=raw)
        else:
            result = RetrievalResult(
                retrieved_context=raw.get("retrieved_context", ""),
                retrieved_sources=raw.get("retrieved_sources", []),
                raw=raw,
            )
        trace = StepTrace(
            step_name="retrieve", step_number=1, wall_time_ms=wall_ms,
            input_text=question, output_text=result.retrieved_context,
            retrieved_context=result.retrieved_context,
            retrieved_sources=result.retrieved_sources,
        )
        return result, trace

    def validate_retrieval(
        self, question: str, retrieved_context: str,
    ) -> tuple[ValidationResult, StepTrace]:
        t0 = time.perf_counter()
        raw = self.agent.validate_retrieval(question, retrieved_context)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        if isinstance(raw, str):
            result = ValidationResult(status=raw)
        else:
            result = ValidationResult(
                status=raw.get("status") or raw.get("router_decision", ""),
                missing_information=raw.get("missing_information", ""),
                useful_information=raw.get("useful_information", ""),
                reasoning=raw.get("reasoning", ""),
                raw=raw,
            )
        trace = StepTrace(
            step_name="validate", step_number=2, wall_time_ms=wall_ms,
            input_text=question, output_text=result.status,
        )
        return result, trace

    def find_missing_information(
        self, question: str, missing_information: str, useful_information: str,
    ) -> tuple[RetrievalResult, StepTrace]:
        t0 = time.perf_counter()
        raw = self.agent.find_missing_information(
            question, missing_information, useful_information,
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        if isinstance(raw, str):
            result = RetrievalResult(retrieved_context=raw)
        else:
            result = RetrievalResult(
                retrieved_context=raw.get("retrieved_context", ""),
                retrieved_sources=raw.get("retrieved_sources", []),
                raw=raw,
            )
        trace = StepTrace(
            step_name="find_missing", step_number=3, wall_time_ms=wall_ms,
            input_text=missing_information, output_text=result.retrieved_context,
            retrieved_context=result.retrieved_context,
            retrieved_sources=result.retrieved_sources,
        )
        return result, trace

    def answer(
        self, question: str, retrieved_context: str,
    ) -> tuple[AnswerResult, StepTrace]:
        t0 = time.perf_counter()
        raw = self.agent.answer(question, retrieved_context)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        if isinstance(raw, str):
            result = AnswerResult(answer=raw)
        else:
            result = AnswerResult(
                answer=raw.get("answer") or raw.get("answer_to_question", ""),
                reasoning=raw.get("reasoning", ""),
                raw=raw,
            )
        trace = StepTrace(
            step_name="answer", step_number=4, wall_time_ms=wall_ms,
            input_text=question, output_text=result.answer,
        )
        return result, trace


# ============================================================
# Dynamic import helper
# ============================================================

def load_agent_from_spec(
    spec: str, agent_model: str = DEFAULT_AGENT_MODEL, device: str = "cuda",
) -> QAFlowAgent:
    """Load an agent.

    Special values:
      'qaagent' — instantiate R1ReasoningRAGAdapter with local Qwen3-8B
      'module.path:ClassName' — dynamic import
    """
    if spec == "qaagent":
        return R1ReasoningRAGAdapter(agent_model=agent_model, device=device)

    if ":" not in spec:
        raise ValueError(
            "Agent spec must be 'qaagent' or 'module.submodule:ClassName'"
        )
    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return GenericAgentAdapter(cls())


# ============================================================
# CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark a QAAgent-style retrieve/validate/retrieve/answer pipeline.",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to the JSON benchmark file.",
    )
    parser.add_argument(
        "--agent-spec",
        default=None,
        help="Agent to benchmark: 'qaagent' for r1-reasoning-rag, or 'module:Class'.",
    )
    parser.add_argument(
        "--agent-model",
        default=DEFAULT_AGENT_MODEL,
        help=f"Local model for agent LLM (default: {DEFAULT_AGENT_MODEL}).",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"Local model for LLM judge (default: {DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help=f"Embedding model for semantic similarity (default: {DEFAULT_EMBED_MODEL}).",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=2,
        help="Maximum retrieval rounds to allow (default: 2).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save full results (per-item + summary) as JSON.",
    )
    parser.add_argument(
        "--gold-sanity",
        action="store_true",
        help="Run with gold oracle agent (pipeline sanity check).",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-judge scoring (lexical metrics only, no GPU needed).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for all local models (default: cuda).",
    )
    return parser


# ============================================================
# CSV export
# ============================================================

# One row per step per question — gives full step-by-step detail.
_CSV_COLUMNS = [
    # Question-level
    "question_id", "question", "rounds_used",
    # Step-level
    "step_name", "step_number", "wall_time_ms",
    # LLM generation (None for retrieval-only steps)
    "input_tokens", "output_tokens",
    "thinking_tokens", "response_tokens",
    "think_time_ms", "generate_time_ms",
    # Content
    "input_text", "output_text", "thinking_text",
    "retrieved_context", "retrieved_sources",
    # Scores (repeated on every row for easy filtering)
    "retrieval_recall_at_5", "retrieval_context_similarity",
    "retrieval_context_recall_judge", "retrieval_context_precision_judge",
    "answer_truthfulness_judge", "answer_faithfulness_judge",
    "answer_similarity", "answer_groundedness",
    # Answer text (on answer step row only for readability)
    "predicted_answer", "gold_answer",
]


def _save_results_csv(results: dict[str, Any], csv_path: Path) -> None:
    """Write one row per step per question to a CSV file."""
    rows: list[dict[str, Any]] = []
    for item in results["per_item"]:
        # Shared score fields for this question
        scores = {
            "question_id": item["id"],
            "question": item["question"],
            "rounds_used": item["rounds_used"],
            "retrieval_recall_at_5": item["retrieval_recall_at_5"],
            "retrieval_context_similarity": item["retrieval_context_similarity"],
            "retrieval_context_recall_judge": item["retrieval_context_recall_judge"],
            "retrieval_context_precision_judge": item["retrieval_context_precision_judge"],
            "answer_truthfulness_judge": item["answer_truthfulness_judge"],
            "answer_faithfulness_judge": item["answer_faithfulness_judge"],
            "answer_similarity": item["answer_similarity"],
            "answer_groundedness": item["answer_groundedness"],
            "predicted_answer": "",
            "gold_answer": "",
        }
        for trace in item.get("step_traces", []):
            gen = trace.get("generation") or {}
            row = {
                **scores,
                "step_name": trace["step_name"],
                "step_number": trace["step_number"],
                "wall_time_ms": f"{trace['wall_time_ms']:.1f}",
                "input_tokens": gen.get("input_tokens", ""),
                "output_tokens": gen.get("output_tokens", ""),
                "thinking_tokens": gen.get("thinking_tokens", ""),
                "response_tokens": gen.get("response_tokens", ""),
                "think_time_ms": f"{gen['think_time_ms']:.1f}" if gen.get("think_time_ms") else "",
                "generate_time_ms": f"{gen['generate_time_ms']:.1f}" if gen.get("generate_time_ms") else "",
                "input_text": trace.get("input_text", ""),
                "output_text": trace.get("output_text", ""),
                "thinking_text": trace.get("thinking_text", ""),
                "retrieved_context": trace.get("retrieved_context", ""),
                "retrieved_sources": "; ".join(trace.get("retrieved_sources", [])),
            }
            # Attach answer text only on the answer step
            if trace["step_name"] == "answer":
                row["predicted_answer"] = item["predicted_answer"]
                row["gold_answer"] = item["gold_answer"]
            rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} questions from {args.dataset}")
    print(f"Agent model : {args.agent_model}")
    print(f"Judge model : {args.judge_model} {'(disabled)' if args.no_judge else ''}")
    print(f"Embed model : {args.embed_model}")
    print(f"Device      : {args.device}")

    if args.gold_sanity:
        benchmark_agent: QAFlowAgent = GoldContextSanityAgent(dataset)
    elif args.agent_spec:
        benchmark_agent = load_agent_from_spec(
            args.agent_spec, agent_model=args.agent_model, device=args.device,
        )
    else:
        raise SystemExit(
            "Provide either --gold-sanity or --agent-spec (e.g. --agent-spec qaagent)"
        )

    use_judge = not args.no_judge
    runner = BenchmarkRunner(
        agent=benchmark_agent,
        max_rounds=args.max_rounds,
        use_judge=use_judge,
        judge_device=args.device,
        judge_model=args.judge_model,
        embed_model=args.embed_model,
    )
    results = runner.run(dataset)

    # Print summary
    print_summary(results["summary"])

    # Also dump raw JSON summary
    print(json.dumps(results["summary"], indent=2, ensure_ascii=False))

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        print(f"\nFull results saved to: {output_path}")

    # --- Always save CSV to dataset/ with today's date + model info ---
    today = datetime.now().strftime("%Y-%m-%d")
    agent_tag = "gold" if args.gold_sanity else (args.agent_model or "unknown")
    judge_tag = "nojudge" if args.no_judge else args.judge_model
    csv_name = f"qaagent_benchmark_{today}_{agent_tag}_{judge_tag}_{args.device}.csv"
    csv_path = DATASET_DIR / csv_name
    _save_results_csv(results, csv_path)
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
