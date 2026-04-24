"""
ReAct RAG LLM Benchmark
========================
Benchmarks the ReAct RAG architecture with retrieval as the only tool.
The model decides whether to answer directly or retrieve documents first,
following the Think → Action → Observe loop for multi-step retrieval.

Uses model_loader.py from langGraph/ to load models, enabling easy
benchmarking across different LLMs.

Usage:
  python react_llm.py                                    # default: qwen3-8b on cuda
  python react_llm.py --model qwen3-8b --device cuda     # explicit model + device
  python react_llm.py --model qwen3-14b --gpu 0,1        # multi-GPU
  python react_llm.py --k 3 --max-steps 5                # 3 runs per question
  python react_llm.py --interactive                       # interactive mode
  python react_llm.py --qa-file path.json                # custom QA file
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Allow importing model_loader from langGraph/
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent / "langGraph"))
from model_loader import load_llm, MODELS_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
DB_DIR = PROJECT_DIR / "db"
DEFAULT_QA_FILE = DATASET_DIR / "agentic_rag_gt.json"

RETRIEVAL_COLLECTION_NAME = "documents"


# ---------------------------------------------------------------------------
# Retrieval tool (singleton vectorstore)
# ---------------------------------------------------------------------------
_retrieval_vectorstore = None


def _inspect_chroma_collection(chroma_dir: Path, collection_name: str) -> tuple[int | None, int]:
    """Return (dimension, embedding_count) for a Chroma collection on disk."""
    sqlite_path = chroma_dir / "chroma.sqlite3"
    if not sqlite_path.exists():
        return None, 0

    conn = sqlite3.connect(sqlite_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, dimension FROM collections WHERE name = ?",
            (collection_name,),
        )
        row = cur.fetchone()
        if row is None:
            return None, 0

        collection_id, dimension = row
        cur.execute(
            """
            SELECT COUNT(*)
            FROM embeddings e
            JOIN segments s ON e.segment_id = s.id
            WHERE s.collection = ?
            """,
            (collection_id,),
        )
        count = cur.fetchone()[0]
        return dimension, count
    finally:
        conn.close()


def _get_vectorstore():
    """Lazy-load and cache the embedding model + Chroma vectorstore."""
    global _retrieval_vectorstore
    if _retrieval_vectorstore is not None:
        return _retrieval_vectorstore

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    embedding_model = MODELS_DIR / "BAAI--bge-large-en-v1.5"
    embedding_model_name = (
        str(embedding_model) if embedding_model.exists()
        else "BAAI/bge-large-en-v1.5"
    )
    embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
    chroma_dir = DB_DIR / "chroma"

    dim, count = _inspect_chroma_collection(chroma_dir, RETRIEVAL_COLLECTION_NAME)
    if count == 0:
        raise RuntimeError(
            f"Chroma collection '{RETRIEVAL_COLLECTION_NAME}' is missing or empty "
            f"in {chroma_dir}. Rebuild the vector DB with vectorDB.py."
        )

    print("Loading retrieval embedding model (BGE-large-en-v1.5)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )

    _retrieval_vectorstore = Chroma(
        collection_name=RETRIEVAL_COLLECTION_NAME,
        persist_directory=str(chroma_dir),
        embedding_function=embeddings,
    )
    print(
        f"Retrieval model ready "
        f"(collection={RETRIEVAL_COLLECTION_NAME}, dim={dim}, count={count}, "
        f"device={embedding_device})."
    )
    return _retrieval_vectorstore


def tool_retrieve(query: str) -> str:
    """Retrieve relevant documents from the vector database."""
    try:
        vectorstore = _get_vectorstore()
    except RuntimeError as e:
        return f"Error: {e}"

    results = vectorstore.similarity_search(query, k=5)
    if not results:
        return f"No relevant documents found for '{query}'."

    lines = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        lines.append(f"--- Document {i} (source: {Path(source).name}) ---")
        lines.append(doc.page_content)
        lines.append("")

    return "\n".join(lines)


TOOLS = {
    "retrieve": {
        "fn": tool_retrieve,
        "description": "Retrieve relevant documents from the knowledge base. "
                       "Input: a natural language query about the topic you need information on.",
    },
}


# ---------------------------------------------------------------------------
# System prompt — ReAct with retrieve-only
# ---------------------------------------------------------------------------
def build_system_prompt() -> str:
    today = datetime.now().strftime("%Y-%m-%d (%A)")
    return f"""You are a helpful assistant that answers questions using the ReAct framework (Reasoning + Acting).

Today's date is {today}.

You have access to one tool:
  - retrieve: Retrieve relevant documents from a knowledge base. Input: a natural language query.

For each step, respond in EXACTLY this format:

Thought: <your reasoning about what to do next>
Action: retrieve
Action Input: <your search query>

After you receive an Observation, continue with another Thought/Action or give the final answer:

Thought: <your reasoning>
Final Answer: <your complete answer to the user>

Rules:
- Always start with a Thought.
- If you already know the answer confidently, you may go directly to Final Answer without using any tool.
- If the question requires specific knowledge you're not sure about, use the retrieve tool.
- You may retrieve multiple times with different queries to gather enough information.
- Each response must be exactly one of:
  1. Thought + Action + Action Input (then wait for Observation)
  2. Thought + Final Answer
- Never include both an Action and a Final Answer in the same response.
- Never write an Observation yourself — they come from tool execution only.
- When you have enough information, use "Final Answer:" to respond.
- Be concise and accurate."""


# ---------------------------------------------------------------------------
# ReAct Agent using model_loader
# ---------------------------------------------------------------------------
class ReActRetrievalAgent:
    """ReAct agent with retrieval-only tool, loaded via model_loader."""

    def __init__(
        self,
        model_name: str = "qwen3-8b",
        device: str = "cuda",
        gpu_ids: list[int] | None = None,
        max_steps: int = 5,
        max_new_tokens: int = 2048,
    ):
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens

        self.model, self.tokenizer = load_llm(
            model_name, device=device, gpu_ids=gpu_ids,
        )
        # Determine input device for tokenized tensors
        # OpenVINO models run on CPU and don't have .parameters()
        try:
            self.input_device = next(self.model.parameters()).device
        except (StopIteration, AttributeError):
            self.input_device = "cpu"

    def _generate(self, messages: list[dict]) -> str:
        """Run a single generation turn."""
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.input_device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @staticmethod
    def _parse_action(text: str) -> tuple[str | None, str | None]:
        action_match = re.search(r"Action:\s*(.+)", text)
        input_match = re.search(r"Action Input:\s*(.+)", text)
        if action_match and input_match:
            return action_match.group(1).strip(), input_match.group(1).strip()
        return None, None

    @staticmethod
    def _parse_final_answer(text: str) -> str | None:
        match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def run(self, question: str) -> str:
        """Run the ReAct loop. Returns the final answer."""
        system_prompt = build_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        for step in range(1, self.max_steps + 1):
            print(f"\n--- Step {step} ---")
            response = self._generate(messages)
            print(response)

            action, action_input = self._parse_action(response)
            if action is not None:
                tool = TOOLS.get(action.lower())
                if tool is None:
                    observation = (
                        f"Error: Unknown tool '{action}'. "
                        f"Available: {', '.join(TOOLS.keys())}"
                    )
                else:
                    observation = tool["fn"](action_input)

                print(f"Observation: {observation[:200]}...")
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Observation: {observation}"})
                continue

            final = self._parse_final_answer(response)
            if final:
                return final

            # Model didn't follow format — nudge it
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": "Please respond with either an Action or a Final Answer.",
            })

        return "Reached maximum steps without a final answer."


# ---------------------------------------------------------------------------
# Tracing data classes
# ---------------------------------------------------------------------------
def _gpu_snapshot() -> int | None:
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated()


def _detect_device(before: int | None) -> str:
    if before is None:
        return "CPU"
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return "GPU" if peak > before else "CPU"


@dataclass
class StepTrace:
    step: int
    thought: str
    tool_called: str | None
    action_input: str | None
    observation: str | None
    generation_ms: float
    tool_ms: float
    generation_device: str
    tool_device: str
    duration_ms: float


@dataclass
class QuestionResult:
    question: str
    reference_answer: str
    predicted_answer: str
    expected_tool: str
    reference_context: str
    traces: list[StepTrace] = field(default_factory=list)
    total_steps: int = 0
    total_duration_ms: float = 0.0
    retrieved_sources: list[str] = field(default_factory=list)
    retrieved_context: str = ""
    used_retrieval: bool = False


# ---------------------------------------------------------------------------
# Benchmark Agent — wraps ReActRetrievalAgent with tracing
# ---------------------------------------------------------------------------
class BenchmarkAgent:
    """Wraps ReActRetrievalAgent with per-step tracing and timing."""

    def __init__(
        self,
        model_name: str = "qwen3-8b",
        device: str = "cuda",
        gpu_ids: list[int] | None = None,
        max_steps: int = 5,
    ):
        self.agent = ReActRetrievalAgent(
            model_name=model_name, device=device,
            gpu_ids=gpu_ids, max_steps=max_steps,
        )
        # Pre-load retrieval vectorstore
        _get_vectorstore()

    def run_traced(self, question: str) -> QuestionResult:
        result = QuestionResult(
            question=question,
            reference_answer="",
            predicted_answer="",
            expected_tool="",
            reference_context="",
        )

        system_prompt = build_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        t_start = time.perf_counter()

        for step in range(1, self.agent.max_steps + 1):
            step_start = time.perf_counter()

            # --- LLM generation ---
            gen_snap = _gpu_snapshot()
            gen_start = time.perf_counter()
            response = self.agent._generate(messages)
            gen_ms = (time.perf_counter() - gen_start) * 1000
            gen_device = _detect_device(gen_snap)

            action, action_input = self.agent._parse_action(response)
            if action is not None:
                # --- Tool execution ---
                tool_snap = _gpu_snapshot()
                tool_start = time.perf_counter()
                tool = TOOLS.get(action.lower())
                if tool is None:
                    observation = f"Error: Unknown tool '{action}'."
                else:
                    observation = tool["fn"](action_input)
                tool_ms = (time.perf_counter() - tool_start) * 1000
                tool_device = _detect_device(tool_snap)

                if action.lower() == "retrieve":
                    result.used_retrieval = True
                    result.retrieved_context = observation
                    for match in re.finditer(
                        r"--- Document \d+ \(source:\s*(.+?)\) ---", observation
                    ):
                        result.retrieved_sources.append(match.group(1).strip())

                step_ms = (time.perf_counter() - step_start) * 1000
                result.traces.append(StepTrace(
                    step=step, thought=response,
                    tool_called=action.lower(),
                    action_input=action_input,
                    observation=observation,
                    generation_ms=gen_ms, tool_ms=tool_ms,
                    generation_device=gen_device, tool_device=tool_device,
                    duration_ms=step_ms,
                ))

                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Observation: {observation}"})
                continue

            final = self.agent._parse_final_answer(response)
            if final:
                step_ms = (time.perf_counter() - step_start) * 1000
                result.traces.append(StepTrace(
                    step=step, thought=response,
                    tool_called=None, action_input=None,
                    observation="[final answer]",
                    generation_ms=gen_ms, tool_ms=0.0,
                    generation_device=gen_device, tool_device="N/A",
                    duration_ms=step_ms,
                ))
                result.predicted_answer = final
                break

            step_ms = (time.perf_counter() - step_start) * 1000
            result.traces.append(StepTrace(
                step=step, thought=response,
                tool_called=None, action_input=None,
                observation="[format error]",
                generation_ms=gen_ms, tool_ms=0.0,
                generation_device=gen_device, tool_device="N/A",
                duration_ms=step_ms,
            ))
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": "Please respond with either an Action or a Final Answer.",
            })
        else:
            result.predicted_answer = "Reached maximum steps without a final answer."

        result.total_steps = len(result.traces)
        result.total_duration_ms = (time.perf_counter() - t_start) * 1000
        return result


# ---------------------------------------------------------------------------
# Question selection — retrieve-only from the QA file
# ---------------------------------------------------------------------------
def select_questions(qa_pairs: list[dict]) -> list[dict]:
    """Select all questions — the model decides whether to retrieve or answer directly."""
    by_tool = {}
    for q in qa_pairs:
        tool = q["expected_tool"]
        by_tool[tool] = by_tool.get(tool, 0) + 1

    print(f"Selected all {len(qa_pairs)} questions:")
    for tool, count in sorted(by_tool.items()):
        print(f"  {tool}: {count}")

    return qa_pairs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_pass_at_k(n: int, c: int, k: int) -> float:
    if n < k:
        return float(c > 0)
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_pass_pow_k(n: int, c: int, k: int) -> float:
    if n == 0:
        return 0.0
    return (c / n) ** k


def _safe_mean(lst: list[float]) -> float:
    clean = [v for v in lst if not (isinstance(v, float) and math.isnan(v))]
    return sum(clean) / len(clean) if clean else float("nan")


def _fmt(val: float) -> str:
    if isinstance(val, float) and math.isnan(val):
        return "N/A"
    return f"{val:.3f}"


# ---------------------------------------------------------------------------
# CSV column definitions
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "question_id", "run", "question", "expected_tool", "reference_answer",
    "predicted_answer",
    "tool_correct", "used_retrieval", "tools_called",
    "pass_at_k", "pass_pow_k",
    "total_steps", "latency_ms",
    "retrieved_sources", "actual_context",
]

STEPS_CSV_COLUMNS = [
    "question_id", "run", "step",
    "thought", "tool_called", "action_input", "observation",
    "generation_ms", "generation_device",
    "tool_ms", "tool_device",
    "step_ms",
]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark(
    qa_pairs: list[dict],
    model_name: str = "qwen3-8b",
    device: str = "cuda",
    gpu_ids: list[int] | None = None,
    k: int = 5,
    max_steps: int = 5,
) -> dict:
    """Run the full benchmark and save raw outputs for later judging."""
    print(f"\n{'='*60}")
    print(f"ReAct RAG LLM Benchmark")
    print(f"  Model: {model_name}")
    print(f"  Questions: {len(qa_pairs)}, k={k}, device={device}")
    gpu_label = f"GPU {gpu_ids}" if gpu_ids else device
    print(f"  Compute: {gpu_label}")
    print(f"{'='*60}\n")

    agent = BenchmarkAgent(
        model_name=model_name, device=device,
        gpu_ids=gpu_ids, max_steps=max_steps,
    )

    csv_rows: list[dict] = []
    step_rows: list[dict] = []
    all_results: list[dict] = []

    for idx, qa in enumerate(qa_pairs):
        question = qa["question"]
        question_id = qa.get("id", idx + 1)
        reference_answer = qa["reference_answer"]
        expected_tool = qa["expected_tool"]
        ref_context = qa.get("reference_context", "")
        metadata = qa.get("metadata", {})

        print(f"\n[{idx+1}/{len(qa_pairs)}] (id={question_id}) {question}")
        print(f"  Expected tool: {expected_tool}")

        correct_tool_count = 0
        retrieval_used_count = 0
        question_runs = []

        for run_i in range(k):
            result = agent.run_traced(question)
            result.reference_answer = reference_answer
            result.expected_tool = expected_tool
            result.reference_context = ref_context

            tools_called = [t.tool_called for t in result.traces if t.tool_called]
            tool_correct = expected_tool in tools_called
            if tool_correct:
                correct_tool_count += 1
            if result.used_retrieval:
                retrieval_used_count += 1

            actual_context = result.retrieved_context
            if not actual_context:
                for trace in result.traces:
                    if trace.tool_called and trace.observation \
                            and not trace.observation.startswith("["):
                        actual_context = trace.observation
                        break

            run_data = {
                "run": run_i + 1,
                "predicted_answer": result.predicted_answer,
                "tools_called": tools_called,
                "tool_correct": tool_correct,
                "used_retrieval": result.used_retrieval,
                "total_steps": result.total_steps,
                "latency_ms": result.total_duration_ms,
                "retrieved_sources": result.retrieved_sources,
                "retrieved_context": result.retrieved_context,
                "actual_context": actual_context,
                "traces": [asdict(t) for t in result.traces],
            }
            question_runs.append(run_data)

            for trace in result.traces:
                step_rows.append({
                    "question_id": question_id,
                    "run": run_i + 1,
                    "step": trace.step,
                    "thought": trace.thought or "",
                    "tool_called": trace.tool_called or "",
                    "action_input": trace.action_input or "",
                    "observation": trace.observation or "",
                    "generation_ms": round(trace.generation_ms, 1),
                    "generation_device": trace.generation_device,
                    "tool_ms": round(trace.tool_ms, 1),
                    "tool_device": trace.tool_device,
                    "step_ms": round(trace.duration_ms, 1),
                })

            print(f"  Run {run_i+1}/{k}: tool_correct={tool_correct}  "
                  f"used_retrieval={result.used_retrieval}  "
                  f"steps={result.total_steps}  latency={result.total_duration_ms:.0f}ms")

        pass_at_k = compute_pass_at_k(k, correct_tool_count, k)
        pass_pow_k = compute_pass_pow_k(k, correct_tool_count, k)

        all_results.append({
            "question_id": question_id,
            "question": question,
            "reference_answer": reference_answer,
            "expected_tool": expected_tool,
            "reference_context": ref_context,
            "metadata": metadata,
            "runs": question_runs,
            "tool_correct_count": correct_tool_count,
            "retrieval_used_count": retrieval_used_count,
            "tool_total_runs": k,
            "pass_at_k": pass_at_k,
            "pass_pow_k": pass_pow_k,
        })

        print(f"  Summary: {correct_tool_count}/{k} correct  "
              f"retrieval_used={retrieval_used_count}/{k}  "
              f"Pass@{k}={pass_at_k:.3f}  Pass^{k}={pass_pow_k:.3f}")

        for run_data in question_runs:
            csv_rows.append({
                "question_id": question_id,
                "run": run_data["run"],
                "question": question,
                "expected_tool": expected_tool,
                "reference_answer": reference_answer,
                "predicted_answer": run_data["predicted_answer"],
                "tool_correct": int(run_data["tool_correct"]),
                "used_retrieval": int(run_data["used_retrieval"]),
                "tools_called": ";".join(run_data["tools_called"]),
                "pass_at_k": pass_at_k,
                "pass_pow_k": pass_pow_k,
                "total_steps": run_data["total_steps"],
                "latency_ms": round(run_data["latency_ms"], 1),
                "retrieved_sources": ";".join(run_data["retrieved_sources"]),
                "actual_context": run_data["actual_context"],
            })

    summary = _compute_summary(all_results, k)

    # Save CSV outputs
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    tag = f"{model_name}_{device}_{today}"

    csv_path = DATASET_DIR / f"react_llm_raw_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_rows)

    steps_csv_path = DATASET_DIR / f"react_llm_steps_{tag}.csv"
    with open(steps_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STEPS_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(step_rows)

    print(f"\nCSV raw runs saved to {csv_path}")
    print(f"CSV step traces saved to {steps_csv_path}")

    _print_summary(summary, k, model_name)
    return {
        "summary": summary,
        "csv_path": str(csv_path),
        "steps_csv_path": str(steps_csv_path),
    }


def _compute_summary(results: list[dict], k: int) -> dict:
    pass_at_ks = [r["pass_at_k"] for r in results]
    pass_pow_ks = [r["pass_pow_k"] for r in results]

    latencies = [rd["latency_ms"] for r in results for rd in r["runs"]]
    steps = [rd["total_steps"] for r in results for rd in r["runs"]]

    # Retrieval usage rate
    total_runs = sum(r["tool_total_runs"] for r in results)
    retrieval_runs = sum(r["retrieval_used_count"] for r in results)
    retrieval_rate = retrieval_runs / total_runs if total_runs > 0 else 0.0

    # Direct answer rate (answered without any tool)
    direct_runs = sum(
        1 for r in results for rd in r["runs"]
        if not rd["used_retrieval"]
    )
    direct_rate = direct_runs / total_runs if total_runs > 0 else 0.0

    return {
        "tool_call": {
            f"pass_at_{k}": _safe_mean(pass_at_ks),
            f"pass_pow_{k}": _safe_mean(pass_pow_ks),
            "retrieval_usage_rate": retrieval_rate,
            "direct_answer_rate": direct_rate,
        },
        "e2e_performance": {
            "avg_latency_ms": _safe_mean(latencies),
            "min_latency_ms": min(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "avg_steps": _safe_mean(steps),
            "min_steps": min(steps) if steps else 0,
            "max_steps": max(steps) if steps else 0,
        },
    }


def _print_summary(summary: dict, k: int, model_name: str):
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS — {model_name}")
    print(f"{'='*60}")

    tc = summary["tool_call"]
    print(f"\n--- Tool Call Metrics ---")
    print(f"  Pass@{k}:              {_fmt(tc[f'pass_at_{k}'])}")
    print(f"  Pass^{k}:              {_fmt(tc[f'pass_pow_{k}'])}")
    print(f"  Retrieval Usage Rate: {_fmt(tc['retrieval_usage_rate'])}")
    print(f"  Direct Answer Rate:   {_fmt(tc['direct_answer_rate'])}")

    pf = summary["e2e_performance"]
    print(f"\n--- E2E Performance ---")
    print(f"  Avg Latency:          {_fmt(pf['avg_latency_ms'])} ms")
    print(f"  Min/Max Latency:      {_fmt(pf['min_latency_ms'])} / {_fmt(pf['max_latency_ms'])} ms")
    print(f"  Avg Steps:            {_fmt(pf['avg_steps'])}")
    print(f"  Min/Max Steps:        {pf['min_steps']} / {pf['max_steps']}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ReAct RAG LLM Benchmark — retrieval-only ReAct agent",
    )
    parser.add_argument("--model", default="qwen3-8b",
                        help="Model preset name or HF model ID (default: qwen3-8b)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device (default: cuda)")
    parser.add_argument("--gpu", default=None,
                        help="GPU index or comma-separated indices (e.g. '0' or '0,1')")
    parser.add_argument("--qa-file", default=None,
                        help=f"Path to QA JSON file (default: {DEFAULT_QA_FILE})")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of runs per question (default: 5)")
    parser.add_argument("--max-steps", type=int, default=5,
                        help="Max ReAct steps per question (default: 5)")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode — ask questions one at a time")
    parser.add_argument("question", nargs="?", default=None,
                        help="Single question to answer (skip benchmark)")
    args = parser.parse_args()

    gpu_ids = None
    if args.gpu is not None:
        gpu_ids = [int(x.strip()) for x in args.gpu.split(",")]

    # Interactive or single-question mode
    if args.interactive or args.question:
        agent = ReActRetrievalAgent(
            model_name=args.model, device=args.device,
            gpu_ids=gpu_ids, max_steps=args.max_steps,
        )

        if args.interactive:
            print("\nReAct Retrieval Agent (type 'quit' to exit)")
            print("-" * 40)
            while True:
                try:
                    q = input("\nYou: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nBye!")
                    break
                if q.lower() in ("quit", "exit", "q"):
                    break
                if not q:
                    continue
                answer = agent.run(q)
                print(f"\nAnswer: {answer}")
        else:
            answer = agent.run(args.question)
            print(f"\nAnswer: {answer}")
        return

    # Benchmark mode
    qa_path = Path(args.qa_file) if args.qa_file else DEFAULT_QA_FILE
    if not qa_path.exists():
        print(f"Error: QA file not found at {qa_path}")
        print("Run qa_generation.py first to generate ground-truth QA pairs.")
        return

    all_qa = json.loads(qa_path.read_text())
    print(f"Loaded {len(all_qa)} QA pairs from {qa_path}")

    qa_pairs = select_questions(all_qa)

    run_benchmark(
        qa_pairs=qa_pairs,
        model_name=args.model,
        device=args.device,
        gpu_ids=gpu_ids,
        k=args.k,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
