"""
ReAct RAG Benchmark
===================
Runs the ReAct benchmark and saves raw outputs for later judging.

This script focuses on execution and tracing only:
  A. Raw run outputs     — predicted answers, tools called, retrieved context
  B. Step traces         — thought/action/observation for each step
  C. Tool Call Metrics   — Pass@k, Pass^k
  D. E2E Performance     — Latency, Steps-to-answer

Loads ground-truth from agentic_rag_gt.json, uses all retrieve
questions + all fred_releases and web_search questions, runs each 5 times.

Judging is handled separately by llm_judge.py.

Usage:
  python react_rag_benchmark.py                       # run (cuda)
  python react_rag_benchmark.py --device cpu           # run on cpu
  python react_rag_benchmark.py --device cuda          # run on gpu
  python react_rag_benchmark.py --qa-file path.json    # use specific QA file
  python react_rag_benchmark.py --k 5                  # runs per question
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
DEFAULT_QA_FILE = DATASET_DIR / "agentic_rag_gt.json"


# ---------------------------------------------------------------------------
# Data classes for tracing
# ---------------------------------------------------------------------------

def _gpu_snapshot() -> int | None:
    """Capture current GPU memory baseline; returns None when CUDA is unavailable."""
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated()


def _detect_device(before: int | None) -> str:
    """Compare peak GPU memory since snapshot to detect GPU activity."""
    if before is None:
        return "CPU"
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return "GPU" if peak > before else "CPU"


@dataclass
class StepTrace:
    step: int
    thought: str              # raw model response text for this step
    tool_called: str | None
    action_input: str | None
    observation: str | None
    tool_ok: bool | None      # True if tool ran without "Error:"; None if no tool call
    generation_ms: float      # LLM generation time (router)
    tool_ms: float            # tool execution time
    generation_device: str    # detected at runtime via CUDA memory
    tool_device: str          # detected at runtime via CUDA memory
    judge_ms: float           # judge model inference time
    judge_device: str         # detected at runtime via CUDA memory
    judge_decision: str       # "SUFFICIENT", "CONTINUE", or "N/A"
    duration_ms: float        # total step time
    # Token counts (router)
    router_prompt_tokens: int
    router_output_tokens: int
    router_thinking_tokens: int   # tokens inside <think>...</think>
    router_truncated: bool        # True if generation hit max_new_tokens
    # Token counts (judge)
    judge_prompt_tokens: int
    judge_output_tokens: int
    judge_thinking_tokens: int
    judge_truncated: bool


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
    # Token aggregates (filled at end of run_traced)
    total_router_output_tokens: int = 0
    total_router_thinking_tokens: int = 0
    total_judge_output_tokens: int = 0
    total_judge_thinking_tokens: int = 0
    inner_router_output_tokens: int = 0   # router tokens in tool-calling steps
    final_router_output_tokens: int = 0   # router tokens in the final-answer step
    router_truncations: int = 0
    judge_truncations: int = 0
    # Tool-call breakdown
    tool_call_attempts: int = 0           # steps with action != None
    tool_call_successes: int = 0          # of those, where tool_ok is True


# ---------------------------------------------------------------------------
# Instrumented Agent — wraps ReActAgent with per-step tracing
# ---------------------------------------------------------------------------
class BenchmarkAgent:
    """Wraps ReActAgent with per-step tracing and timing."""

    def __init__(
        self,
        device: str = "cuda",
        max_steps: int = 5,
        router_model_path: str | None = None,
        judge_model_path: str | None = None,
    ):
        from react_agent import ReActAgent, _get_vectorstore, DEFAULT_ROUTER_MODEL, DEFAULT_JUDGE_MODEL

        kwargs: dict = {"device": device, "max_steps": max_steps}
        if router_model_path is not None:
            kwargs["router_model_path"] = router_model_path
        if judge_model_path is not None:
            kwargs["judge_model_path"] = judge_model_path

        self.agent = ReActAgent(**kwargs)
        self.router_model_name = router_model_path or DEFAULT_ROUTER_MODEL
        self.judge_model_name = judge_model_path or DEFAULT_JUDGE_MODEL
        # Pre-load retrieval embedding model to avoid repeated load warnings
        _get_vectorstore()

    def run_traced(self, question: str) -> QuestionResult:
        from react_agent import TOOLS, build_system_prompt

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

        observations: list[str] = []  # collected tool observations for the judge
        t_start = time.perf_counter()

        for step in range(1, self.agent.max_steps + 1):
            step_start = time.perf_counter()

            # --- Router LLM generation (detect device via CUDA memory) ---
            gen_snap = _gpu_snapshot()
            gen_start = time.perf_counter()
            response, gen_stats = self.agent._generate(messages)
            gen_ms = (time.perf_counter() - gen_start) * 1000
            gen_device = _detect_device(gen_snap)

            # Empty judge stats — only filled when a tool call triggers the judge
            empty_judge_stats = {
                "prompt_tokens": 0, "output_tokens": 0,
                "thinking_tokens": 0, "truncated": False,
            }

            # Parse action first. The model can emit a hallucinated
            # Observation and Final Answer in the same generation as an
            # Action, and we want to execute the real tool instead.
            action, action_input = self.agent._parse_action(response)
            if action is not None:
                # --- Tool execution (detect device via CUDA memory) ---
                tool_snap = _gpu_snapshot()
                tool_start = time.perf_counter()
                tool = TOOLS.get(action.lower())
                if tool is None:
                    observation = f"Error: Unknown tool '{action}'."
                    tool_ok = False
                else:
                    observation = tool["fn"](action_input)
                    tool_ok = not observation.startswith("Error:")
                tool_ms = (time.perf_counter() - tool_start) * 1000
                tool_device = _detect_device(tool_snap)

                # Extract source docs from retrieve observations
                if action.lower() == "retrieve":
                    result.retrieved_context = observation
                    for match in re.finditer(
                        r"--- Document \d+ \(source:\s*(.+?)\) ---", observation
                    ):
                        result.retrieved_sources.append(match.group(1).strip())

                observations.append(observation)

                # --- Judge model: decide SUFFICIENT or CONTINUE ---
                judge_snap = _gpu_snapshot()
                judge_start = time.perf_counter()
                sufficient, judge_response, judge_stats = self.agent._judge_sufficiency(
                    question, observations,
                )
                judge_ms = (time.perf_counter() - judge_start) * 1000
                judge_device = _detect_device(judge_snap)
                judge_decision = "SUFFICIENT" if sufficient else "CONTINUE"

                step_ms = (time.perf_counter() - step_start) * 1000
                result.traces.append(StepTrace(
                    step=step, thought=response,
                    tool_called=action.lower(),
                    action_input=action_input,
                    observation=observation,
                    tool_ok=tool_ok,
                    generation_ms=gen_ms, tool_ms=tool_ms,
                    generation_device=gen_device, tool_device=tool_device,
                    judge_ms=judge_ms, judge_device=judge_device,
                    judge_decision=judge_decision,
                    duration_ms=step_ms,
                    router_prompt_tokens=gen_stats["prompt_tokens"],
                    router_output_tokens=gen_stats["output_tokens"],
                    router_thinking_tokens=gen_stats["thinking_tokens"],
                    router_truncated=gen_stats["truncated"],
                    judge_prompt_tokens=judge_stats["prompt_tokens"],
                    judge_output_tokens=judge_stats["output_tokens"],
                    judge_thinking_tokens=judge_stats["thinking_tokens"],
                    judge_truncated=judge_stats["truncated"],
                ))

                messages.append({"role": "assistant", "content": response})

                if sufficient:
                    # Judge says we have enough — prompt router for Final Answer
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Observation: {observation}\n\n"
                            "The information gathered is sufficient. "
                            "Please provide your Final Answer now."
                        ),
                    })
                else:
                    messages.append({"role": "user", "content": f"Observation: {observation}"})
                continue

            # Check for final answer only if no action was parsed.
            final = self.agent._parse_final_answer(response)
            if final:
                step_ms = (time.perf_counter() - step_start) * 1000
                result.traces.append(StepTrace(
                    step=step, thought=response,
                    tool_called=None, action_input=None,
                    observation="[final answer]",
                    tool_ok=None,
                    generation_ms=gen_ms, tool_ms=0.0,
                    generation_device=gen_device, tool_device="N/A",
                    judge_ms=0.0, judge_device="N/A",
                    judge_decision="N/A",
                    duration_ms=step_ms,
                    router_prompt_tokens=gen_stats["prompt_tokens"],
                    router_output_tokens=gen_stats["output_tokens"],
                    router_thinking_tokens=gen_stats["thinking_tokens"],
                    router_truncated=gen_stats["truncated"],
                    judge_prompt_tokens=empty_judge_stats["prompt_tokens"],
                    judge_output_tokens=empty_judge_stats["output_tokens"],
                    judge_thinking_tokens=empty_judge_stats["thinking_tokens"],
                    judge_truncated=empty_judge_stats["truncated"],
                ))
                result.predicted_answer = final
                break

            step_ms = (time.perf_counter() - step_start) * 1000
            result.traces.append(StepTrace(
                step=step, thought=response,
                tool_called=None, action_input=None,
                observation="[format error]",
                tool_ok=None,
                generation_ms=gen_ms, tool_ms=0.0,
                generation_device=gen_device, tool_device="N/A",
                judge_ms=0.0, judge_device="N/A",
                judge_decision="N/A",
                duration_ms=step_ms,
                router_prompt_tokens=gen_stats["prompt_tokens"],
                router_output_tokens=gen_stats["output_tokens"],
                router_thinking_tokens=gen_stats["thinking_tokens"],
                router_truncated=gen_stats["truncated"],
                judge_prompt_tokens=empty_judge_stats["prompt_tokens"],
                judge_output_tokens=empty_judge_stats["output_tokens"],
                judge_thinking_tokens=empty_judge_stats["thinking_tokens"],
                judge_truncated=empty_judge_stats["truncated"],
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

        # Aggregate token + tool-call metrics
        result.total_router_output_tokens = sum(t.router_output_tokens for t in result.traces)
        result.total_router_thinking_tokens = sum(t.router_thinking_tokens for t in result.traces)
        result.total_judge_output_tokens = sum(t.judge_output_tokens for t in result.traces)
        result.total_judge_thinking_tokens = sum(t.judge_thinking_tokens for t in result.traces)
        result.inner_router_output_tokens = sum(
            t.router_output_tokens for t in result.traces if t.tool_called is not None
        )
        final_traces = [t for t in result.traces if t.observation == "[final answer]"]
        if final_traces:
            result.final_router_output_tokens = final_traces[-1].router_output_tokens
        result.router_truncations = sum(1 for t in result.traces if t.router_truncated)
        result.judge_truncations = sum(1 for t in result.traces if t.judge_truncated)

        tool_steps = [t for t in result.traces if t.tool_called is not None]
        result.tool_call_attempts = len(tool_steps)
        result.tool_call_successes = sum(1 for t in tool_steps if t.tool_ok)

        return result


# ---------------------------------------------------------------------------
# Question Selection
# ---------------------------------------------------------------------------
def select_questions(qa_pairs: list[dict], num_retrieve: int = 5) -> list[dict]:
    """Select all retrieve questions and all non-retrieve questions."""
    retrieve_qs = [q for q in qa_pairs if q["expected_tool"] == "retrieve"]
    other_qs = [q for q in qa_pairs if q["expected_tool"] != "retrieve"]

    selected = retrieve_qs + other_qs

    print(f"Selected {len(selected)} questions:")
    print(f"  retrieve:      {len(retrieve_qs)} (all)")
    for tool in sorted(set(q["expected_tool"] for q in other_qs)):
        count = sum(1 for q in other_qs if q["expected_tool"] == tool)
        print(f"  {tool}: {count} (all)")

    return selected


# ---------------------------------------------------------------------------
# Tool Call Metrics
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


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "question_id", "run", "question", "expected_tool", "reference_answer",
    "predicted_answer",
    # Tool call (per-run)
    "tool_correct", "tool_exec_correct", "tools_called",
    "tool_call_attempts", "tool_call_successes",
    # Tool call (per-question, same across runs)
    "pass_at_k", "pass_pow_k",
    # E2E performance
    "total_steps", "latency_ms",
    # Token usage (per-run aggregates)
    "total_router_tokens", "inner_router_tokens", "final_router_tokens",
    "total_router_thinking_tokens",
    "total_judge_tokens", "total_judge_thinking_tokens",
    "router_truncations", "judge_truncations",
    # Extra
    "retrieved_sources",
    "actual_context",
]

STEPS_CSV_COLUMNS = [
    "question_id", "run", "step",
    "thought", "tool_called", "action_input", "observation", "tool_ok",
    "generation_ms", "generation_device",
    "router_prompt_tokens", "router_output_tokens",
    "router_thinking_tokens", "router_truncated",
    "tool_ms", "tool_device",
    "judge_ms", "judge_device", "judge_decision",
    "judge_prompt_tokens", "judge_output_tokens",
    "judge_thinking_tokens", "judge_truncated",
    "step_ms",
]


def run_benchmark(
    qa_pairs: list[dict],
    device: str = "cuda",
    k: int = 5,
    max_steps: int = 5,
    router_model_path: str | None = None,
    judge_model_path: str | None = None,
) -> dict:
    """Run the full benchmark and return raw results for later judging."""
    print(f"\n{'='*60}")
    print(f"ReAct RAG Benchmark — {len(qa_pairs)} questions, k={k}, device={device}")
    if router_model_path:
        print(f"  Router model: {router_model_path}")
    if judge_model_path:
        print(f"  Judge model:  {judge_model_path}")
    print(f"{'='*60}\n")

    agent = BenchmarkAgent(
        device=device, max_steps=max_steps,
        router_model_path=router_model_path,
        judge_model_path=judge_model_path,
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
        question_runs = []

        for run_i in range(k):
            result = agent.run_traced(question)
            result.reference_answer = reference_answer
            result.expected_tool = expected_tool
            result.reference_context = ref_context

            tools_called = [t.tool_called for t in result.traces if t.tool_called]
            tool_correct = expected_tool in tools_called
            tool_exec_correct = any(
                t.tool_called == expected_tool and t.tool_ok
                for t in result.traces
            )
            if tool_correct:
                correct_tool_count += 1

            # --- Determine the context the agent actually used ---
            # For retrieve: the actual documents retrieved by the agent.
            # For fred_releases/web_search: the first non-placeholder tool observation.
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
                "tool_exec_correct": tool_exec_correct,
                "tool_call_attempts": result.tool_call_attempts,
                "tool_call_successes": result.tool_call_successes,
                "total_steps": result.total_steps,
                "latency_ms": result.total_duration_ms,
                "total_router_tokens": result.total_router_output_tokens,
                "inner_router_tokens": result.inner_router_output_tokens,
                "final_router_tokens": result.final_router_output_tokens,
                "total_router_thinking_tokens": result.total_router_thinking_tokens,
                "total_judge_tokens": result.total_judge_output_tokens,
                "total_judge_thinking_tokens": result.total_judge_thinking_tokens,
                "router_truncations": result.router_truncations,
                "judge_truncations": result.judge_truncations,
                "retrieved_sources": result.retrieved_sources,
                "retrieved_context": result.retrieved_context,
                "actual_context": actual_context,
                "traces": [asdict(t) for t in result.traces],
            }
            question_runs.append(run_data)

            # Collect step-level rows for steps CSV
            for trace in result.traces:
                step_rows.append({
                    "question_id": question_id,
                    "run": run_i + 1,
                    "step": trace.step,
                    "thought": trace.thought or "",
                    "tool_called": trace.tool_called or "",
                    "action_input": trace.action_input or "",
                    "observation": trace.observation or "",
                    "tool_ok": "" if trace.tool_ok is None else int(bool(trace.tool_ok)),
                    "generation_ms": round(trace.generation_ms, 1),
                    "generation_device": trace.generation_device,
                    "router_prompt_tokens": trace.router_prompt_tokens,
                    "router_output_tokens": trace.router_output_tokens,
                    "router_thinking_tokens": trace.router_thinking_tokens,
                    "router_truncated": int(trace.router_truncated),
                    "tool_ms": round(trace.tool_ms, 1),
                    "tool_device": trace.tool_device,
                    "judge_ms": round(trace.judge_ms, 1),
                    "judge_device": trace.judge_device,
                    "judge_decision": trace.judge_decision,
                    "judge_prompt_tokens": trace.judge_prompt_tokens,
                    "judge_output_tokens": trace.judge_output_tokens,
                    "judge_thinking_tokens": trace.judge_thinking_tokens,
                    "judge_truncated": int(trace.judge_truncated),
                    "step_ms": round(trace.duration_ms, 1),
                })

            print(
                f"  Run {run_i+1}/{k}: tool_correct={tool_correct} "
                f"exec_ok={tool_exec_correct} "
                f"steps={result.total_steps} "
                f"latency={result.total_duration_ms:.0f}ms "
                f"tokens(router/judge)={result.total_router_output_tokens}/"
                f"{result.total_judge_output_tokens} "
                f"inner={result.inner_router_output_tokens} "
                f"final={result.final_router_output_tokens}"
            )

        # Pass@k and Pass^k
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
            "tool_total_runs": k,
            "pass_at_k": pass_at_k,
            "pass_pow_k": pass_pow_k,
        })

        print(f"  Summary: {correct_tool_count}/{k} correct  "
              f"Pass@{k}={pass_at_k:.3f}  Pass^{k}={pass_pow_k:.3f}")

        # Build CSV rows
        for run_data in question_runs:
            csv_rows.append({
                "question_id": question_id,
                "run": run_data["run"],
                "question": question,
                "expected_tool": expected_tool,
                "reference_answer": reference_answer,
                "predicted_answer": run_data["predicted_answer"],
                "tool_correct": int(run_data["tool_correct"]),
                "tool_exec_correct": int(run_data["tool_exec_correct"]),
                "tools_called": ";".join(run_data["tools_called"]),
                "tool_call_attempts": run_data["tool_call_attempts"],
                "tool_call_successes": run_data["tool_call_successes"],
                "pass_at_k": pass_at_k,
                "pass_pow_k": pass_pow_k,
                "total_steps": run_data["total_steps"],
                "latency_ms": round(run_data["latency_ms"], 1),
                "total_router_tokens": run_data["total_router_tokens"],
                "inner_router_tokens": run_data["inner_router_tokens"],
                "final_router_tokens": run_data["final_router_tokens"],
                "total_router_thinking_tokens": run_data["total_router_thinking_tokens"],
                "total_judge_tokens": run_data["total_judge_tokens"],
                "total_judge_thinking_tokens": run_data["total_judge_thinking_tokens"],
                "router_truncations": run_data["router_truncations"],
                "judge_truncations": run_data["judge_truncations"],
                "retrieved_sources": ";".join(run_data["retrieved_sources"]),
                "actual_context": run_data["actual_context"],
            })

    # Aggregate summary
    summary = _compute_summary(all_results, k)

    # Save CSV files only
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    # Sanitize model names for filenames (strip paths, replace separators)
    router_tag = Path(agent.router_model_name).name.replace("/", "-")
    judge_tag = Path(agent.judge_model_name).name.replace("/", "-")

    csv_path = DATASET_DIR / f"react_agent_benchmark_raw_{router_tag}_{judge_tag}_{today}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_rows)

    steps_csv_path = DATASET_DIR / f"react_agent_benchmark_step_{router_tag}_{judge_tag}_{today}.csv"
    with open(steps_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STEPS_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(step_rows)

    print(f"\nCSV raw runs saved to {csv_path}")
    print(f"CSV step traces saved to {steps_csv_path}")

    _print_summary(summary, k)
    return {
        "summary": summary,
        "csv_path": str(csv_path),
        "steps_csv_path": str(steps_csv_path),
    }


def _fmt(val: float) -> str:
    if isinstance(val, float) and math.isnan(val):
        return "N/A"
    return f"{val:.3f}"


def _safe_mean(lst: list[float]) -> float:
    clean = [v for v in lst if not (isinstance(v, float) and math.isnan(v))]
    return sum(clean) / len(clean) if clean else float("nan")


def _compute_summary(results: list[dict], k: int) -> dict:
    # Tool call
    pass_at_ks = [r["pass_at_k"] for r in results]
    pass_pow_ks = [r["pass_pow_k"] for r in results]

    # E2E performance (all runs)
    latencies = [rd["latency_ms"] for r in results for rd in r["runs"]]
    steps = [rd["total_steps"] for r in results for rd in r["runs"]]

    # Tool execution success (per-run)
    tool_exec_corrects = [int(rd["tool_exec_correct"]) for r in results for rd in r["runs"]]
    tool_attempts = [rd["tool_call_attempts"] for r in results for rd in r["runs"]]
    tool_successes = [rd["tool_call_successes"] for r in results for rd in r["runs"]]
    sum_attempts = sum(tool_attempts)
    sum_successes = sum(tool_successes)

    # Tokens (per-run)
    total_router = [rd["total_router_tokens"] for r in results for rd in r["runs"]]
    inner_router = [rd["inner_router_tokens"] for r in results for rd in r["runs"]]
    final_router = [rd["final_router_tokens"] for r in results for rd in r["runs"]]
    router_thinking = [rd["total_router_thinking_tokens"] for r in results for rd in r["runs"]]
    total_judge = [rd["total_judge_tokens"] for r in results for rd in r["runs"]]
    judge_thinking = [rd["total_judge_thinking_tokens"] for r in results for rd in r["runs"]]
    router_trunc = [rd["router_truncations"] for r in results for rd in r["runs"]]
    judge_trunc = [rd["judge_truncations"] for r in results for rd in r["runs"]]

    def _trunc_rate(counts: list[int]) -> float:
        return _safe_mean([float(c > 0) for c in counts])

    return {
        "tool_call": {
            f"pass_at_{k}": _safe_mean(pass_at_ks),
            f"pass_pow_{k}": _safe_mean(pass_pow_ks),
            "tool_exec_correct_rate": _safe_mean(tool_exec_corrects),
            "per_call_success_rate": (
                sum_successes / sum_attempts if sum_attempts else float("nan")
            ),
            "avg_tool_attempts_per_run": _safe_mean(tool_attempts),
        },
        "e2e_performance": {
            "avg_latency_ms": _safe_mean(latencies),
            "min_latency_ms": min(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "avg_steps": _safe_mean(steps),
            "min_steps": min(steps) if steps else 0,
            "max_steps": max(steps) if steps else 0,
        },
        "tokens": {
            "avg_router_total": _safe_mean(total_router),
            "avg_router_inner": _safe_mean(inner_router),
            "avg_router_final": _safe_mean(final_router),
            "avg_router_thinking": _safe_mean(router_thinking),
            "avg_judge_total": _safe_mean(total_judge),
            "avg_judge_thinking": _safe_mean(judge_thinking),
            "router_truncation_rate": _trunc_rate(router_trunc),
            "judge_truncation_rate": _trunc_rate(judge_trunc),
        },
    }


def _print_summary(summary: dict, k: int):
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*60}")

    tc = summary["tool_call"]
    print(f"\n--- Tool Call Metrics ---")
    print(f"  Pass@{k}:                  {_fmt(tc[f'pass_at_{k}'])}")
    print(f"  Pass^{k}:                  {_fmt(tc[f'pass_pow_{k}'])}")
    print(f"  Tool exec-correct rate:   {_fmt(tc['tool_exec_correct_rate'])}")
    print(f"  Per-call success rate:    {_fmt(tc['per_call_success_rate'])}")
    print(f"  Avg tool attempts/run:    {_fmt(tc['avg_tool_attempts_per_run'])}")

    pf = summary["e2e_performance"]
    print(f"\n--- E2E Performance ---")
    print(f"  Avg Latency:              {_fmt(pf['avg_latency_ms'])} ms")
    print(f"  Min/Max Latency:          {_fmt(pf['min_latency_ms'])} / {_fmt(pf['max_latency_ms'])} ms")
    print(f"  Avg Steps:                {_fmt(pf['avg_steps'])}")
    print(f"  Min/Max Steps:            {pf['min_steps']} / {pf['max_steps']}")

    tk = summary["tokens"]
    print(f"\n--- Token Generation ---")
    print(f"  Avg router tokens/run:    {_fmt(tk['avg_router_total'])}")
    print(f"    inner (tool steps):     {_fmt(tk['avg_router_inner'])}")
    print(f"    final-answer step:      {_fmt(tk['avg_router_final'])}")
    print(f"    inside <think>:         {_fmt(tk['avg_router_thinking'])}")
    print(f"  Avg judge tokens/run:     {_fmt(tk['avg_judge_total'])}")
    print(f"    inside <think>:         {_fmt(tk['avg_judge_thinking'])}")
    print(f"  Router truncation rate:   {_fmt(tk['router_truncation_rate'])}")
    print(f"  Judge  truncation rate:   {_fmt(tk['judge_truncation_rate'])}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ReAct RAG Benchmark")
    parser.add_argument("--qa-file", default=None,
                        help=f"Path to QA JSON file (default: {DEFAULT_QA_FILE})")
    parser.add_argument("--num-retrieve", type=int, default=10,
                        help="Deprecated: retrieve questions are no longer sampled; all are used")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of runs per question (default: 5)")
    parser.add_argument("--max-steps", type=int, default=5,
                        help="Max ReAct steps per question (default: 5)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device to run on (default: cuda)")
    parser.add_argument("--router-model", default=None,
                        help="Path to router model (default: Qwen3-8B)")
    parser.add_argument("--judge-model", default=None,
                        help="Path to judge model (default: Qwen3-14B)")
    args = parser.parse_args()

    qa_path = Path(args.qa_file) if args.qa_file else DEFAULT_QA_FILE
    if not qa_path.exists():
        print(f"Error: QA file not found at {qa_path}")
        print("Run qa_generation.py first to generate ground-truth QA pairs.")
        return

    all_qa = json.loads(qa_path.read_text())
    print(f"Loaded {len(all_qa)} QA pairs from {qa_path}")

    # Select questions: all retrieve + all others
    qa_pairs = select_questions(all_qa, num_retrieve=args.num_retrieve)

    run_benchmark(
        qa_pairs=qa_pairs,
        device=args.device,
        k=args.k,
        max_steps=args.max_steps,
        router_model_path=args.router_model,
        judge_model_path=args.judge_model,
    )


if __name__ == "__main__":
    main()
