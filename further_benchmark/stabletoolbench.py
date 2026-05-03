"""
StableToolBench Tool-Calling Benchmark
======================================
Tool-call efficiency benchmark on the StableToolBench dataset
(https://github.com/THUNLP-MT/StableToolBench). The router LLM is given the
RapidAPI-style tool specs that ship with each query and must decide which
sequence of tool calls (if any) answers the user query, then produce a final
answer.

Categories covered (StableToolBench `solvable_queries` test splits):
  - G1_instruction   : single-tool, single-instruction
  - G1_category      : single-tool, intra-category
  - G1_tool          : single-tool, intra-tool
  - G2_category      : multi-tool, intra-category
  - G2_instruction   : multi-tool, multi-instruction
  - G3_instruction   : multi-tool, intra-collection (hardest)

We do NOT spin up the StableToolBench virtual-API server. Instead we mock tool
execution (return a synthetic observation built from each tool's
`template_response`). Scoring is therefore driven by:
  - Tool-execution correctness : did the agent call any "relevant API" set
                                 (per-question gold) for every relevant pair?
                                 (mapped 0..1: covered_relevant / total_relevant)
  - Final-answer correctness   : LLM judge comparing predicted answer to query.
  - Final-answer faithfulness  : LLM judge over (observations → predicted)
                                 — does the answer stick to what tools returned?

Metrics produced (per question, averaged at the end):
  - Tool-execution correctness
  - Tool-call attempts / successes
  - Number of agent loops             (# router steps until Final Answer)
  - End-to-end latency (ms)
  - Router inner-token length         (tokens in tool-calling steps)
  - Final-answer correctness          (1-5, normalized 0..1)
  - Final-answer faithfulness         (1-5, normalized 0..1)
  - Final-answer length (chars / tokens)

Usage:
  # 1. Cache StableToolBench query files (only the first time):
  python stabletoolbench.py --download-only

  # 2. Run the full benchmark (torch GPU router + judge):
  python stabletoolbench.py --categories G1_instruction,G2_category \
                            --num-samples 30 --runs 1

  # 3. OpenVINO router on CPU, torch judge on GPU:
  python stabletoolbench.py --router qwen3-8b-ov-int4 --device cpu \
                            --judge qwen3-14b --judge-gpu 0

  # Single-question debugging:
  python stabletoolbench.py --interactive --category G1_instruction
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Make langGraph + the parent project importable
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "langGraph"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_loader import load_llm, MODEL_PRESETS, MODELS_DIR  # noqa: E402


def _is_openvino_model(model_obj, name_or_id: str) -> bool:
    """Detect whether `load_llm` returned an OpenVINO model.

    OpenVINO models from `optimum.intel` are NOT torch.nn.Modules, so they
    have no `.parameters()` and inputs must stay on CPU.
    """
    preset = MODEL_PRESETS.get(name_or_id, {})
    if preset.get("type") == "llm-openvino":
        return True
    cls_module = type(model_obj).__module__
    if "optimum" in cls_module and "openvino" in cls_module:
        return True
    local_path = MODELS_DIR / name_or_id.replace("/", "--")
    if (local_path / "openvino_model.xml").exists():
        return True
    return False


# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------
DATASET_DIR = Path(__file__).resolve().parent / "dataset" / "stabletoolbench"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "stabletoolbench"

DEFAULT_ROUTER_MODEL = "qwen3-8b"
DEFAULT_JUDGE_MODEL = "qwen3-14b"

# Source for the StableToolBench `solvable_queries/test_instruction` JSON files.
# Override with --data-base if you mirror the repo elsewhere.
STB_DATA_BASE = (
    "https://raw.githubusercontent.com/THUNLP-MT/StableToolBench/master/"
    "solvable_queries/test_instruction"
)

CATEGORY_FILES: dict[str, str] = {
    "G1_instruction": "G1_instruction.json",
    "G1_category": "G1_category.json",
    "G1_tool": "G1_tool.json",
    "G2_category": "G2_category.json",
    "G2_instruction": "G2_instruction.json",
    "G3_instruction": "G3_instruction.json",
}


# ---------------------------------------------------------------------------
# Dataset loading / caching
# ---------------------------------------------------------------------------
def _http_download(url: str, dest: Path) -> Path:
    import urllib.error
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"  ↓ {url}")
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "stabletoolbench-bench/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        raise FileNotFoundError(f"{url} → HTTP {e.code}") from e
    dest.write_bytes(data)
    return dest


def ensure_stb_files(
    base_url: str = STB_DATA_BASE, categories: list[str] | None = None,
) -> None:
    cats = categories or list(CATEGORY_FILES.keys())
    print(f"Caching StableToolBench files in {DATASET_DIR} (source: {base_url})")
    failures: list[str] = []
    for cat in cats:
        fname = CATEGORY_FILES[cat]
        try:
            _http_download(f"{base_url}/{fname}", DATASET_DIR / fname)
        except FileNotFoundError as exc:
            failures.append(f"{fname}: {exc}")
    if failures:
        msg = ("Failed to fetch StableToolBench files (check --data-base):\n  - "
               + "\n  - ".join(failures))
        raise SystemExit(msg)


def _read_json_array(path: Path) -> list[dict]:
    """StableToolBench files are JSON arrays."""
    text = path.read_text().strip()
    if text.startswith("["):
        return json.loads(text)
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def load_category(category: str, num_samples: int = 0) -> list[dict]:
    fname = CATEGORY_FILES[category]
    data_path = DATASET_DIR / fname
    if not data_path.exists():
        print(f"  (missing {data_path}, skipping)")
        return []
    items = _read_json_array(data_path)
    for it in items:
        it["__source_file__"] = fname
        it["__category__"] = category
    if num_samples and num_samples > 0:
        items = items[:num_samples]
    return items


# ---------------------------------------------------------------------------
# StableToolBench tool-spec → OpenAI function-spec conversion
# ---------------------------------------------------------------------------
_TYPE_MAP = {
    "STRING": "string", "string": "string",
    "NUMBER": "number", "number": "number",
    "INTEGER": "integer", "integer": "integer",
    "BOOLEAN": "boolean", "boolean": "boolean",
    "ARRAY": "array",   "array": "array",
    "OBJECT": "object", "object": "object",
}


def _normalize_tool_id(tool_name: str, api_name: str) -> str:
    """Build a deterministic, JSON-safe identifier for a (tool, api) pair."""
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", f"{tool_name}__{api_name}").strip("_")
    return cleaned[:80] or "tool"


def _params_to_schema(required: list[dict], optional: list[dict]) -> dict:
    props: dict[str, Any] = {}
    req_names: list[str] = []
    for p in required or []:
        n = p.get("name") or "arg"
        props[n] = {
            "type": _TYPE_MAP.get(p.get("type", "string"), "string"),
            "description": p.get("description", "") or "",
        }
        if "default" in p:
            props[n]["default"] = p["default"]
        req_names.append(n)
    for p in optional or []:
        n = p.get("name") or "arg"
        props[n] = {
            "type": _TYPE_MAP.get(p.get("type", "string"), "string"),
            "description": p.get("description", "") or "",
        }
        if "default" in p:
            props[n]["default"] = p["default"]
    return {"type": "object", "properties": props, "required": req_names}


def stb_tools_to_specs(stb_tools: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    """Convert StableToolBench `available_tools` entries into OpenAI specs.

    Returns:
      specs   : list of OpenAI function specs (name, description, parameters)
      lookup  : map normalized_name -> original {tool_name, api_name, raw}
                so we can score against `relevant APIs` later.
    """
    specs: list[dict] = []
    lookup: dict[str, dict] = {}
    for t in stb_tools or []:
        tool_name = t.get("tool_name") or t.get("toolName") or ""
        api_name = t.get("api_name") or t.get("name") or ""
        if not tool_name and not api_name:
            continue
        norm = _normalize_tool_id(tool_name, api_name)
        desc_bits = [
            t.get("api_description") or t.get("description") or "",
            f"(tool={tool_name})" if tool_name else "",
            f"category={t.get('category_name')}" if t.get("category_name") else "",
        ]
        specs.append({
            "name": norm,
            "description": " ".join(b for b in desc_bits if b).strip()[:600],
            "parameters": _params_to_schema(
                t.get("required_parameters", []),
                t.get("optional_parameters", []),
            ),
        })
        lookup[norm] = {
            "tool_name": tool_name,
            "api_name": api_name,
            "raw": t,
        }
    return specs, lookup


def gold_relevant_pairs(sample: dict) -> list[tuple[str, str]]:
    """Pull `[ [tool_name, api_name], ... ]` from the sample (relevant APIs)."""
    raw = (sample.get("relevant APIs")
           or sample.get("relevant_APIs")
           or sample.get("relevantAPIs")
           or [])
    pairs: list[tuple[str, str]] = []
    for entry in raw:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            pairs.append((str(entry[0]), str(entry[1])))
        elif isinstance(entry, dict):
            pairs.append((str(entry.get("tool_name", "")),
                          str(entry.get("api_name", ""))))
    return pairs


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def _funcs_to_block(funcs: list[dict]) -> str:
    return json.dumps(funcs, indent=2, ensure_ascii=False)


def build_router_system_prompt(funcs: list[dict]) -> str:
    today = datetime.now().strftime("%Y-%m-%d (%A)")
    return f"""You are a tool-calling assistant. Today is {today}.

You have the following tools available (OpenAI function-spec format):

{_funcs_to_block(funcs)}

Each step must be EXACTLY one of:

(1) Tool call (one OR many in parallel):
Thought: <reasoning>
Action: [
  {{"name": "<tool_name>", "arguments": {{"<arg>": <value>, ...}}}},
  ...
]

(2) Final answer:
Thought: <reasoning>
Final Answer: <concise, user-facing answer grounded in the observations>

Rules:
- Always start with a Thought.
- Action must be a JSON list of {{"name", "arguments"}} objects, even for one call.
- Argument values must respect the parameter types in the spec.
- Never write Observation: yourself — observations come back from the tools.
- Only call tools that appear in the spec above.
- When you have enough info, give the Final Answer (short, direct, faithful to observations).
- If none of the tools can help, return `Final Answer: I cannot answer that with the available tools.`
"""


JUDGE_CORRECTNESS_SYS = (
    "You are an impartial evaluator for tool-using QA correctness.\n"
    "Given a USER QUERY and the AGENT'S FINAL ANSWER, score how well the answer "
    "addresses the query intent on a 1-5 scale "
    "(1=irrelevant/wrong, 3=partial, 5=fully correct & complete).\n"
    "Respond with ONLY the integer."
)

JUDGE_FAITHFULNESS_SYS = (
    "You are an impartial evaluator for answer faithfulness.\n"
    "Given the OBSERVATIONS the agent received from its tools and the AGENT'S FINAL ANSWER, "
    "decide whether every factual claim in the answer is supported by the observations.\n"
    "Score 1-5 (1=mostly fabricated, 3=partly supported, 5=every claim is grounded). "
    "If the agent had NO observations and the answer makes no factual claims (e.g. it admits "
    "it cannot answer), score 5.\n"
    "Respond with ONLY the integer."
)


# ---------------------------------------------------------------------------
# Mock tool executor
# ---------------------------------------------------------------------------
def mock_execute(name: str, args: dict, *, lookup: dict[str, dict]) -> str:
    """Return a stub observation derived from the tool's `template_response`.

    StableToolBench ships a `template_response` per API in `available_tools`;
    we surface that (truncated) so the router can chain calls plausibly. If
    the agent invents a tool that isn't in the spec, we say so.
    """
    info = lookup.get(name)
    if not info:
        return f"[error] Unknown tool '{name}'. Pick one from the spec."
    template = info["raw"].get("template_response") or {}
    body = json.dumps(template, default=str, ensure_ascii=False)[:600]
    api_label = f"{info['tool_name']}::{info['api_name']}"
    return f"[mock-{api_label}] args={json.dumps(args, default=str)[:200]} → {body}"


# ---------------------------------------------------------------------------
# Router agent
# ---------------------------------------------------------------------------
class STBRouterAgent:
    def __init__(
        self,
        router_model: str = DEFAULT_ROUTER_MODEL,
        judge_model: str | None = DEFAULT_JUDGE_MODEL,
        device: str = "cuda",
        router_gpu_ids: list[int] | None = None,
        judge_gpu_ids: list[int] | None = None,
        max_steps: int = 8,
        max_new_tokens: int = 2048,
    ):
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens

        print(f"Loading router LLM: {router_model}")
        self.router, self.router_tok = load_llm(
            router_model, device=device, gpu_ids=router_gpu_ids,
        )
        self.router_is_openvino = _is_openvino_model(self.router, router_model)
        if self.router_is_openvino:
            self.router_input_device = "cpu"
            print("  Router runtime: OpenVINO (inputs on CPU)")
        else:
            try:
                self.router_input_device = next(self.router.parameters()).device
            except (StopIteration, AttributeError):
                self.router_input_device = "cpu"
            print(f"  Router runtime: torch ({self.router_input_device})")

        self.use_judge = judge_model is not None
        if self.use_judge:
            print(f"Loading judge LLM: {judge_model}")
            self.judge, self.judge_tok = load_llm(
                judge_model, device=device, gpu_ids=judge_gpu_ids,
            )
            self.judge_is_openvino = _is_openvino_model(self.judge, judge_model)
            if self.judge_is_openvino:
                self.judge_input_device = "cpu"
                print("  Judge runtime: OpenVINO (inputs on CPU)")
            else:
                try:
                    self.judge_input_device = next(self.judge.parameters()).device
                except (StopIteration, AttributeError):
                    self.judge_input_device = "cpu"
                print(f"  Judge runtime: torch ({self.judge_input_device})")
        else:
            self.judge = self.judge_tok = None
            self.judge_input_device = None
            self.judge_is_openvino = False

    # ---- Generation ---------------------------------------------------
    def _generate_router(self, messages: list[dict]) -> tuple[str, dict]:
        text = self.router_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.router_tok(text, return_tensors="pt").to(self.router_input_device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        with torch.no_grad():
            out = self.router.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        new_ids = out[0][prompt_tokens:]
        out_tokens = int(new_ids.shape[0])
        truncated = out_tokens >= self.max_new_tokens

        raw = self.router_tok.decode(new_ids, skip_special_tokens=False)
        thinks = re.findall(r"<think>(.*?)</think>", raw, re.DOTALL)
        thinking_tokens = sum(
            len(self.router_tok.encode(b, add_special_tokens=False)) for b in thinks
        )
        decoded = self.router_tok.decode(new_ids, skip_special_tokens=True).strip()
        return decoded, {
            "prompt_tokens": prompt_tokens,
            "output_tokens": out_tokens,
            "thinking_tokens": thinking_tokens,
            "truncated": truncated,
        }

    def _generate_judge(self, messages, max_new_tokens: int = 16) -> str:
        if not self.use_judge:
            return ""
        try:
            text = self.judge_tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.judge_tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        inputs = self.judge_tok(text, return_tensors="pt").to(self.judge_input_device)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            out = self.judge.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=0.0,
            )
        decoded = self.judge_tok.decode(
            out[0][prompt_tokens:], skip_special_tokens=True,
        ).strip()
        return re.sub(r"<think>.*?</think>", "", decoded, flags=re.DOTALL).strip()

    def judge_score(self, sys_msg: str, user_msg: str) -> float:
        if not self.use_judge:
            return float("nan")
        out = self._generate_judge(
            [{"role": "system", "content": sys_msg},
             {"role": "user", "content": user_msg}],
        )
        m = re.search(r"\b([1-5])\b", out)
        return float(m.group(1)) if m else float("nan")

    # ---- Parsing ------------------------------------------------------
    @staticmethod
    def parse_actions(text: str) -> list[tuple[str, dict]]:
        m = re.search(r"Action:\s*(\[.*?\]|\{.*?\})\s*(?:\n|$)", text, re.DOTALL)
        if not m:
            return []
        block = m.group(1)
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, dict):
            parsed = [parsed]
        out = []
        for c in parsed:
            if not isinstance(c, dict):
                continue
            name = c.get("name") or c.get("tool")
            args = c.get("arguments") or c.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            if name:
                out.append((name, args))
        return out

    @staticmethod
    def parse_final(text: str) -> str | None:
        m = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
        return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------
@dataclass
class StepTrace:
    step: int
    raw_response: str
    actions: list[dict]
    final: str | None
    router_output_tokens: int
    router_thinking_tokens: int
    truncated: bool
    duration_ms: float
    observations: list[str] = field(default_factory=list)


@dataclass
class QuestionResult:
    qid: str
    category: str
    source_file: str
    question: str
    relevant_pairs: list[tuple[str, str]] = field(default_factory=list)
    allowed_tool_names: set[str] = field(default_factory=set)
    name_to_pair: dict[str, tuple[str, str]] = field(default_factory=dict)

    predicted_answer: str = ""
    traces: list[StepTrace] = field(default_factory=list)

    # Aggregated metrics
    tool_call_attempts: int = 0
    tool_call_successes: int = 0       # parsed AND named tool exists in spec
    wrong_tool_count: int = 0
    n_agent_loops: int = 0             # router steps until Final Answer
    tool_exec_correct: float = float("nan")  # 0..1 coverage of relevant pairs
    answer_correctness: float = float("nan")  # judge 1-5 → 0..1
    answer_faithfulness: float = float("nan")  # judge 1-5 → 0..1
    answer_length_chars: int = 0
    answer_length_tokens: int = 0
    router_truncations: int = 0
    total_router_tokens: int = 0
    inner_router_tokens: int = 0
    final_router_tokens: int = 0
    total_router_thinking_tokens: int = 0
    total_duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Per-question runner
# ---------------------------------------------------------------------------
class STBRunner:
    def __init__(self, agent: STBRouterAgent):
        self.agent = agent

    def _step_loop(
        self,
        messages: list[dict],
        result: QuestionResult,
        *,
        lookup: dict[str, dict],
    ) -> bool:
        for _ in range(1, self.agent.max_steps + 1):
            t0 = time.perf_counter()
            response, stats = self.agent._generate_router(messages)
            ms = (time.perf_counter() - t0) * 1000

            actions = self.agent.parse_actions(response)
            final = self.agent.parse_final(response) if not actions else None

            trace = StepTrace(
                step=len(result.traces) + 1,
                raw_response=response,
                actions=[{"name": n, "arguments": a} for n, a in actions],
                final=final,
                router_output_tokens=stats["output_tokens"],
                router_thinking_tokens=stats["thinking_tokens"],
                truncated=stats["truncated"],
                duration_ms=ms,
            )
            result.traces.append(trace)
            if stats["truncated"]:
                result.router_truncations += 1

            messages.append({"role": "assistant", "content": response})

            if actions:
                result.tool_call_attempts += len(actions)
                obs_lines = []
                for name, args in actions:
                    if name in result.allowed_tool_names:
                        result.tool_call_successes += 1
                    else:
                        result.wrong_tool_count += 1
                    obs = mock_execute(name, args, lookup=lookup)
                    obs_lines.append(f"- {name}: {obs}")
                    trace.observations.append(obs)
                messages.append({
                    "role": "user",
                    "content": "Observation:\n" + "\n".join(obs_lines),
                })
                continue

            if final is not None:
                result.predicted_answer = final
                return True

            messages.append({
                "role": "user",
                "content": "Please respond with either an Action JSON list or a Final Answer.",
            })
        return False

    def run_question(self, sample: dict) -> QuestionResult:
        # StableToolBench files use `api_list`; some forks use `available_tools`.
        stb_tools = sample.get("api_list") or sample.get("available_tools") or []
        specs, lookup = stb_tools_to_specs(stb_tools)
        relevant = gold_relevant_pairs(sample)
        question_text = sample.get("query", "") or ""

        # Map normalized spec names back to (tool_name, api_name) gold pairs.
        name_to_pair = {n: (info["tool_name"], info["api_name"])
                        for n, info in lookup.items()}

        qid = str(sample.get("query_id", sample.get("id", ""))) or f"unk-{id(sample)}"
        result = QuestionResult(
            qid=qid,
            category=sample["__category__"],
            source_file=sample["__source_file__"],
            question=question_text[:300],
            relevant_pairs=relevant,
            allowed_tool_names=set(lookup.keys()),
            name_to_pair=name_to_pair,
        )

        sys_prompt = build_router_system_prompt(specs)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question_text},
        ]

        t_start = time.perf_counter()
        self._step_loop(messages, result, lookup=lookup)
        result.total_duration_ms = (time.perf_counter() - t_start) * 1000

        # ---- Tool execution correctness: relevant-pair coverage ----
        called_pairs = set()
        for tr in result.traces:
            for a in tr.actions:
                pair = name_to_pair.get(a["name"])
                if pair:
                    called_pairs.add(pair)
        if relevant:
            covered = sum(1 for p in relevant if tuple(p) in called_pairs)
            result.tool_exec_correct = covered / len(relevant)
        else:
            result.tool_exec_correct = float("nan")

        # ---- Agent loop count + token aggregates ----
        result.n_agent_loops = len(result.traces)
        result.total_router_tokens = sum(t.router_output_tokens for t in result.traces)
        result.total_router_thinking_tokens = sum(
            t.router_thinking_tokens for t in result.traces
        )
        result.inner_router_tokens = sum(
            t.router_output_tokens for t in result.traces if t.actions
        )
        final_traces = [t for t in result.traces if t.final is not None]
        if final_traces:
            result.final_router_tokens = final_traces[-1].router_output_tokens

        # ---- Answer length ----
        result.answer_length_chars = len(result.predicted_answer)
        try:
            result.answer_length_tokens = len(
                self.agent.router_tok.encode(
                    result.predicted_answer, add_special_tokens=False))
        except Exception:
            result.answer_length_tokens = len(result.predicted_answer.split())

        # ---- LLM judges ----
        if self.agent.use_judge and result.predicted_answer:
            corr = self.agent.judge_score(
                JUDGE_CORRECTNESS_SYS,
                f"USER QUERY:\n{question_text}\n\n"
                f"AGENT'S FINAL ANSWER:\n{result.predicted_answer}",
            )
            result.answer_correctness = (
                (corr - 1) / 4.0 if isinstance(corr, float) and not math.isnan(corr)
                else float("nan")
            )

            obs_blob = "\n".join(
                obs for tr in result.traces for obs in tr.observations
            ) or "(no tool observations)"
            faith = self.agent.judge_score(
                JUDGE_FAITHFULNESS_SYS,
                f"OBSERVATIONS:\n{obs_blob[:4000]}\n\n"
                f"AGENT'S FINAL ANSWER:\n{result.predicted_answer}",
            )
            result.answer_faithfulness = (
                (faith - 1) / 4.0 if isinstance(faith, float) and not math.isnan(faith)
                else float("nan")
            )
        return result


# ---------------------------------------------------------------------------
# CSV / summary
# ---------------------------------------------------------------------------
PER_QUESTION_COLUMNS = [
    "qid", "run", "category", "source_file", "question",
    "predicted_answer",
    "tool_exec_correct",
    "tool_call_attempts", "tool_call_successes", "wrong_tool_count",
    "wrong_tool_rate",
    "n_agent_loops",
    "answer_correctness", "answer_faithfulness",
    "answer_length_chars", "answer_length_tokens",
    "router_truncations",
    "total_router_tokens", "inner_router_tokens", "final_router_tokens",
    "total_router_thinking_tokens",
    "latency_ms",
]

STEP_COLUMNS = [
    "qid", "run", "step", "n_actions", "tool_names",
    "is_final", "truncated",
    "router_output_tokens", "router_thinking_tokens", "duration_ms",
]


def _safe_mean(values) -> float:
    clean = [v for v in values if isinstance(v, (int, float))
             and not (isinstance(v, float) and math.isnan(v))]
    return sum(clean) / len(clean) if clean else float("nan")


def _fmt(v) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "N/A"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def run_benchmark(
    samples_by_cat: dict[str, list[dict]],
    runner: STBRunner,
    output_tag: str,
    runs_per_sample: int = 1,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_q_path = output_dir / f"stb_qaagent_{output_tag}.csv"
    steps_path = output_dir / f"stb_qaagent_steps_{output_tag}.csv"
    summary_path = output_dir / f"stb_qaagent_summary_{output_tag}.csv"

    per_q_rows: list[dict] = []
    step_rows: list[dict] = []

    for cat, samples in samples_by_cat.items():
        print(f"\n=== Category: {cat} ({len(samples)} samples) ===")
        for idx, sample in enumerate(samples, 1):
            qid = str(sample.get("query_id", sample.get("id", idx)))
            print(f"[{cat} {idx}/{len(samples)}] {qid}")
            for run_i in range(1, runs_per_sample + 1):
                result = runner.run_question(sample)

                wrong_tool_rate = (
                    result.wrong_tool_count / result.tool_call_attempts
                    if result.tool_call_attempts else float("nan")
                )
                per_q_rows.append({
                    "qid": qid,
                    "run": run_i,
                    "category": cat,
                    "source_file": result.source_file,
                    "question": result.question,
                    "predicted_answer": result.predicted_answer[:500],
                    "tool_exec_correct": result.tool_exec_correct,
                    "tool_call_attempts": result.tool_call_attempts,
                    "tool_call_successes": result.tool_call_successes,
                    "wrong_tool_count": result.wrong_tool_count,
                    "wrong_tool_rate": wrong_tool_rate,
                    "n_agent_loops": result.n_agent_loops,
                    "answer_correctness": result.answer_correctness,
                    "answer_faithfulness": result.answer_faithfulness,
                    "answer_length_chars": result.answer_length_chars,
                    "answer_length_tokens": result.answer_length_tokens,
                    "router_truncations": result.router_truncations,
                    "total_router_tokens": result.total_router_tokens,
                    "inner_router_tokens": result.inner_router_tokens,
                    "final_router_tokens": result.final_router_tokens,
                    "total_router_thinking_tokens": result.total_router_thinking_tokens,
                    "latency_ms": round(result.total_duration_ms, 1),
                })
                for tr in result.traces:
                    step_rows.append({
                        "qid": qid,
                        "run": run_i,
                        "step": tr.step,
                        "n_actions": len(tr.actions),
                        "tool_names": ";".join(a["name"] for a in tr.actions),
                        "is_final": int(tr.final is not None),
                        "truncated": int(tr.truncated),
                        "router_output_tokens": tr.router_output_tokens,
                        "router_thinking_tokens": tr.router_thinking_tokens,
                        "duration_ms": round(tr.duration_ms, 1),
                    })
                print(
                    f"   → exec={_fmt(result.tool_exec_correct)} "
                    f"loops={result.n_agent_loops} "
                    f"attempts={result.tool_call_attempts} "
                    f"wrong={result.wrong_tool_count} "
                    f"corr={_fmt(result.answer_correctness)} "
                    f"faith={_fmt(result.answer_faithfulness)} "
                    f"len={result.answer_length_tokens}t "
                    f"latency={result.total_duration_ms:.0f}ms"
                )

    with open(per_q_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_QUESTION_COLUMNS)
        w.writeheader()
        w.writerows(per_q_rows)
    with open(steps_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STEP_COLUMNS)
        w.writeheader()
        w.writerows(step_rows)

    def _summarize(rows):
        return {
            "n": len(rows),
            "tool_exec_correct": _safe_mean([r["tool_exec_correct"] for r in rows]),
            "tool_call_attempts": _safe_mean([r["tool_call_attempts"] for r in rows]),
            "tool_call_successes": _safe_mean([r["tool_call_successes"] for r in rows]),
            "wrong_tool_rate": _safe_mean([r["wrong_tool_rate"] for r in rows]),
            "n_agent_loops": _safe_mean([r["n_agent_loops"] for r in rows]),
            "answer_correctness": _safe_mean([r["answer_correctness"] for r in rows]),
            "answer_faithfulness": _safe_mean([r["answer_faithfulness"] for r in rows]),
            "answer_length_tokens": _safe_mean([r["answer_length_tokens"] for r in rows]),
            "answer_length_chars": _safe_mean([r["answer_length_chars"] for r in rows]),
            "inner_router_tokens": _safe_mean([r["inner_router_tokens"] for r in rows]),
            "final_router_tokens": _safe_mean([r["final_router_tokens"] for r in rows]),
            "total_router_tokens": _safe_mean([r["total_router_tokens"] for r in rows]),
            "total_router_thinking_tokens": _safe_mean(
                [r["total_router_thinking_tokens"] for r in rows]),
            "router_truncations": _safe_mean([r["router_truncations"] for r in rows]),
            "latency_ms": _safe_mean([r["latency_ms"] for r in rows]),
        }

    summary_rows = [{"category": "ALL", **_summarize(per_q_rows)}]
    for cat in samples_by_cat:
        summary_rows.append({
            "category": cat,
            **_summarize([r for r in per_q_rows if r["category"] == cat]),
        })

    cols = ["category", "n", "tool_exec_correct", "tool_call_attempts",
            "tool_call_successes", "wrong_tool_rate", "n_agent_loops",
            "answer_correctness", "answer_faithfulness",
            "answer_length_tokens", "answer_length_chars",
            "inner_router_tokens", "final_router_tokens", "total_router_tokens",
            "total_router_thinking_tokens", "router_truncations", "latency_ms"]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\nPer-question CSV : {per_q_path}")
    print(f"Step trace CSV   : {steps_path}")
    print(f"Summary CSV      : {summary_path}")

    print(f"\n{'='*70}\nSTABLETOOLBENCH BENCHMARK SUMMARY\n{'='*70}")
    for row in summary_rows:
        print(f"\n[{row['category']}] (n={row['n']})")
        for k in cols[2:]:
            print(f"  {k:<32}: {_fmt(row[k])}")
    return {"summary_rows": summary_rows, "csv_path": str(per_q_path)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="StableToolBench tool-calling benchmark (router efficiency metrics)",
    )
    parser.add_argument("--router", default=DEFAULT_ROUTER_MODEL)
    parser.add_argument("--judge", default=DEFAULT_JUDGE_MODEL,
                        help="Judge model (correctness + faithfulness). "
                             "Empty string disables.")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"],
                        help="Compute device. Defaults to 'cpu' for OpenVINO "
                             "presets, 'cuda' otherwise.")
    parser.add_argument("--router-gpu", default=None)
    parser.add_argument("--judge-gpu", default=None)
    parser.add_argument("--categories",
                        default="G1_instruction,G2_category,G3_instruction",
                        help=f"Comma-separated list from {list(CATEGORY_FILES)}")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--data-base", default=STB_DATA_BASE,
                        help="Base URL for StableToolBench solvable_queries JSONs.")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--category", default="G1_instruction",
                        help="Category to use in --interactive mode")
    args = parser.parse_args()

    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    for c in cats:
        if c not in CATEGORY_FILES:
            sys.exit(f"Unknown category {c}; choose from {list(CATEGORY_FILES)}")

    ensure_stb_files(args.data_base, cats)
    if args.download_only:
        return

    def parse_gpu(s):
        return [int(x) for x in s.split(",")] if s else None

    def _is_ov_preset(name: str | None) -> bool:
        if not name:
            return False
        preset = MODEL_PRESETS.get(name, {})
        return preset.get("type") == "llm-openvino" or "-ov" in name

    if args.device is None:
        args.device = "cpu" if _is_ov_preset(args.router) else "cuda"
        print(f"Auto-selected --device={args.device} based on --router={args.router}")

    agent = STBRouterAgent(
        router_model=args.router,
        judge_model=args.judge or None,
        device=args.device,
        router_gpu_ids=parse_gpu(args.router_gpu),
        judge_gpu_ids=parse_gpu(args.judge_gpu),
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
    )
    runner = STBRunner(agent)

    if args.interactive:
        cat = args.category
        samples = load_category(cat, num_samples=5)
        if not samples:
            sys.exit(f"No samples found for {cat}; run --download-only first.")
        sample = samples[0]
        qid = sample.get("query_id", sample.get("id", "?"))
        print(f"\nInteractive: {cat} / {qid}")
        print("Q:", sample.get("query", "")[:500])
        result = runner.run_question(sample)
        print("\nPredicted:", result.predicted_answer)
        print(
            f"loops={result.n_agent_loops} "
            f"attempts={result.tool_call_attempts} "
            f"wrong={result.wrong_tool_count} "
            f"exec={_fmt(result.tool_exec_correct)} "
            f"corr={_fmt(result.answer_correctness)} "
            f"faith={_fmt(result.answer_faithfulness)} "
            f"latency={result.total_duration_ms:.0f}ms"
        )
        return

    samples_by_cat = {c: load_category(c, num_samples=args.num_samples) for c in cats}
    today = datetime.now().strftime("%Y-%m-%d")
    tag = f"{args.router}_{args.judge or 'no-judge'}_{today}".replace("/", "-")
    run_benchmark(
        samples_by_cat, runner,
        output_tag=tag,
        runs_per_sample=args.runs,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
