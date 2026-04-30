"""
BFCL v4 Tool-Calling Benchmark
==============================
Tool-call efficiency benchmark on the Berkeley Function Calling Leaderboard v4
dataset. The router LLM is given an OpenAI-style function spec per question and
must emit one or more tool calls (or a Final Answer for irrelevance / web QA).

Categories covered (subset of BFCL v4):
  - single_turn        : simple_*, multiple, parallel, parallel_multiple, irrelevance, live_*
  - multi_turn         : multi_turn_base, long_context, miss_func, miss_param
  - multi_hop_web      : web_search          (multi-hop search → fetch → answer)

For multi-turn we DO NOT spin up the full BFCL stateful API simulator. Instead
we mock tool execution (return a placeholder observation) and score the agent
by the sequence of (tool_name, arguments) it produced against the ground-truth
sequence. This gives us tool-execution-correctness as a path/argument match
rate, which is the metric we care about for *router* efficiency.

Metrics produced (per question, averaged at the end):
  - Tool-execution correctness   : did the agent's call sequence match GT?
                                   (single-turn: AST match; multi-turn: path match;
                                    web: LLM-judged answer correctness)
  - Tool-call attempts           : total calls emitted by the router
  - Tool-call successes          : attempts that parsed AND named a known tool
  - Wrong-tool rate              : attempts whose tool name is not in the spec
                                   (for multi-turn: not in expected GT either)
  - Argument correctness         : fraction of attempts whose args match GT
  - Router truncations           : router generations that hit max_new_tokens
  - Total router tokens          : all router output tokens across the question
  - Inner-router tokens          : router tokens spent in tool-calling steps
  - Final-router tokens          : router tokens in the final-answer / final step
  - Total router thinking tokens : sum of `<think>...</think>` tokens

Usage:
  # 1. Cache BFCL v4 files (only the first time):
  python bfcl_function_call.py --download-only

  # 2. Run the full benchmark on every covered category (torch GPU):
  python bfcl_function_call.py --categories single_turn,multi_turn,multi_hop_web \\
                                --num-samples 50 --runs 1

  # 3. Run with an OpenVINO router on CPU (any --router preset whose name
  #    ends in -ov / -ov-fp16 / -ov-int4 / -ov-int8 routes through
  #    optimum.intel.OVModelForCausalLM via model_loader):
  python bfcl_function_call.py --router qwen3-8b-ov-int4 --device cpu \\
                                --judge qwen3-14b --judge-gpu 0
  # Mix-and-match: OV router on CPU, torch judge on GPU.

  # Single-question debugging:
  python bfcl_function_call.py --interactive --category single_turn

OpenVINO presets available via model_loader (see /langGraph/model_loader.py):
  qwen3-8b-ov, qwen3-8b-ov-fp16, qwen3-8b-ov-int4, qwen3-14b-ov
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, field
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

    OpenVINO models from `optimum.intel` are NOT `torch.nn.Module`s, so they
    have no `.parameters()` and inputs must stay on CPU. Detection paths:
      1. The model_loader preset is tagged `llm-openvino`.
      2. The class lives in optimum.intel.openvino.
      3. A locally-resolved path contains `openvino_model.xml`.
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
DATASET_DIR = Path(__file__).resolve().parent / "dataset" / "bfcl_v4"
GOLD_DIR = DATASET_DIR / "possible_answer"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "bfcl"

DEFAULT_ROUTER_MODEL = "qwen3-8b"
DEFAULT_JUDGE_MODEL = "qwen3-14b"

# Base URL for BFCL v4 data files. The official source is the gorilla GitHub repo;
# `tuandunghcmut/BFCL_v4_information` on HF only ships documentation, not JSONs.
# Override via --data-base if you've mirrored it elsewhere.
BFCL_DATA_BASE = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
    "berkeley-function-call-leaderboard/bfcl_eval/data"
)

# Files we ship per category. Keys = category name used everywhere downstream.
CATEGORY_FILES: dict[str, list[str]] = {
    "single_turn": [
        "BFCL_v4_simple_python.json",
        "BFCL_v4_multiple.json",
        "BFCL_v4_parallel.json",
        "BFCL_v4_parallel_multiple.json",
        "BFCL_v4_irrelevance.json",
        "BFCL_v4_live_simple.json",
        "BFCL_v4_live_multiple.json",
    ],
    "multi_turn": [
        "BFCL_v4_multi_turn_base.json",
        "BFCL_v4_multi_turn_long_context.json",
        "BFCL_v4_multi_turn_miss_func.json",
        "BFCL_v4_multi_turn_miss_param.json",
    ],
    "multi_hop_web": [
        "BFCL_v4_web_search.json",
    ],
}

# `irrelevance` expects ZERO tool calls — handled specially in scoring.
IRRELEVANCE_FILES = {"BFCL_v4_irrelevance.json"}


# ---------------------------------------------------------------------------
# Dataset loading / caching
# ---------------------------------------------------------------------------
def _http_download(url: str, dest: Path) -> Path:
    """Download a single file via HTTP, caching it at `dest`. No-op if already cached."""
    import urllib.error
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"  ↓ {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "bfcl-qaagent/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        raise FileNotFoundError(f"{url} → HTTP {e.code}") from e
    dest.write_bytes(data)
    return dest


def ensure_bfcl_files(
    base_url: str = BFCL_DATA_BASE, categories: list[str] | None = None,
) -> None:
    """Make sure all dataset + ground-truth files for the requested categories exist."""
    cats = categories or list(CATEGORY_FILES.keys())
    print(f"Caching BFCL v4 files in {DATASET_DIR} (source: {base_url})")
    for cat in cats:
        for fname in CATEGORY_FILES[cat]:
            try:
                _http_download(f"{base_url}/{fname}", DATASET_DIR / fname)
            except FileNotFoundError as exc:
                print(f"    (missing data file {fname}: {exc})")
                continue
            # possible_answer/ holds the AST ground truth (not present for web_search).
            try:
                _http_download(
                    f"{base_url}/possible_answer/{fname}", GOLD_DIR / fname,
                )
            except FileNotFoundError as exc:
                print(f"    (no possible_answer for {fname}: {exc})")


def _read_json_array(path: Path) -> list[dict]:
    """BFCL files are JSONL-ish: list of objects, one per line OR a JSON array."""
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
    """Load all samples for `category`, attaching ground-truth + source filename."""
    samples: list[dict] = []
    for fname in CATEGORY_FILES[category]:
        data_path = DATASET_DIR / fname
        gold_path = GOLD_DIR / fname
        if not data_path.exists():
            print(f"  (missing {data_path}, skipping)")
            continue
        items = _read_json_array(data_path)
        gold_by_id: dict[str, dict] = {}
        if gold_path.exists():
            for g in _read_json_array(gold_path):
                gold_by_id[g["id"]] = g
        for it in items:
            it["__source_file__"] = fname
            it["__category__"] = category
            it["__gold__"] = gold_by_id.get(it["id"], {})
        samples.extend(items)

    if num_samples and num_samples > 0:
        # Stratified slice: take first N per file so we don't hit only one split.
        per_file: dict[str, list] = {}
        for s in samples:
            per_file.setdefault(s["__source_file__"], []).append(s)
        cap = max(1, num_samples // max(1, len(per_file)))
        samples = []
        for f, items in per_file.items():
            samples.extend(items[:cap])
    return samples


# ---------------------------------------------------------------------------
# Ground-truth call parsing (BFCL uses Python-call strings)
# ---------------------------------------------------------------------------
def _literal(node: ast.AST) -> Any:
    """Best-effort literal eval that also tolerates bare Names like `True`/`None`."""
    try:
        return ast.literal_eval(node)
    except Exception:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return ast.unparse(node)
        return ast.unparse(node)


def parse_call_string(s: str) -> tuple[str | None, dict[str, Any]] | None:
    """Parse a Python-style call string `fn(a=1, b='x')` → (name, kwargs)."""
    s = (s or "").strip()
    if not s:
        return None
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError:
        return None
    if not isinstance(tree.body, ast.Call):
        return None
    call = tree.body
    if isinstance(call.func, ast.Name):
        name = call.func.id
    elif isinstance(call.func, ast.Attribute):
        name = call.func.attr  # BFCL multi-turn calls are e.g. `FileSystem.cd(...)`
    else:
        return None
    kwargs = {kw.arg: _literal(kw.value) for kw in call.keywords}
    # Positional args are rare in BFCL gold but we keep them under "_args"
    if call.args:
        kwargs["__positional__"] = [_literal(a) for a in call.args]
    return name, kwargs


def parse_call_list(strings: list[str]) -> list[tuple[str, dict]]:
    out = []
    for s in strings or []:
        parsed = parse_call_string(s)
        if parsed is not None and parsed[0]:
            out.append(parsed)
    return out


def parse_gold_single_turn(gold_list: list) -> list[tuple[str, dict]]:
    """Parse single-turn BFCL gold (alts-aware format).

    Format:
      [
        {"<func_name>": {"<arg>": [<accepted_value>, <alt1>, ...], ...}},
        ...   # one dict per expected call (parallel)
      ]
    Returned tuples preserve the inner alts dict verbatim — the matcher below
    knows that each value is a list of accepted alternatives.
    """
    out: list[tuple[str, dict]] = []
    for entry in gold_list or []:
        if isinstance(entry, str):
            # Some splits ship call-strings instead of alts dicts; fall back to AST parse.
            parsed = parse_call_string(entry)
            if parsed is not None:
                # Wrap each scalar arg in a 1-element list so the alts-matcher works uniformly.
                name, kwargs = parsed
                out.append((name, {k: [v] for k, v in kwargs.items()
                                   if k != "__positional__"}))
            continue
        if not isinstance(entry, dict):
            continue
        for name, args in entry.items():
            out.append((name, args or {}))
    return out


# ---------------------------------------------------------------------------
# Call-comparison helpers
# ---------------------------------------------------------------------------
def _norm_value(v: Any) -> Any:
    """Light normalization for arg comparison: case-insensitive strings, float→tuple."""
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, list):
        return tuple(_norm_value(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _norm_value(val)) for k, val in v.items()))
    return v


def args_match(pred: dict, gold: dict) -> bool:
    """Plain equality match — used for multi-turn where gold is a single call-string.

    All gold (key, value) pairs must be present in pred with equal normalized values.
    Pred may set extra optional args.
    """
    pred_n = {k: _norm_value(v) for k, v in pred.items() if k != "__positional__"}
    gold_n = {k: _norm_value(v) for k, v in gold.items() if k != "__positional__"}
    for k, v in gold_n.items():
        if k not in pred_n:
            return False
        if pred_n[k] != v:
            return False
    return True


def _accepts(pred_val: Any, alts: list) -> bool:
    """Does pred_val match any of the accepted alternatives?

    For nested dicts, each leaf in `alt` is itself wrapped in a list of alts
    (BFCL's recursive convention). `""` in alts means the value can also be
    empty / missing (the caller handles the missing-key case before reaching here).
    """
    if not isinstance(alts, list):
        alts = [alts]
    for alt in alts:
        if alt == "" and (pred_val in (None, "", [], {})):
            return True
        if isinstance(alt, dict) and isinstance(pred_val, dict):
            ok = True
            for k, sub_alts in alt.items():
                sub_alts_l = sub_alts if isinstance(sub_alts, list) else [sub_alts]
                if k not in pred_val:
                    if any(a == "" for a in sub_alts_l):
                        continue
                    ok = False
                    break
                if not _accepts(pred_val[k], sub_alts_l):
                    ok = False
                    break
            if ok:
                return True
            continue
        if _norm_value(pred_val) == _norm_value(alt):
            return True
    return False


def args_match_bfcl(pred: dict, gold_alts: dict) -> bool:
    """Single-turn alts-aware match.

    `gold_alts` is `{arg_name: [accepted_values...]}`. Every gold arg must be
    satisfied by `pred`; an arg with `""` among its alts may be omitted.
    """
    pred_n = {k: v for k, v in (pred or {}).items() if k != "__positional__"}
    for k, alts in gold_alts.items():
        alts_l = alts if isinstance(alts, list) else [alts]
        if k not in pred_n:
            if any(a == "" for a in alts_l):
                continue
            return False
        if not _accepts(pred_n[k], alts_l):
            return False
    return True


# ---------------------------------------------------------------------------
# Function-spec resolution
# ---------------------------------------------------------------------------
# BFCL multi_turn_* and web_search files don't ship inline "function" specs —
# they only declare `involved_classes`. The actual class methods live in BFCL's
# Python simulator. Since we don't run the simulator (we mock tool execution),
# we synthesize a minimal spec per question:
#   - multi_turn   : derive method name + arg names from the gold ground_truth
#                    call sequence (we know which calls the agent should make).
#   - web_search   : hard-code the two WebSearchAPI methods our mock executor
#                    already understands (search_engine_query, fetch_url_content).
WEB_SEARCH_FUNCS: list[dict] = [
    {
        "name": "search_engine_query",
        "description": "Run a web search and return a list of results "
                       "(title, url, snippet) for the given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url_content",
        "description": "Fetch and return the textual content of a web page by URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Absolute URL to fetch.",
                },
            },
            "required": ["url"],
        },
    },
]


def _funcs_from_gold(gold_obj: dict, involved_classes: list[str]) -> list[dict]:
    """Build minimal function specs from the gold ground-truth call sequence.

    Each unique gold function name → one synthetic spec. Argument keys are
    pulled from any call to that function in the gold so the model sees the
    parameter set it'll be scored on.
    """
    raw = gold_obj.get("ground_truth") or []
    # Multi-turn ground truth is `[[turn0_calls...], [turn1_calls...], ...]`,
    # web_search ground truth is just answer strings (handled elsewhere).
    flat: list[str] = []
    for item in raw:
        if isinstance(item, list):
            flat.extend(s for s in item if isinstance(s, str))
        elif isinstance(item, str):
            flat.append(item)

    by_name: dict[str, set[str]] = {}
    for s in flat:
        parsed = parse_call_string(s)
        if parsed is None:
            continue
        name, kwargs = parsed
        keys = {k for k in kwargs.keys() if k != "__positional__"}
        by_name.setdefault(name, set()).update(keys)

    classes_blob = ", ".join(involved_classes) if involved_classes else "the simulator"
    specs: list[dict] = []
    for name, keys in by_name.items():
        specs.append({
            "name": name,
            "description": (
                f"Method exposed by {classes_blob}. "
                f"Synthesized from BFCL ground truth — match the parameter set exactly."
            ),
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "string"} for k in sorted(keys)},
                "required": [],
            },
        })
    return specs


def resolve_funcs(sample: dict) -> list[dict]:
    """Return the list of OpenAI-style function specs for `sample`.

    Handles all three BFCL shapes:
      - inline `function` (single_turn / live_*)
      - `involved_classes` only with gold ground truth (multi_turn)
      - `involved_classes` only without per-call gold (web_search)
    """
    if sample.get("function"):
        return sample["function"]

    category = sample.get("__category__", "")
    involved = sample.get("involved_classes", []) or []
    if category == "multi_hop_web" or "WebSearchAPI" in involved:
        return WEB_SEARCH_FUNCS

    gold = sample.get("__gold__", {}) or {}
    specs = _funcs_from_gold(gold, involved)
    if specs:
        return specs

    # Last resort: stub one entry per involved class so the prompt still renders.
    return [
        {
            "name": cls,
            "description": f"Available class API: {cls}.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
        for cls in involved
    ]


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def _funcs_to_block(funcs: list[dict]) -> str:
    """Render the tool spec list as compact JSON for the system prompt."""
    return json.dumps(funcs, indent=2, ensure_ascii=False)


def build_router_system_prompt(
    funcs: list[dict],
    *,
    multi_turn: bool,
    web_search: bool,
    irrelevance_hint: bool,
) -> str:
    today = datetime.now().strftime("%Y-%m-%d (%A)")
    base = f"""You are a tool-calling assistant. Today is {today}.

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
Final Answer: <short answer or "no_call" if no tool is appropriate>

Rules:
- Always start with a Thought.
- Action must be a JSON list of {{"name", "arguments"}} objects, even for one call.
- Argument values must respect the parameter types in the spec.
- Never write Observation: yourself — observations come back from the tools.
- When you have enough info, give the Final Answer (short, direct).
"""
    if irrelevance_hint:
        base += (
            "- If the user request CANNOT be fulfilled with any of the available tools,\n"
            '  reply with `Final Answer: no_call` and do NOT emit an Action.\n'
        )
    if multi_turn:
        base += (
            "- This is a multi-turn task: the user may add new instructions across turns.\n"
            "  Issue tool calls for the CURRENT turn only; wait for the next user turn afterwards.\n"
            "- If the user turn is empty, decide whether to call more tools or stop.\n"
        )
    if web_search:
        base += (
            "- This is a multi-hop web research task: typically call `search_engine_query`\n"
            "  first, then `fetch_url_content` on a chosen URL, then synthesize a Final Answer.\n"
        )
    return base


JUDGE_WEB_SYS = (
    "You are an impartial evaluator for multi-hop QA correctness.\n"
    "Compare PREDICTED ANSWER to any of the REFERENCE ANSWERS by semantic meaning.\n"
    "Score 1-5 (1=wrong, 5=fully correct). Respond with ONLY the integer."
)


# ---------------------------------------------------------------------------
# Mock tool executor
# ---------------------------------------------------------------------------
def mock_execute(name: str, args: dict, *, source_file: str) -> str:
    """Return a stub observation. We do NOT run the real BFCL API simulator."""
    if name == "search_engine_query":
        q = args.get("query") or args.get("q") or ""
        return (f"[mock-search] Top result for '{q}': "
                f"https://example.com/{abs(hash(q)) % 10_000} — synthetic snippet.")
    if name == "fetch_url_content":
        url = args.get("url", "")
        return f"[mock-fetch] Content of {url}: (truncated synthetic page body)."
    return f"[mock-tool] {name}({json.dumps(args, default=str)[:200]}) → ok"


# ---------------------------------------------------------------------------
# Router agent
# ---------------------------------------------------------------------------
class BFCLRouterAgent:
    def __init__(
        self,
        router_model: str = DEFAULT_ROUTER_MODEL,
        judge_model: str | None = DEFAULT_JUDGE_MODEL,
        device: str = "cuda",
        router_gpu_ids: list[int] | None = None,
        judge_gpu_ids: list[int] | None = None,
        max_steps: int = 6,
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
            print(f"  Router runtime: OpenVINO (inputs on CPU)")
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
                print(f"  Judge runtime: OpenVINO (inputs on CPU)")
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

    def _generate_judge(self, messages, max_new_tokens=16) -> str:
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

    def judge_correctness(self, question: str, gold_alts: list[str], pred: str) -> float:
        if not self.use_judge:
            return float("nan")
        gold_text = "\n- ".join(gold_alts) if gold_alts else "(none)"
        out = self._generate_judge([
            {"role": "system", "content": JUDGE_WEB_SYS},
            {"role": "user", "content":
                f"QUESTION:\n{question}\n\nREFERENCE ANSWERS:\n- {gold_text}\n\n"
                f"PREDICTED ANSWER:\n{pred}"},
        ])
        m = re.search(r"\b([1-5])\b", out)
        return float(m.group(1)) if m else float("nan")

    # ---- Parsing ------------------------------------------------------
    @staticmethod
    def parse_actions(text: str) -> list[tuple[str, dict]]:
        """Parse `Action: [ {name, arguments}, ... ]` into a list of (name, args)."""
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
    turn: int                       # which user turn (0 for single-turn)
    raw_response: str
    actions: list[dict]             # [{name, arguments}]
    final: str | None
    router_output_tokens: int
    router_thinking_tokens: int
    truncated: bool
    duration_ms: float


@dataclass
class QuestionResult:
    qid: str
    category: str
    source_file: str
    question_repr: str
    gold_calls_per_turn: list[list[tuple[str, dict]]]   # for multi-turn
    gold_call_alts: list[tuple[str, dict]]              # for single-turn
    gold_answer_alts: list[str]                         # for web_search / irrelevance
    allowed_tool_names: set[str] = field(default_factory=set)

    predicted_answer: str = ""
    traces: list[StepTrace] = field(default_factory=list)

    # Aggregated metrics
    tool_call_attempts: int = 0
    tool_call_successes: int = 0       # parsed AND named tool exists in spec
    wrong_tool_count: int = 0
    arg_correct_count: int = 0
    tool_exec_correct: float = float("nan")   # 0/1 or LLM 1-5 normalized
    router_truncations: int = 0
    total_router_tokens: int = 0
    inner_router_tokens: int = 0
    final_router_tokens: int = 0
    total_router_thinking_tokens: int = 0
    total_duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Per-category runner
# ---------------------------------------------------------------------------
class BFCLRunner:
    def __init__(self, agent: BFCLRouterAgent):
        self.agent = agent

    # ---- Helpers ------------------------------------------------------
    def _step_loop(
        self,
        messages: list[dict],
        result: QuestionResult,
        *,
        turn: int,
        max_steps: int,
        execute_tools: bool,
        source_file: str,
    ) -> bool:
        """Run inner Thought/Action loop until Final Answer or max_steps.

        Returns True if a Final Answer was produced.
        """
        for step in range(1, max_steps + 1):
            t0 = time.perf_counter()
            response, stats = self.agent._generate_router(messages)
            ms = (time.perf_counter() - t0) * 1000

            actions = self.agent.parse_actions(response)
            final = self.agent.parse_final(response) if not actions else None

            result.traces.append(StepTrace(
                step=len(result.traces) + 1,
                turn=turn,
                raw_response=response,
                actions=[{"name": n, "arguments": a} for n, a in actions],
                final=final,
                router_output_tokens=stats["output_tokens"],
                router_thinking_tokens=stats["thinking_tokens"],
                truncated=stats["truncated"],
                duration_ms=ms,
            ))
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
                    obs = (mock_execute(name, args, source_file=source_file)
                           if execute_tools else "[ok]")
                    obs_lines.append(f"- {name}: {obs}")
                messages.append({
                    "role": "user",
                    "content": "Observation:\n" + "\n".join(obs_lines),
                })
                continue

            if final is not None:
                result.predicted_answer = final
                return True

            # Format error — nudge once and continue
            messages.append({
                "role": "user",
                "content": "Please respond with either an Action JSON list or a Final Answer.",
            })
        return False

    # ---- Single-turn (simple/multiple/parallel/parallel_multiple/irrelevance) ----
    def run_single_turn(self, sample: dict) -> QuestionResult:
        funcs = resolve_funcs(sample)
        question_msgs = sample["question"][0]   # list[ {role, content} ]
        gold = sample.get("__gold__", {})
        gold_calls_raw = gold.get("ground_truth", []) or []
        # BFCL single-turn gold is alts-aware: [{name: {arg: [accepted_alts]}}, ...]
        gold_calls = parse_gold_single_turn(gold_calls_raw)
        is_irrelevance = sample["__source_file__"] in IRRELEVANCE_FILES

        result = QuestionResult(
            qid=sample["id"],
            category=sample["__category__"],
            source_file=sample["__source_file__"],
            question_repr=question_msgs[0].get("content", "")[:300],
            gold_calls_per_turn=[gold_calls],
            gold_call_alts=gold_calls,
            gold_answer_alts=[],
            allowed_tool_names={f["name"] for f in funcs},
        )

        sys_prompt = build_router_system_prompt(
            funcs, multi_turn=False, web_search=False, irrelevance_hint=is_irrelevance,
        )
        messages = [{"role": "system", "content": sys_prompt}, *question_msgs]

        t_start = time.perf_counter()
        # Single-turn: usually one router step, but allow up to 2 in case of format error.
        self._step_loop(
            messages, result,
            turn=0, max_steps=2,
            execute_tools=False,
            source_file=sample["__source_file__"],
        )
        result.total_duration_ms = (time.perf_counter() - t_start) * 1000

        # ---- Score this question ----
        # Collate the agent's predicted calls (across all steps in this turn)
        pred_calls = [(a["name"], a["arguments"])
                      for tr in result.traces for a in tr.actions]

        if is_irrelevance:
            # Correct iff the agent emitted ZERO tool calls and produced a final answer.
            result.tool_exec_correct = 1.0 if not pred_calls else 0.0
        else:
            # Coverage: every gold call must be matched by some (still-unused) predicted call.
            used_pred: set[int] = set()
            covered = 0
            for g_name, g_alts in gold_calls:
                for p_idx, (p_name, p_args) in enumerate(pred_calls):
                    if p_idx in used_pred:
                        continue
                    if p_name == g_name and args_match_bfcl(p_args, g_alts):
                        used_pred.add(p_idx)
                        covered += 1
                        break
            # Argument-correctness count: any pred whose name+args satisfy some gold entry.
            for p_idx, (p_name, p_args) in enumerate(pred_calls):
                for g_name, g_alts in gold_calls:
                    if p_name == g_name and args_match_bfcl(p_args, g_alts):
                        result.arg_correct_count += 1
                        break
            result.tool_exec_correct = (
                covered / len(gold_calls) if gold_calls else float("nan")
            )

        self._aggregate_tokens(result)
        return result

    # ---- Multi-turn (multi_turn_*) ----
    def run_multi_turn(self, sample: dict) -> QuestionResult:
        funcs = resolve_funcs(sample)
        turns: list[list[dict]] = sample["question"]
        gold = sample.get("__gold__", {})
        gold_per_turn_raw = gold.get("ground_truth", []) or []
        gold_calls_per_turn = [
            parse_call_list(turn_calls if isinstance(turn_calls, list) else [turn_calls])
            for turn_calls in gold_per_turn_raw
        ]
        # Pad to len(turns)
        while len(gold_calls_per_turn) < len(turns):
            gold_calls_per_turn.append([])

        result = QuestionResult(
            qid=sample["id"],
            category=sample["__category__"],
            source_file=sample["__source_file__"],
            question_repr=(turns[0][0]["content"] if turns and turns[0] else "")[:300],
            gold_calls_per_turn=gold_calls_per_turn,
            gold_call_alts=[],
            gold_answer_alts=[],
            allowed_tool_names={f["name"] for f in funcs},
        )

        sys_prompt = build_router_system_prompt(
            funcs, multi_turn=True, web_search=False, irrelevance_hint=False,
        )
        messages = [{"role": "system", "content": sys_prompt}]
        # Optional: stuff initial_config into a system message so the agent sees env state.
        if sample.get("initial_config"):
            messages.append({
                "role": "system",
                "content": f"Initial environment state:\n{json.dumps(sample['initial_config'])[:4000]}",
            })

        per_turn_pred: list[list[tuple[str, dict]]] = []
        t_start = time.perf_counter()
        for t_idx, user_msgs in enumerate(turns):
            if user_msgs:
                messages.extend(user_msgs)
            else:
                messages.append({"role": "user", "content": "(continue)"})

            # Snapshot trace count so we can collect this turn's calls
            n_traces_before = len(result.traces)
            self._step_loop(
                messages, result,
                turn=t_idx, max_steps=self.agent.max_steps,
                execute_tools=True,
                source_file=sample["__source_file__"],
            )
            this_turn = []
            for tr in result.traces[n_traces_before:]:
                for a in tr.actions:
                    this_turn.append((a["name"], a["arguments"]))
            per_turn_pred.append(this_turn)

        result.total_duration_ms = (time.perf_counter() - t_start) * 1000

        # ---- Score: per-turn coverage of gold calls ----
        total_gold = 0
        total_covered = 0
        for gold_turn, pred_turn in zip(gold_calls_per_turn, per_turn_pred):
            total_gold += len(gold_turn)
            for g in gold_turn:
                if any(p[0] == g[0] and args_match(p[1], g[1]) for p in pred_turn):
                    total_covered += 1
            for p in pred_turn:
                if any(p[0] == g[0] and args_match(p[1], g[1]) for g in gold_turn):
                    result.arg_correct_count += 1
        result.tool_exec_correct = (
            total_covered / total_gold if total_gold else float("nan")
        )
        self._aggregate_tokens(result)
        return result

    # ---- Multi-hop web reasoning (web_search) ----
    def run_multi_hop_web(self, sample: dict) -> QuestionResult:
        funcs = resolve_funcs(sample)
        question_msgs = sample["question"][0]
        gold = sample.get("__gold__", {})
        gold_answers = gold.get("ground_truth", []) or []

        result = QuestionResult(
            qid=sample["id"],
            category=sample["__category__"],
            source_file=sample["__source_file__"],
            question_repr=question_msgs[0].get("content", "")[:300],
            gold_calls_per_turn=[],
            gold_call_alts=[],
            gold_answer_alts=gold_answers,
            allowed_tool_names={f["name"] for f in funcs},
        )

        sys_prompt = build_router_system_prompt(
            funcs, multi_turn=False, web_search=True, irrelevance_hint=False,
        )
        messages = [{"role": "system", "content": sys_prompt}, *question_msgs]

        t_start = time.perf_counter()
        self._step_loop(
            messages, result,
            turn=0, max_steps=self.agent.max_steps,
            execute_tools=True,
            source_file=sample["__source_file__"],
        )
        result.total_duration_ms = (time.perf_counter() - t_start) * 1000

        # Score by LLM-judged correctness against gold answer alternatives.
        score = self.agent.judge_correctness(
            question_msgs[0]["content"], gold_answers, result.predicted_answer,
        )
        # Normalize 1-5 → 0..1 (1 maps to 0, 5 maps to 1) for "exec correctness".
        if isinstance(score, float) and not math.isnan(score):
            result.tool_exec_correct = (score - 1) / 4.0
        else:
            result.tool_exec_correct = float("nan")
        self._aggregate_tokens(result)
        return result

    # ---- Aggregation helpers ----
    def _aggregate_tokens(self, result: QuestionResult) -> None:
        result.total_router_tokens = sum(t.router_output_tokens for t in result.traces)
        result.total_router_thinking_tokens = sum(
            t.router_thinking_tokens for t in result.traces
        )
        # Inner = steps that emitted at least one tool call
        result.inner_router_tokens = sum(
            t.router_output_tokens for t in result.traces if t.actions
        )
        # Final = last step that produced a Final Answer
        final_traces = [t for t in result.traces if t.final is not None]
        if final_traces:
            result.final_router_tokens = final_traces[-1].router_output_tokens


# ---------------------------------------------------------------------------
# CSV / summary
# ---------------------------------------------------------------------------
PER_QUESTION_COLUMNS = [
    "qid", "run", "category", "source_file", "question",
    "predicted_answer",
    "tool_exec_correct",
    "tool_call_attempts", "tool_call_successes", "wrong_tool_count",
    "wrong_tool_rate", "arg_correct_count", "argument_correctness",
    "router_truncations",
    "total_router_tokens", "inner_router_tokens", "final_router_tokens",
    "total_router_thinking_tokens",
    "latency_ms",
]

STEP_COLUMNS = [
    "qid", "run", "step", "turn", "n_actions", "tool_names",
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
    runner: BFCLRunner,
    output_tag: str,
    runs_per_sample: int = 1,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_q_path = output_dir / f"bfcl_qaagent_{output_tag}.csv"
    steps_path = output_dir / f"bfcl_qaagent_steps_{output_tag}.csv"
    summary_path = output_dir / f"bfcl_qaagent_summary_{output_tag}.csv"

    per_q_rows: list[dict] = []
    step_rows: list[dict] = []

    runner_fn = {
        "single_turn": runner.run_single_turn,
        "multi_turn": runner.run_multi_turn,
        "multi_hop_web": runner.run_multi_hop_web,
    }

    for cat, samples in samples_by_cat.items():
        print(f"\n=== Category: {cat} ({len(samples)} samples) ===")
        for idx, sample in enumerate(samples, 1):
            qid = sample["id"]
            print(f"[{cat} {idx}/{len(samples)}] {qid}")
            for run_i in range(1, runs_per_sample + 1):
                result = runner_fn[cat](sample)

                wrong_tool_rate = (
                    result.wrong_tool_count / result.tool_call_attempts
                    if result.tool_call_attempts else float("nan")
                )
                argument_correctness = (
                    result.arg_correct_count / result.tool_call_attempts
                    if result.tool_call_attempts else float("nan")
                )
                per_q_rows.append({
                    "qid": qid,
                    "run": run_i,
                    "category": cat,
                    "source_file": result.source_file,
                    "question": result.question_repr,
                    "predicted_answer": result.predicted_answer[:500],
                    "tool_exec_correct": result.tool_exec_correct,
                    "tool_call_attempts": result.tool_call_attempts,
                    "tool_call_successes": result.tool_call_successes,
                    "wrong_tool_count": result.wrong_tool_count,
                    "wrong_tool_rate": wrong_tool_rate,
                    "arg_correct_count": result.arg_correct_count,
                    "argument_correctness": argument_correctness,
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
                        "turn": tr.turn,
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
                    f"attempts={result.tool_call_attempts} "
                    f"wrong={result.wrong_tool_count} "
                    f"arg_ok={result.arg_correct_count} "
                    f"trunc={result.router_truncations} "
                    f"router_tok={result.total_router_tokens}"
                )

    with open(per_q_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_QUESTION_COLUMNS)
        w.writeheader()
        w.writerows(per_q_rows)
    with open(steps_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STEP_COLUMNS)
        w.writeheader()
        w.writerows(step_rows)

    # Summary per category + overall
    def _summarize(rows):
        return {
            "n": len(rows),
            "tool_exec_correct": _safe_mean([r["tool_exec_correct"] for r in rows]),
            "tool_call_attempts": _safe_mean([r["tool_call_attempts"] for r in rows]),
            "tool_call_successes": _safe_mean([r["tool_call_successes"] for r in rows]),
            "wrong_tool_rate": _safe_mean([r["wrong_tool_rate"] for r in rows]),
            "argument_correctness": _safe_mean([r["argument_correctness"] for r in rows]),
            "router_truncations": _safe_mean([r["router_truncations"] for r in rows]),
            "total_router_tokens": _safe_mean([r["total_router_tokens"] for r in rows]),
            "inner_router_tokens": _safe_mean([r["inner_router_tokens"] for r in rows]),
            "final_router_tokens": _safe_mean([r["final_router_tokens"] for r in rows]),
            "total_router_thinking_tokens": _safe_mean(
                [r["total_router_thinking_tokens"] for r in rows]),
            "latency_ms": _safe_mean([r["latency_ms"] for r in rows]),
        }

    summary_rows = [{"category": "ALL", **_summarize(per_q_rows)}]
    for cat in samples_by_cat:
        summary_rows.append({
            "category": cat,
            **_summarize([r for r in per_q_rows if r["category"] == cat]),
        })

    cols = ["category", "n", "tool_exec_correct", "tool_call_attempts",
            "tool_call_successes", "wrong_tool_rate", "argument_correctness",
            "router_truncations", "total_router_tokens",
            "inner_router_tokens", "final_router_tokens",
            "total_router_thinking_tokens", "latency_ms"]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\nPer-question CSV : {per_q_path}")
    print(f"Step trace CSV   : {steps_path}")
    print(f"Summary CSV      : {summary_path}")

    print(f"\n{'='*70}\nBFCL v4 BENCHMARK SUMMARY\n{'='*70}")
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
        description="BFCL v4 tool-calling benchmark (router efficiency metrics)",
    )
    parser.add_argument("--router", default=DEFAULT_ROUTER_MODEL)
    parser.add_argument("--judge", default=DEFAULT_JUDGE_MODEL,
                        help="Judge model (used for multi-hop web QA scoring). "
                             "Pass empty string to disable.")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"],
                        help="Compute device. Defaults to 'cpu' for OpenVINO "
                             "presets (-ov / -ov-int4 / etc.), 'cuda' otherwise.")
    parser.add_argument("--router-gpu", default=None)
    parser.add_argument("--judge-gpu", default=None)
    parser.add_argument("--categories",
                        default="single_turn,multi_turn,multi_hop_web",
                        help="Comma-separated list from "
                             "{single_turn,multi_turn,multi_hop_web}")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Samples per category (stratified across files).")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--data-base", default=BFCL_DATA_BASE,
                        help="Base URL for BFCL v4 JSON files "
                             "(default: gorilla GitHub raw)")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--category", default="single_turn",
                        help="Category to use in --interactive mode")
    args = parser.parse_args()

    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    for c in cats:
        if c not in CATEGORY_FILES:
            sys.exit(f"Unknown category {c}; choose from {list(CATEGORY_FILES)}")

    ensure_bfcl_files(args.data_base, cats)
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

    agent = BFCLRouterAgent(
        router_model=args.router,
        judge_model=args.judge or None,
        device=args.device,
        router_gpu_ids=parse_gpu(args.router_gpu),
        judge_gpu_ids=parse_gpu(args.judge_gpu),
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
    )
    runner = BFCLRunner(agent)

    if args.interactive:
        cat = args.category
        samples = load_category(cat, num_samples=5)
        if not samples:
            sys.exit(f"No samples found for {cat}; run --download-only first.")
        sample = samples[0]
        print(f"\nInteractive: {cat} / {sample['id']}")
        print("Q:", sample["question"][0][0]["content"][:500])
        fn = {"single_turn": runner.run_single_turn,
              "multi_turn": runner.run_multi_turn,
              "multi_hop_web": runner.run_multi_hop_web}[cat]
        result = fn(sample)
        print("\nPredicted:", result.predicted_answer)
        print(f"attempts={result.tool_call_attempts} "
              f"wrong_tools={result.wrong_tool_count} "
              f"arg_correct={result.arg_correct_count} "
              f"exec_correct={_fmt(result.tool_exec_correct)}")
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
