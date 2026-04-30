"""
HotpotQA ReAct RAG Benchmark
============================
ReAct (Reason → Act → Observe) RAG benchmark on the HotpotQA dataset.

The router LLM follows a Thought / Action / Observation loop with a single
`retrieve` tool that hits the Chroma vector DB created by `db_creator.py`.
After each retrieval, an optional **judge LLM** decides whether the
information gathered is sufficient — this is what gives us the
*judge stop quality* metric (does the judge stop only when the gold
supporting facts have been retrieved?).

Metrics produced (per question, averaged at the end):
  - Retrieval success           : did the retrieved set cover ALL gold supporting titles?
  - Retrieval recall             : |retrieved ∩ gold_titles| / |gold_titles|
  - Retrieval precision          : |retrieved ∩ gold_titles| / |retrieved_titles|
  - Number of agent loops        : total ReAct steps before final answer
  - Judge stop quality           : when the judge said SUFFICIENT, were all gold titles
                                   actually retrieved? (proxy: precision of the stop signal)
  - Final answer correctness     : LLM-judged 1-5 against the gold answer
  - Final answer faithfulness    : LLM-judged 1-5 against the retrieved context
  - Final answer truthfulness    : LLM-judged 1-5 against the gold supporting context
  - Final answer length          : tokens in the final answer (router tokenizer)
  - Router inner-token length    : router output tokens spent in tool-calling steps
                                   (i.e. all router tokens minus the final-answer step)

Usage:
  # 1. Build the HotpotQA Chroma DB (only the first time):
  python db_creator.py --build

  # 2. Run the benchmark (qwen3-8b router + qwen3-14b judge by default):
  python hotpotqa_react_llm.py --num-samples 50

  # Single-question debugging:
  python hotpotqa_react_llm.py --interactive
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
# Make langGraph + the parent project importable
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "langGraph"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_loader import load_llm, load_embedding, MODELS_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------
# All model weights (router LLM, judge LLM, embedding model) are loaded
# from /agentic_rag/models/ first via model_loader; missing models fall
# back to the HuggingFace hub.
DEFAULT_MODELS_DIR = PROJECT_DIR / "models"
DATASET_DIR = PROJECT_DIR / "dataset"
# Benchmark CSV / JSON results land here (one folder per benchmark family).
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DB_DIR = PROJECT_DIR / "db" / "hotpotqa_chroma"
SAMPLES_JSON = DATASET_DIR / "hotpotqa_samples.json"

DEFAULT_COLLECTION = "hotpotqa"
DEFAULT_ROUTER_MODEL = "qwen3-8b"
DEFAULT_JUDGE_MODEL = "qwen3-14b"
# Preset name in model_loader → resolves to /agentic_rag/models/BAAI--bge-large-en-v1.5
DEFAULT_EMBEDDING = "bge-large"


# ---------------------------------------------------------------------------
# Vector store (singleton)
# ---------------------------------------------------------------------------
_retrieval_vectorstore = None


def _inspect_chroma_collection(
    chroma_dir: Path, collection_name: str,
) -> tuple[int | None, int]:
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
            SELECT COUNT(*) FROM embeddings e
            JOIN segments s ON e.segment_id = s.id
            WHERE s.collection = ?
            """,
            (collection_id,),
        )
        return dimension, cur.fetchone()[0]
    finally:
        conn.close()


def _get_vectorstore(
    persist_dir: Path = DB_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING,
):
    global _retrieval_vectorstore
    if _retrieval_vectorstore is not None:
        return _retrieval_vectorstore

    from langchain_chroma import Chroma

    dim, count = _inspect_chroma_collection(persist_dir, collection_name)
    if count == 0:
        raise RuntimeError(
            f"Chroma collection '{collection_name}' is missing or empty in "
            f"{persist_dir}. Run `python db_creator.py --build` first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Loading retrieval embedding from {DEFAULT_MODELS_DIR}: "
        f"{embedding_model} (device={device})"
    )
    embeddings = load_embedding(embedding_model, device=device)

    _retrieval_vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )
    print(
        f"Retrieval ready (collection={collection_name}, dim={dim}, count={count})"
    )
    return _retrieval_vectorstore


# ---------------------------------------------------------------------------
# Retrieval tool
# ---------------------------------------------------------------------------
RETRIEVE_K = 5


def tool_retrieve(query: str) -> tuple[str, list[str]]:
    """Return (text_observation, list_of_titles_retrieved)."""
    try:
        vectorstore = _get_vectorstore()
    except RuntimeError as exc:
        return f"Error: {exc}", []

    results = vectorstore.similarity_search(query, k=RETRIEVE_K)
    if not results:
        return f"No relevant documents found for '{query}'.", []

    titles: list[str] = []
    lines: list[str] = []
    for i, doc in enumerate(results, 1):
        title = doc.metadata.get("title") or doc.metadata.get("source") or "unknown"
        titles.append(title)
        lines.append(f"--- Document {i} (title: {title}) ---")
        lines.append(doc.page_content)
        lines.append("")
    return "\n".join(lines), titles


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
def build_router_system_prompt() -> str:
    today = datetime.now().strftime("%Y-%m-%d (%A)")
    return f"""You are a helpful assistant that answers multi-hop questions using the ReAct framework (Reasoning + Acting).

Today's date is {today}.

You have access to one tool:
  - retrieve: Retrieve relevant Wikipedia paragraphs from a knowledge base. Input: a natural language query.

Each step must be EXACTLY one of:

(1) Thought + Action:
Thought: <your reasoning about what to look up next>
Action: retrieve
Action Input: <a focused natural-language search query>

(2) Thought + Final Answer:
Thought: <your reasoning>
Final Answer: <a concise answer — usually a single phrase, name, or number>

Rules:
- Always start with a Thought.
- If you already know the answer confidently from your own knowledge and
  the information you have is sufficient, you may skip the retrieve tool
  entirely and go straight to the Final Answer. State this explicitly in
  your Thought (e.g. "I already know the answer with high confidence, no
  retrieval needed.") before giving the Final Answer.
- HotpotQA questions are multi-hop: if you are not fully confident, you
  usually need at least two retrieval calls, one for each entity / fact involved.
- Never include both an Action and a Final Answer in the same response.
- Never write Observation: yourself — observations come from the tool only.
- When you have enough information, give the Final Answer.
- The Final Answer should be short and direct (a name, date, number, or short phrase),
  not a long explanation."""


JUDGE_STOP_SYSTEM = (
    "You are a judge that decides whether the information gathered "
    "so far is sufficient to answer the user's question.\n\n"
    "You will be given:\n"
    "  1. The original QUESTION.\n"
    "  2. One or more OBSERVATIONS from retrieval calls.\n\n"
    "Decide whether the observations contain enough information to "
    "provide a complete and accurate answer.\n\n"
    "Respond with exactly one word: SUFFICIENT or CONTINUE.\n"
    "  - SUFFICIENT: The observations already contain enough information.\n"
    "  - CONTINUE:   More retrieval is needed."
)


# ---------------------------------------------------------------------------
# ReAct agent (router + judge)
# ---------------------------------------------------------------------------
class ReActHotpotAgent:
    """ReAct router with retrieval tool and a sufficiency judge."""

    def __init__(
        self,
        router_model: str = DEFAULT_ROUTER_MODEL,
        judge_model: str | None = DEFAULT_JUDGE_MODEL,
        device: str = "cuda",
        router_gpu_ids: list[int] | None = None,
        judge_gpu_ids: list[int] | None = None,
        max_steps: int = 6,
        max_new_tokens: int = 1024,
    ):
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.use_judge = judge_model is not None

        # --- Router model ---
        print(f"Loading router LLM: {router_model}")
        self.router, self.router_tok = load_llm(
            router_model, device=device, gpu_ids=router_gpu_ids,
        )
        try:
            self.router_input_device = next(self.router.parameters()).device
        except (StopIteration, AttributeError):
            self.router_input_device = "cpu"

        # --- Judge model (optional) ---
        if self.use_judge:
            print(f"Loading judge LLM: {judge_model}")
            self.judge, self.judge_tok = load_llm(
                judge_model, device=device, gpu_ids=judge_gpu_ids,
            )
            try:
                self.judge_input_device = next(self.judge.parameters()).device
            except (StopIteration, AttributeError):
                self.judge_input_device = "cpu"
        else:
            self.judge = self.judge_tok = None
            self.judge_input_device = None

    # ---- generation helpers --------------------------------------------
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

    def _generate_judge(
        self, messages: list[dict], max_new_tokens: int = 64,
    ) -> tuple[str, dict]:
        if not self.use_judge:
            return "", {"prompt_tokens": 0, "output_tokens": 0,
                        "thinking_tokens": 0, "truncated": False}

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
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
            )
        new_ids = out[0][prompt_tokens:]
        out_tokens = int(new_ids.shape[0])
        decoded = self.judge_tok.decode(new_ids, skip_special_tokens=True).strip()
        decoded = re.sub(r"<think>.*?</think>", "", decoded, flags=re.DOTALL).strip()
        return decoded, {
            "prompt_tokens": prompt_tokens,
            "output_tokens": out_tokens,
            "thinking_tokens": 0,
            "truncated": out_tokens >= max_new_tokens,
        }

    # ---- judging helpers ------------------------------------------------
    def judge_stop(
        self, question: str, observations: list[str],
    ) -> tuple[bool, str, dict]:
        if not self.use_judge or not observations:
            return False, "", {"prompt_tokens": 0, "output_tokens": 0,
                               "thinking_tokens": 0, "truncated": False}
        obs_text = "\n\n".join(
            f"[Observation {i}]\n{o}" for i, o in enumerate(observations, 1)
        )
        msgs = [
            {"role": "system", "content": JUDGE_STOP_SYSTEM},
            {"role": "user",
             "content": f"QUESTION:\n{question}\n\nOBSERVATIONS:\n{obs_text}"},
        ]
        text, stats = self._generate_judge(msgs, max_new_tokens=8)
        return "sufficient" in text.lower(), text, stats

    def judge_score(
        self,
        system_msg: str,
        user_msg: str,
        max_new_tokens: int = 16,
    ) -> tuple[float, str]:
        """Run the judge and parse a 1-5 integer score from its response."""
        if not self.use_judge:
            return float("nan"), ""
        text, _ = self._generate_judge(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_new_tokens=max_new_tokens,
        )
        m = re.search(r"\b([1-5])\b", text)
        return (float(m.group(1)) if m else float("nan")), text

    # ---- parsing helpers -----------------------------------------------
    @staticmethod
    def _parse_action(text: str) -> tuple[str | None, str | None]:
        a = re.search(r"Action:\s*(.+)", text)
        i = re.search(r"Action Input:\s*(.+)", text)
        if a and i:
            return a.group(1).strip(), i.group(1).strip()
        return None, None

    @staticmethod
    def _parse_final(text: str) -> str | None:
        m = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
        return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Tracing data classes
# ---------------------------------------------------------------------------
@dataclass
class StepTrace:
    step: int
    thought: str
    tool_called: str | None
    action_input: str | None
    observation: str | None
    retrieved_titles: list[str]
    judge_decision: str           # "SUFFICIENT" | "CONTINUE" | "N/A"
    judge_response: str
    router_output_tokens: int
    router_thinking_tokens: int
    judge_output_tokens: int
    generation_ms: float
    judge_ms: float
    duration_ms: float


@dataclass
class QuestionResult:
    qid: str
    question: str
    gold_answer: str
    gold_titles: list[str]
    gold_context: str
    predicted_answer: str = ""
    traces: list[StepTrace] = field(default_factory=list)
    retrieved_titles: list[str] = field(default_factory=list)
    retrieved_context: str = ""
    total_steps: int = 0
    total_duration_ms: float = 0.0
    inner_router_tokens: int = 0          # tokens spent in tool-calling steps
    final_router_tokens: int = 0          # tokens in the final-answer step
    total_router_tokens: int = 0
    final_answer_tokens: int = 0          # token length of just the final answer
    judge_stopped: bool = False
    judge_stop_step: int | None = None


# ---------------------------------------------------------------------------
# ReAct + tracing wrapper
# ---------------------------------------------------------------------------
class HotpotReActRunner:
    """Runs the ReAct loop with full tracing for HotpotQA samples."""

    def __init__(self, agent: ReActHotpotAgent):
        self.agent = agent
        # Pre-load vectorstore so the first question doesn't pay the load cost
        _get_vectorstore()

    def run(self, sample: dict) -> QuestionResult:
        question = sample["question"]
        gold_answer = sample["answer"]
        # Gold supporting titles (de-duplicated, preserving order)
        gold_titles_ordered: list[str] = []
        seen = set()
        for title, _sid in sample.get("supporting_facts", []):
            if title not in seen:
                seen.add(title)
                gold_titles_ordered.append(title)

        # Gold context = sentences from supporting paragraphs only
        title_to_sents = {t: s for t, s in sample.get("context", [])}
        gold_context_parts = []
        for t in gold_titles_ordered:
            sents = title_to_sents.get(t, [])
            if sents:
                gold_context_parts.append(f"[{t}] " + " ".join(sents))
        gold_context = "\n".join(gold_context_parts)

        result = QuestionResult(
            qid=sample.get("id", ""),
            question=question,
            gold_answer=gold_answer,
            gold_titles=gold_titles_ordered,
            gold_context=gold_context,
        )

        system_prompt = build_router_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        observations: list[str] = []
        all_titles: list[str] = []
        all_context_chunks: list[str] = []

        t_start = time.perf_counter()

        for step in range(1, self.agent.max_steps + 1):
            step_start = time.perf_counter()

            # --- Router generation ---
            gen_start = time.perf_counter()
            response, gen_stats = self.agent._generate_router(messages)
            gen_ms = (time.perf_counter() - gen_start) * 1000

            judge_decision = "N/A"
            judge_response = ""
            judge_stats = {"output_tokens": 0}
            judge_ms = 0.0
            tool_called: str | None = None
            action_input: str | None = None
            observation: str | None = None
            step_titles: list[str] = []

            action, action_input = self.agent._parse_action(response)
            if action is not None:
                tool_called = action.lower()

                if tool_called == "retrieve":
                    observation, step_titles = tool_retrieve(action_input)
                else:
                    observation = f"Error: Unknown tool '{action}'."

                if step_titles:
                    all_titles.extend(step_titles)
                    all_context_chunks.append(observation)
                observations.append(observation)

                # --- Judge sufficiency ---
                if self.agent.use_judge:
                    j_start = time.perf_counter()
                    sufficient, judge_response, judge_stats = self.agent.judge_stop(
                        question, observations,
                    )
                    judge_ms = (time.perf_counter() - j_start) * 1000
                    judge_decision = "SUFFICIENT" if sufficient else "CONTINUE"
                    if sufficient and not result.judge_stopped:
                        result.judge_stopped = True
                        result.judge_stop_step = step

                step_ms = (time.perf_counter() - step_start) * 1000
                result.traces.append(StepTrace(
                    step=step, thought=response,
                    tool_called=tool_called, action_input=action_input,
                    observation=observation, retrieved_titles=step_titles,
                    judge_decision=judge_decision, judge_response=judge_response,
                    router_output_tokens=gen_stats["output_tokens"],
                    router_thinking_tokens=gen_stats["thinking_tokens"],
                    judge_output_tokens=judge_stats.get("output_tokens", 0),
                    generation_ms=gen_ms, judge_ms=judge_ms,
                    duration_ms=step_ms,
                ))

                messages.append({"role": "assistant", "content": response})
                if judge_decision == "SUFFICIENT":
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Observation: {observation}\n\n"
                            "The information gathered is sufficient. "
                            "Please provide your Final Answer now."
                        ),
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": f"Observation: {observation}",
                    })
                continue

            final = self.agent._parse_final(response)
            if final:
                step_ms = (time.perf_counter() - step_start) * 1000
                result.traces.append(StepTrace(
                    step=step, thought=response,
                    tool_called=None, action_input=None,
                    observation="[final answer]",
                    retrieved_titles=[],
                    judge_decision="N/A", judge_response="",
                    router_output_tokens=gen_stats["output_tokens"],
                    router_thinking_tokens=gen_stats["thinking_tokens"],
                    judge_output_tokens=0,
                    generation_ms=gen_ms, judge_ms=0.0,
                    duration_ms=step_ms,
                ))
                result.predicted_answer = final
                break

            # Format error — nudge and continue
            step_ms = (time.perf_counter() - step_start) * 1000
            result.traces.append(StepTrace(
                step=step, thought=response,
                tool_called=None, action_input=None,
                observation="[format error]",
                retrieved_titles=[],
                judge_decision="N/A", judge_response="",
                router_output_tokens=gen_stats["output_tokens"],
                router_thinking_tokens=gen_stats["thinking_tokens"],
                judge_output_tokens=0,
                generation_ms=gen_ms, judge_ms=0.0,
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

        # Aggregate token counts
        result.total_router_tokens = sum(t.router_output_tokens for t in result.traces)
        result.inner_router_tokens = sum(
            t.router_output_tokens for t in result.traces if t.tool_called is not None
        )
        final_traces = [t for t in result.traces if t.observation == "[final answer]"]
        if final_traces:
            result.final_router_tokens = final_traces[-1].router_output_tokens

        # Length of just the final-answer string in router tokens
        if result.predicted_answer:
            result.final_answer_tokens = len(
                self.agent.router_tok.encode(
                    result.predicted_answer, add_special_tokens=False,
                )
            )

        # Deduplicated retrieved titles, preserving first-seen order
        seen = set()
        dedup_titles: list[str] = []
        for t in all_titles:
            if t not in seen:
                seen.add(t)
                dedup_titles.append(t)
        result.retrieved_titles = dedup_titles
        result.retrieved_context = "\n\n".join(all_context_chunks)
        return result


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def retrieval_metrics(retrieved: list[str], gold: list[str]) -> dict:
    """Compute retrieval success / recall / precision over Wikipedia titles."""
    gold_set = set(gold)
    retr_set = set(retrieved)
    if not gold_set:
        return {"success": float("nan"), "recall": float("nan"),
                "precision": float("nan")}
    overlap = gold_set & retr_set
    recall = len(overlap) / len(gold_set)
    precision = (len(overlap) / len(retr_set)) if retr_set else 0.0
    success = 1.0 if gold_set.issubset(retr_set) else 0.0
    return {"success": success, "recall": recall, "precision": precision}


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


# ---------------------------------------------------------------------------
# Judge prompts for end-of-question scoring
# ---------------------------------------------------------------------------
CORRECTNESS_SYS = (
    "You are an impartial evaluator for HotpotQA answer correctness.\n"
    "Compare PREDICTED ANSWER to REFERENCE ANSWER by semantic meaning, not exact wording.\n"
    "Score 1-5:\n"
    "  1 = Completely incorrect\n"
    "  2 = Mostly incorrect; misses the key point\n"
    "  3 = Partially correct; captures some important parts\n"
    "  4 = Mostly correct; minor inaccuracies\n"
    "  5 = Fully correct and complete\n"
    "Respond with ONLY the score as a single integer (1-5)."
)

FAITHFULNESS_SYS = (
    "You are an impartial evaluator for answer faithfulness.\n"
    "Decide whether the ANSWER is supported by the CONTEXT (which is what the agent actually retrieved).\n"
    "Score 1-5:\n"
    "  1 = Completely unfaithful; contradicts or fabricates\n"
    "  2 = Mostly unfaithful; major unsupported claims\n"
    "  3 = Partially faithful; mixed support\n"
    "  4 = Mostly faithful; minor unsupported details\n"
    "  5 = Fully faithful; all claims are supported\n"
    "Respond with ONLY the score as a single integer (1-5)."
)

TRUTHFULNESS_SYS = (
    "You are an impartial evaluator for answer truthfulness.\n"
    "Decide whether the ANSWER is consistent with the GOLD CONTEXT (the ground-truth supporting evidence).\n"
    "Score 1-5:\n"
    "  1 = Completely false relative to the gold context\n"
    "  2 = Mostly false; major contradictions\n"
    "  3 = Partly true; mixed accuracy\n"
    "  4 = Mostly true; minor inaccuracies\n"
    "  5 = Fully true given the gold context\n"
    "Respond with ONLY the score as a single integer (1-5)."
)


def _clip(text: str, limit: int = 4000) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]..."


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------
PER_QUESTION_COLUMNS = [
    "qid", "run", "question", "gold_answer", "predicted_answer",
    "gold_titles", "retrieved_titles",
    "retrieval_success", "retrieval_recall", "retrieval_precision",
    "agent_loops", "judge_stopped", "judge_stop_step",
    "judge_stop_quality",
    "correctness", "faithfulness", "truthfulness",
    "final_answer_tokens", "inner_router_tokens",
    "final_router_tokens", "total_router_tokens",
    "latency_ms",
]

STEP_COLUMNS = [
    "qid", "run", "step", "tool_called", "action_input",
    "judge_decision", "router_output_tokens",
    "generation_ms", "judge_ms", "step_ms",
    "retrieved_titles",
]

SUMMARY_COLUMNS = [
    "metric", "mean", "n",
]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark(
    samples: list[dict],
    runner: HotpotReActRunner,
    output_tag: str,
    runs_per_sample: int = 3,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Run the benchmark, repeating each sample `runs_per_sample` times.

    Saves three CSVs in `output_dir`:
      - hotpotqa_react_llm_<tag>.csv         (per question × run)
      - hotpotqa_react_llm_steps_<tag>.csv   (per step trace)
      - hotpotqa_react_llm_summary_<tag>.csv (mean of every metric)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_q_path = output_dir / f"hotpotqa_react_llm_{output_tag}.csv"
    steps_path = output_dir / f"hotpotqa_react_llm_steps_{output_tag}.csv"
    summary_path = output_dir / f"hotpotqa_react_llm_summary_{output_tag}.csv"

    per_q_rows: list[dict] = []
    step_rows: list[dict] = []

    total_runs = len(samples) * runs_per_sample
    print(
        f"\nRunning HotpotQA ReAct benchmark on {len(samples)} samples × "
        f"{runs_per_sample} runs each = {total_runs} total runs..."
    )
    for idx, sample in enumerate(samples, 1):
        qid = sample.get("id", str(idx))
        print(f"\n[{idx}/{len(samples)}] (qid={qid}) {sample['question'][:120]}")

        for run_i in range(1, runs_per_sample + 1):
            print(f"  -- run {run_i}/{runs_per_sample} --")
            result = runner.run(sample)

            # ---- Retrieval metrics ----
            retr = retrieval_metrics(result.retrieved_titles, result.gold_titles)

            # ---- Judge stop quality ----
            if result.judge_stopped:
                judge_stop_quality = 1.0 if set(result.gold_titles).issubset(
                    set(result.retrieved_titles)
                ) else 0.0
            else:
                judge_stop_quality = float("nan")

            # ---- LLM-judged answer scores ----
            correctness, _ = runner.agent.judge_score(
                CORRECTNESS_SYS,
                f"QUESTION:\n{sample['question']}\n\n"
                f"REFERENCE ANSWER:\n{result.gold_answer}\n\n"
                f"PREDICTED ANSWER:\n{result.predicted_answer}",
            )
            faithfulness, _ = runner.agent.judge_score(
                FAITHFULNESS_SYS,
                f"CONTEXT:\n{_clip(result.retrieved_context)}\n\n"
                f"ANSWER:\n{result.predicted_answer}",
            )
            truthfulness, _ = runner.agent.judge_score(
                TRUTHFULNESS_SYS,
                f"GOLD CONTEXT:\n{_clip(result.gold_context)}\n\n"
                f"ANSWER:\n{result.predicted_answer}",
            )

            per_q_rows.append({
                "qid": qid,
                "run": run_i,
                "question": result.question,
                "gold_answer": result.gold_answer,
                "predicted_answer": result.predicted_answer,
                "gold_titles": ";".join(result.gold_titles),
                "retrieved_titles": ";".join(result.retrieved_titles),
                "retrieval_success": retr["success"],
                "retrieval_recall": retr["recall"],
                "retrieval_precision": retr["precision"],
                "agent_loops": result.total_steps,
                "judge_stopped": int(result.judge_stopped),
                "judge_stop_step": result.judge_stop_step or "",
                "judge_stop_quality": judge_stop_quality,
                "correctness": correctness,
                "faithfulness": faithfulness,
                "truthfulness": truthfulness,
                "final_answer_tokens": result.final_answer_tokens,
                "inner_router_tokens": result.inner_router_tokens,
                "final_router_tokens": result.final_router_tokens,
                "total_router_tokens": result.total_router_tokens,
                "latency_ms": round(result.total_duration_ms, 1),
            })
            for tr in result.traces:
                step_rows.append({
                    "qid": qid,
                    "run": run_i,
                    "step": tr.step,
                    "tool_called": tr.tool_called or "",
                    "action_input": tr.action_input or "",
                    "judge_decision": tr.judge_decision,
                    "router_output_tokens": tr.router_output_tokens,
                    "generation_ms": round(tr.generation_ms, 1),
                    "judge_ms": round(tr.judge_ms, 1),
                    "step_ms": round(tr.duration_ms, 1),
                    "retrieved_titles": ";".join(tr.retrieved_titles),
                })

            print(
                f"     → loops={result.total_steps} "
                f"recall={retr['recall']:.2f} precision={retr['precision']:.2f} "
                f"correctness={_fmt(correctness)} "
                f"inner_tokens={result.inner_router_tokens} "
                f"final_tokens={result.final_answer_tokens}"
            )

    # ---- Write outputs ----
    with open(per_q_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_QUESTION_COLUMNS)
        w.writeheader()
        w.writerows(per_q_rows)
    with open(steps_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STEP_COLUMNS)
        w.writeheader()
        w.writerows(step_rows)

    summary = {
        "n_questions": len({r["qid"] for r in per_q_rows}),
        "n_runs": len(per_q_rows),
        "runs_per_sample": runs_per_sample,
        "retrieval_success": _safe_mean([r["retrieval_success"] for r in per_q_rows]),
        "retrieval_recall":  _safe_mean([r["retrieval_recall"]  for r in per_q_rows]),
        "retrieval_precision": _safe_mean([r["retrieval_precision"] for r in per_q_rows]),
        "agent_loops":       _safe_mean([r["agent_loops"]      for r in per_q_rows]),
        "judge_stop_rate":   _safe_mean([r["judge_stopped"]    for r in per_q_rows]),
        "judge_stop_quality": _safe_mean([r["judge_stop_quality"] for r in per_q_rows]),
        "correctness":       _safe_mean([r["correctness"]      for r in per_q_rows]),
        "faithfulness":      _safe_mean([r["faithfulness"]     for r in per_q_rows]),
        "truthfulness":      _safe_mean([r["truthfulness"]     for r in per_q_rows]),
        "final_answer_tokens": _safe_mean([r["final_answer_tokens"] for r in per_q_rows]),
        "inner_router_tokens": _safe_mean([r["inner_router_tokens"] for r in per_q_rows]),
        "final_router_tokens": _safe_mean([r["final_router_tokens"] for r in per_q_rows]),
        "total_router_tokens": _safe_mean([r["total_router_tokens"] for r in per_q_rows]),
        "latency_ms":        _safe_mean([r["latency_ms"]       for r in per_q_rows]),
    }

    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        for metric, val in summary.items():
            w.writerow({"metric": metric, "mean": val, "n": summary["n_runs"]})

    print(f"\nPer-run CSV      : {per_q_path}")
    print(f"Step trace CSV   : {steps_path}")
    print(f"Summary CSV      : {summary_path}")

    _print_summary(summary)
    return {"summary": summary, "csv_path": str(per_q_path)}


def _print_summary(s: dict):
    print(
        f"\n{'='*60}\n"
        f"HOTPOTQA REACT BENCHMARK SUMMARY  "
        f"(n_questions={s['n_questions']}, runs_per_sample={s['runs_per_sample']}, "
        f"n_runs={s['n_runs']})\n{'='*60}"
    )
    print(f"\n--- Retrieval ---")
    print(f"  Success (all gold titles hit) : {_fmt(s['retrieval_success'])}")
    print(f"  Recall                         : {_fmt(s['retrieval_recall'])}")
    print(f"  Precision                      : {_fmt(s['retrieval_precision'])}")
    print(f"\n--- Loop / Judge ---")
    print(f"  Avg agent loops                : {_fmt(s['agent_loops'])}")
    print(f"  Judge stop rate                : {_fmt(s['judge_stop_rate'])}")
    print(f"  Judge stop quality             : {_fmt(s['judge_stop_quality'])}")
    print(f"\n--- Final Answer (LLM-judged 1-5) ---")
    print(f"  Correctness                    : {_fmt(s['correctness'])}")
    print(f"  Faithfulness                   : {_fmt(s['faithfulness'])}")
    print(f"  Truthfulness                   : {_fmt(s['truthfulness'])}")
    print(f"\n--- Token Lengths ---")
    print(f"  Final answer tokens (avg)      : {_fmt(s['final_answer_tokens'])}")
    print(f"  Router inner tokens (avg)      : {_fmt(s['inner_router_tokens'])}")
    print(f"  Router final tokens (avg)      : {_fmt(s['final_router_tokens'])}")
    print(f"  Router total tokens (avg)      : {_fmt(s['total_router_tokens'])}")
    print(f"\n--- E2E ---")
    print(f"  Latency ms (avg)               : {_fmt(s['latency_ms'])}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------
def load_samples(num_samples: int, samples_file: Path | None) -> list[dict]:
    """Load HotpotQA samples, preferring the JSON saved by db_creator.py."""
    if samples_file and samples_file.exists():
        print(f"Loading HotpotQA samples from {samples_file}")
        all_samples = json.loads(samples_file.read_text())
    else:
        print("No saved samples found; loading directly from HuggingFace.")
        from db_creator import load_hotpotqa
        all_samples = load_hotpotqa(num_samples=0)

    if num_samples and num_samples > 0:
        return all_samples[:num_samples]
    return all_samples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="HotpotQA ReAct LLM benchmark with retrieval + judge metrics",
    )
    parser.add_argument("--router", default=DEFAULT_ROUTER_MODEL,
                        help=f"Router model (default: {DEFAULT_ROUTER_MODEL})")
    parser.add_argument("--judge", default=DEFAULT_JUDGE_MODEL,
                        help=f"Judge model (default: {DEFAULT_JUDGE_MODEL}). "
                             f"Pass empty string to disable.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--router-gpu", default=None,
                        help="GPU index/indices for router (e.g. '0' or '0,1')")
    parser.add_argument("--judge-gpu", default=None,
                        help="GPU index/indices for judge (e.g. '1' or '2,3')")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of HotpotQA samples to evaluate (default: 50)")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of runs per sample (default: 3)")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help=f"Directory to write CSVs to (default: {OUTPUT_DIR})")
    parser.add_argument("--max-steps", type=int, default=6,
                        help="Max ReAct steps per question (default: 6)")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Max new tokens per router generation (default: 1024)")
    parser.add_argument("--samples-file", default=str(SAMPLES_JSON),
                        help=f"HotpotQA samples JSON (default: {SAMPLES_JSON})")
    parser.add_argument("--interactive", action="store_true",
                        help="Run a single interactive question (no metrics)")

    args = parser.parse_args()

    def parse_gpu(s):
        if s is None:
            return None
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    judge_arg = args.judge if args.judge else None

    agent = ReActHotpotAgent(
        router_model=args.router,
        judge_model=judge_arg,
        device=args.device,
        router_gpu_ids=parse_gpu(args.router_gpu),
        judge_gpu_ids=parse_gpu(args.judge_gpu),
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
    )
    runner = HotpotReActRunner(agent)

    if args.interactive:
        print("\nHotpotQA ReAct interactive mode (Ctrl-D to exit)")
        while True:
            try:
                q = input("\nQ: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                continue
            sample = {"id": "interactive", "question": q, "answer": "",
                      "context": [], "supporting_facts": []}
            res = runner.run(sample)
            print(f"\nA: {res.predicted_answer}\n")
            print(f"  steps={res.total_steps} retrieved={res.retrieved_titles}")
        return

    samples = load_samples(args.num_samples, Path(args.samples_file))
    if not samples:
        print("No samples to evaluate. Run db_creator.py --build first.")
        return

    today = datetime.now().strftime("%Y-%m-%d")
    tag = f"{args.router}_{args.judge or 'no-judge'}_{today}".replace("/", "-")
    run_benchmark(
        samples, runner,
        output_tag=tag,
        runs_per_sample=args.runs,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
