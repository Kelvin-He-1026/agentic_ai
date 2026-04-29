"""
ReAct Agent — Router (Qwen3-8B) + Judge (Qwen3-14B)
=====================================================

A ReAct (Reason-Act-Observe) agent loop that combines two models:

  * **Router** (Qwen3-8B) — follows the ReAct pattern to reason
    step-by-step and select tools (Thought → Action → Action Input).
  * **Judge** (Qwen3-14B) — after each tool observation, decides whether
    the gathered information is sufficient to answer the question or
    whether more steps are needed.  The judge evaluates the observations
    against the question only (no ground truth).

Usage:
    python react_agent.py "What is the capital of France?"
    python react_agent.py "What is 42 * 17 + 3?"
    python react_agent.py --interactive
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

import sys
import torch
from dotenv import load_dotenv

# Make langGraph importable from part3/
sys.path.insert(0, str(Path(__file__).resolve().parent / "langGraph"))
from model_loader import load_llm

load_dotenv()

# ---------------------------------------------------------------------------
# Model defaults — preset names from model_loader or paths
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_ROUTER_MODEL = "qwen3-8b"
DEFAULT_JUDGE_MODEL = "qwen3-14b"

# ---------------------------------------------------------------------------
# Tools — add more here as needed
# ---------------------------------------------------------------------------
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Macro release watchlist — mapped by FRED release_id
# Fields: (label, tier, release_time_ET)
# Tier 1: Market-moving, high impact
# Tier 2: Important, moderate impact
# Tier 3: Regional / secondary
# NOTE: ISM Manufacturing/Services PMI were removed from FRED in 2016.
#       They are published directly by ISM at 10:00 AM ET (not tracked here).
MACRO_WATCHLIST: dict[int, tuple[str, int, str]] = {
    # --- Tier 1 ---
    10:  ("Consumer Price Index",                                 1, "8:30 AM"),
    54:  ("Personal Income and Outlays",                          1, "8:30 AM"),
    50:  ("Employment Situation",                                 1, "8:30 AM"),
    101: ("FOMC Press Release",                                   1, "2:00 PM"),
    53:  ("Gross Domestic Product",                               1, "8:30 AM"),
    9:   ("Advance Monthly Sales for Retail and Food Services",   1, "8:30 AM"),
    # --- Tier 2 ---
    46:  ("Producer Price Index",                                 2, "8:30 AM"),
    192: ("Job Openings and Labor Turnover Survey",               2, "10:00 AM"),
    180: ("Unemployment Insurance Weekly Claims Report",          2, "8:30 AM"),
    469: ("State Unemployment Insurance Weekly Claims Report",    2, "8:30 AM"),
    91:  ("Surveys of Consumers",                                 2, "10:00 AM"),
    27:  ("New Residential Construction",                         2, "8:30 AM"),
    291: ("Existing Home Sales",                                  2, "10:00 AM"),
    97:  ("New Residential Sales",                                2, "10:00 AM"),
    95:  ("Manufacturer's Shipments, Inventories, and Orders",    2, "8:30 AM"),
    # --- Tier 3 ---
    219: ("Chicago Fed National Activity Index",                  3, "8:30 AM"),
    221: ("Chicago Fed National Financial Conditions Index",      3, "8:30 AM"),
    321: ("Empire State Manufacturing Survey",                    3, "8:30 AM"),
    322: ("Business Leaders Survey",                              3, "8:30 AM"),
    351: ("Manufacturing Business Outlook Survey",                3, "8:30 AM"),
    352: ("Nonmanufacturing Business Outlook Survey",             3, "8:30 AM"),
    374: ("Texas Manufacturing Outlook Survey",                   3, "10:30 AM"),
    199: ("S&P Cotality Case-Shiller Home Price Indices",         3, "9:00 AM"),
    171: ("House Price Index",                                    3, "10:00 AM"),
    148: ("Housing Units Authorized By Building Permits",         3, "8:30 AM"),
    296: ("Housing Vacancies and Homeownership",                  3, "10:00 AM"),
    229: ("Construction Spending",                                3, "10:00 AM"),
    11:  ("Employment Cost Index",                                3, "8:30 AM"),
    13:  ("G.17 Industrial Production and Capacity Utilization",  3, "9:15 AM"),
    14:  ("G.19 Consumer Credit",                                 3, "3:00 PM"),
    194: ("ADP National Employment Report",                       3, "8:15 AM"),
    51:  ("U.S. International Trade in Goods and Services",       3, "8:30 AM"),
    188: ("U.S. Import and Export Price Indexes",                 3, "8:30 AM"),
}

TIER_LABELS = {1: "Tier 1", 2: "Tier 2", 3: "Tier 3"}
DATASET_DIR = Path(__file__).resolve().parent / "dataset"


def tool_fred_releases(query: str) -> str:
    """Fetch upcoming macroeconomic data releases from FRED.

    Input: 'this_week' for remaining releases this week,
           'next_week' for next week's releases.
    """
    if not FRED_API_KEY:
        return "Error: FRED_API_KEY environment variable not set. Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"

    try:
        import requests
    except ImportError:
        return "Error: 'requests' package not installed. Run: pip install requests"

    today = datetime.now(ET).date()
    weekday = today.weekday()  # Mon=0, Sun=6

    q = query.lower().strip()
    if "next" in q:
        start_date = today + timedelta(days=(7 - weekday))
        end_date = start_date + timedelta(days=4)
    else:
        start_date = today
        days_to_friday = 4 - weekday
        end_date = today + timedelta(days=max(days_to_friday, 0))

    url = "https://api.stlouisfed.org/fred/releases/dates"
    params = {
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "realtime_start": start_date.isoformat(),
        "realtime_end": end_date.isoformat(),
        "include_release_dates_with_no_data": "true",
        "sort_order": "asc",
        "limit": 1000,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return f"Error fetching FRED data: {e}"

    releases = data.get("release_dates", [])
    if not releases:
        return f"No upcoming releases found between {start_date} and {end_date}."

    # Filter to watchlist and collect rows
    rows: list[tuple[str, str, str, str]] = []  # (date, time_et, release, tier)
    for r in releases:
        rid = r["release_id"]
        if rid not in MACRO_WATCHLIST:
            continue
        label, tier, time_et = MACRO_WATCHLIST[rid]
        rows.append((r["date"], time_et, label, TIER_LABELS[tier]))

    if not rows:
        return f"No major macro releases scheduled between {start_date} and {end_date}."

    # Sort by date then time
    time_sort_key = lambda row: (row[0], datetime.strptime(row[1], "%I:%M %p"))
    rows.sort(key=time_sort_key)

    # Build markdown table
    md_lines = [
        f"# Macro Releases: {start_date} to {end_date}\n",
        f"Generated: {datetime.now(ET).strftime('%Y-%m-%d %I:%M %p')} ET\n",
        "| Date | Time (ET) | Release | Tier |",
        "|------|-----------|---------|------|",
    ]
    for date, time_et, label, tier in rows:
        md_lines.append(f"| {date} | {time_et} | {label} | {tier} |")

    md_content = "\n".join(md_lines) + "\n"

    # Save to dataset folder
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"macro_releases_{start_date}_to_{end_date}.md"
    filepath = DATASET_DIR / filename
    filepath.write_text(md_content)

    return f"{md_content}\nSaved to {filepath}"


def tool_tavily_search(query: str) -> str:
    """Search the web for information using Tavily."""
    if not TAVILY_API_KEY:
        return "Error: TAVILY_API_KEY environment variable not set. Get a key at https://tavily.com"

    try:
        from tavily import TavilyClient
    except ImportError:
        return "Error: 'tavily-python' package not installed. Run: pip install tavily-python"

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        results = client.search(
            query=query,
            search_depth="basic",
            topic="general",
            max_results=5,
            include_answer=True,
        )
    except Exception as e:
        return f"Error searching Tavily: {e}"

    lines = []
    answer = results.get("answer")
    if answer:
        lines.append(f"Summary: {answer}\n")

    lines.append("Sources:")
    for i, r in enumerate(results.get("results", []), 1):
        title = r.get("title", "No title")
        url = r.get("url", "")
        content = r.get("content", "")
        if len(content) > 300:
            content = content[:300] + "..."
        lines.append(f"\n{i}. {title}")
        lines.append(f"   URL: {url}")
        lines.append(f"   {content}")

    return "\n".join(lines)


DB_DIR = Path(__file__).resolve().parent / "db"
RETRIEVAL_COLLECTION_NAME = "documents"
RETRIEVAL_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# ---------------------------------------------------------------------------
# Pre-loaded retrieval components (singleton)
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
    if _retrieval_vectorstore is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma

        embedding_model = MODELS_DIR / "BAAI--bge-large-en-v1.5"
        embedding_model_name = (
            str(embedding_model) if embedding_model.exists()
            else RETRIEVAL_EMBEDDING_MODEL
        )
        embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
        chroma_dir = DB_DIR / "chroma"

        dim, count = _inspect_chroma_collection(
            chroma_dir, RETRIEVAL_COLLECTION_NAME
        )
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
            "Retrieval model ready "
            f"(collection={RETRIEVAL_COLLECTION_NAME}, dim={dim}, count={count}, device={embedding_device})."
        )

    return _retrieval_vectorstore


def tool_retrieve(query: str) -> str:
    """Retrieve relevant documents from the vector database using a natural language query."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
    except ImportError:
        return "Error: Required packages not installed. Run: pip install langchain-huggingface langchain-chroma"

    chroma_dir = DB_DIR / "chroma"
    if not chroma_dir.exists():
        return "Error: No vector database found at db/chroma. Build it first with vectorDB.py."

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
        content = doc.page_content
        # if len(content) > 500:
        #     content = content[:500] + "..."
        lines.append(content)
        lines.append("")

    return "\n".join(lines)


# Tool registry
TOOLS = {
    "fred_releases": {
        "fn": tool_fred_releases,
        "description": "List upcoming macroeconomic data releases from FRED. Input: 'this_week' for remaining releases this week, or 'next_week' for next week's releases.",
    },
    "web_search": {
        "fn": tool_tavily_search,
        "description": "Search the web for information, news, and current events. Input: a search query on any topic (e.g., 'latest AI news', 'S&P 500 market today', 'weather in New York').",
    },
    "retrieve": {
        "fn": tool_retrieve,
        "description": "Retrieve relevant documents from the knowledge base. Input: a natural language query about AI, machine learning, RAG, neural networks, or related research topics.",
    },
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
def build_system_prompt() -> str:
    today = datetime.now(ET).strftime("%Y-%m-%d (%A)")
    tool_descriptions = "\n".join(
        f"  - {name}: {info['description']}" for name, info in TOOLS.items()
    )
    return f"""You are a helpful assistant that solves problems step by step using the ReAct framework.

Today's date is {today}.

Available tools:
{tool_descriptions}

For each step, respond in EXACTLY this format:

Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: <input to the tool>

After you receive an Observation, continue with another Thought/Action or give the final answer:

Thought: <your reasoning>
Final Answer: <your complete answer to the user>

Rules:
- Always start with a Thought.
- Each response must be exactly one of these two forms:
  1. Thought + Action + Action Input
  2. Thought + Final Answer
- Never include both an Action and a Final Answer in the same response.
- Never write an Observation yourself. Observations are provided only by tool execution.
- If you choose an Action, stop after Action Input and wait for the Observation.
- When you have enough information, use "Final Answer:" to respond.
- Be concise."""


# ---------------------------------------------------------------------------
# ReAct Agent
# ---------------------------------------------------------------------------
class ReActAgent:
    """ReAct agent with a router LLM (Qwen3-8B) and a judge LLM (Qwen3-14B).

    The router follows the ReAct pattern (Thought → Action → Action Input) to
    select and invoke tools.  After each tool observation the judge evaluates
    whether the information gathered so far is sufficient to answer the
    original question.  If the judge says "sufficient", the router is prompted
    to emit a Final Answer; otherwise the loop continues.
    """

    def __init__(
        self,
        router_model_path: str | Path = DEFAULT_ROUTER_MODEL,
        judge_model_path: str | Path = DEFAULT_JUDGE_MODEL,
        device: str = "cuda",
        max_steps: int = 5,
        max_new_tokens: int = 2048,
    ):
        self.device = device
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens

        # --- Router model ---
        self.model, self.tokenizer = load_llm(
            str(router_model_path), device=device,
        )
        # Determine where to send inputs (OpenVINO models run on CPU)
        try:
            self.router_input_device = next(self.model.parameters()).device
        except (StopIteration, AttributeError):
            self.router_input_device = "cpu"

        # --- Judge model ---
        self.judge_model, self.judge_tokenizer = load_llm(
            str(judge_model_path), device=device,
        )
        try:
            self.judge_input_device = next(self.judge_model.parameters()).device
        except (StopIteration, AttributeError):
            self.judge_input_device = "cpu"

    def _generate(self, messages: list[dict]) -> tuple[str, dict]:
        """Run a single generation turn on the router model.

        Returns (text, stats) where stats has:
            prompt_tokens   — input length in tokens
            output_tokens   — number of newly generated tokens
            thinking_tokens — tokens inside <think>...</think>, if any
            truncated       — True if output_tokens hit max_new_tokens
        """
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.router_input_device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        new_token_ids = output[0][prompt_tokens:]
        output_tokens = int(new_token_ids.shape[0])
        truncated = output_tokens >= self.max_new_tokens

        # Count <think>...</think> content tokens, if present
        raw_with_specials = self.tokenizer.decode(new_token_ids, skip_special_tokens=False)
        think_blocks = re.findall(r"<think>(.*?)</think>", raw_with_specials, re.DOTALL)
        thinking_tokens = sum(
            len(self.tokenizer.encode(b, add_special_tokens=False)) for b in think_blocks
        )

        decoded = self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        stats = {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "truncated": truncated,
        }
        return decoded, stats

    def _parse_action(self, text: str) -> tuple[str | None, str | None]:
        """Extract Action and Action Input from model output."""
        action_match = re.search(r"Action:\s*(.+)", text)
        input_match = re.search(r"Action Input:\s*(.+)", text)
        if action_match and input_match:
            return action_match.group(1).strip(), input_match.group(1).strip()
        return None, None

    def _parse_final_answer(self, text: str) -> str | None:
        """Extract Final Answer from model output."""
        match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    # ------------------------------------------------------------------
    # Judge helpers
    # ------------------------------------------------------------------
    def _judge_generate(self, messages: list[dict], max_new_tokens: int = 128) -> tuple[str, dict]:
        """Run a single generation turn on the judge model.

        Returns (text, stats) — same stats schema as `_generate`.
        """
        text = self.judge_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.judge_tokenizer(text, return_tensors="pt").to(self.judge_input_device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        with torch.no_grad():
            output = self.judge_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        new_token_ids = output[0][prompt_tokens:]
        output_tokens = int(new_token_ids.shape[0])
        truncated = output_tokens >= max_new_tokens

        raw_with_specials = self.judge_tokenizer.decode(new_token_ids, skip_special_tokens=False)
        think_blocks = re.findall(r"<think>(.*?)</think>", raw_with_specials, re.DOTALL)
        thinking_tokens = sum(
            len(self.judge_tokenizer.encode(b, add_special_tokens=False)) for b in think_blocks
        )

        result = self.judge_tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
        stats = {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "truncated": truncated,
        }
        return result, stats

    def _judge_sufficiency(
        self, question: str, observations: list[str],
    ) -> tuple[bool, str, dict]:
        """Ask the judge model whether the gathered observations are sufficient
        to answer the question.

        Returns (is_sufficient, raw_judge_response, judge_stats).
        """
        obs_text = "\n\n".join(
            f"[Observation {i}]\n{obs}" for i, obs in enumerate(observations, 1)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a judge that decides whether the information gathered "
                    "so far is sufficient to answer the user's question.\n\n"
                    "You will be given:\n"
                    "  1. The original QUESTION.\n"
                    "  2. One or more OBSERVATIONS from tool calls.\n\n"
                    "Decide whether the observations contain enough information to "
                    "provide a complete and accurate answer to the question.\n\n"
                    "Respond with exactly one word: SUFFICIENT or CONTINUE.\n"
                    "  - SUFFICIENT: The observations already contain enough "
                    "information to fully answer the question.\n"
                    "  - CONTINUE: More information is needed; the agent should "
                    "call additional tools."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{question}\n\n"
                    f"OBSERVATIONS:\n{obs_text}"
                ),
            },
        ]
        response, stats = self._judge_generate(messages)
        is_sufficient = "sufficient" in response.lower()
        return is_sufficient, response, stats

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, question: str) -> str:
        """Run the ReAct loop on a question. Returns the final answer."""
        system_prompt = build_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        observations: list[str] = []  # collected tool observations for the judge

        for step in range(1, self.max_steps + 1):
            print(f"\n--- Step {step} ---")

            response, _gen_stats = self._generate(messages)
            print(response)

            # Parse action first. The model sometimes includes a hallucinated
            # Observation and Final Answer in the same generation as an
            # Action, and we want the real tool execution to win.
            action, action_input = self._parse_action(response)
            if action is not None:
                # Execute tool
                tool = TOOLS.get(action.lower())
                if tool is None:
                    observation = f"Error: Unknown tool '{action}'. Available: {', '.join(TOOLS.keys())}"
                else:
                    observation = tool["fn"](action_input)

                print(f"Observation: {observation}")
                observations.append(observation)

                # --- Judge: is the gathered info sufficient? ---
                sufficient, judge_response, _judge_stats = self._judge_sufficiency(
                    question, observations,
                )
                print(f"Judge: {judge_response} -> {'SUFFICIENT' if sufficient else 'CONTINUE'}")

                messages.append({"role": "assistant", "content": response})

                if sufficient:
                    # Tell the router the judge is satisfied — ask for Final Answer
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Observation: {observation}\n\n"
                            "The information gathered is sufficient. "
                            "Please provide your Final Answer now."
                        ),
                    })
                else:
                    # Normal ReAct continuation
                    messages.append({
                        "role": "user",
                        "content": f"Observation: {observation}",
                    })
                continue

            # Check for final answer only if there was no action.
            final = self._parse_final_answer(response)
            if final:
                return final

            if action is None:
                # Model didn't follow format — ask it to try again
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": "Please respond with either an Action or a Final Answer.",
                })
                continue

        return "Reached maximum steps without a final answer."


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ReAct Agent with Qwen3-8B router + Qwen3-14B judge")
    parser.add_argument("question", nargs="?", default=None,
                        help="Question to answer")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode — ask multiple questions")
    parser.add_argument("--device", default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--max-steps", type=int, default=5,
                        help="Maximum ReAct steps (default: 5)")
    parser.add_argument("--router-model", default=DEFAULT_ROUTER_MODEL,
                        help=f"Router model: preset name or path (default: {DEFAULT_ROUTER_MODEL})")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                        help=f"Judge model: preset name or path (default: {DEFAULT_JUDGE_MODEL})")
    args = parser.parse_args()

    agent = ReActAgent(
        router_model_path=args.router_model,
        judge_model_path=args.judge_model,
        device=args.device,
        max_steps=args.max_steps,
    )

    if args.interactive:
        print("\nReAct Agent (type 'quit' to exit)")
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
    elif args.question:
        answer = agent.run(args.question)
        print(f"\nAnswer: {answer}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
