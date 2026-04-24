"""
QA Ground Truth Generator
=========================
Generates ground-truth QA pairs for benchmarking the ReAct agent.

For each tool (fred_releases, web_search), generates 5 questions by calling
the actual tool API and constructing QA pairs from real output. No LLM needed.

Each QA entry contains:
  - question:           the natural language question
  - expected_tool:      which tool should be called (fred_releases / web_search)
  - reference_context:  the raw tool output used to generate the QA pair
  - reference_answer:   the expected answer derived from the context

Usage:
  python qa_generation.py                          # generate and save
  python qa_generation.py --output path/to/qa.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"

# Make sure react_agent is importable
sys.path.insert(0, str(PROJECT_DIR))


# ---------------------------------------------------------------------------
# Fred releases tool — 5 questions (calls FRED API, QA crafted by Claude)
# ---------------------------------------------------------------------------
FRED_QA_TEMPLATES = [
    {
        "tool_input": "this_week",
        "question": "What major economic data releases are scheduled for this week?",
        "answer_template": (
            "This week's major releases include: {releases}. "
            "Key Tier 1 releases are {tier1}."
        ),
    },
    {
        "tool_input": "next_week",
        "question": "What macroeconomic reports are coming out next week?",
        "answer_template": (
            "Next week's scheduled releases include: {releases}. "
            "Notable Tier 1 events are {tier1}."
        ),
    },
    {
        "tool_input": "this_week",
        "question": "Are there any Tier 1 economic releases remaining this week?",
        "answer_template": (
            "Yes, this week's Tier 1 releases are: {tier1}."
        ),
    },
    {
        "tool_input": "next_week",
        "question": "Which Federal Reserve or employment reports are due next week?",
        "answer_template": (
            "Next week's Fed and employment-related releases include: {fed_employment}."
        ),
    },
    {
        "tool_input": "this_week",
        "question": "What inflation or GDP data is being released this week?",
        "answer_template": (
            "This week's inflation and GDP releases include: {inflation_gdp}."
        ),
    },
]


def _parse_fred_table(context: str) -> dict:
    """Extract structured data from FRED markdown table output."""
    releases = []
    tier1 = []
    fed_employment = []
    inflation_gdp = []

    for line in context.split("\n"):
        if not line.startswith("|") or "Date" in line or "---" in line:
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < 4:
            continue
        date, time_et, name, tier = parts[0], parts[1], parts[2], parts[3]
        entry = f"{name} ({date} {time_et})"
        releases.append(entry)
        if tier == "Tier 1":
            tier1.append(entry)
        if any(kw in name.lower() for kw in ["fomc", "employment", "unemployment", "labor"]):
            fed_employment.append(entry)
        if any(kw in name.lower() for kw in ["consumer price", "producer price",
                                               "gdp", "gross domestic", "inflation",
                                               "personal income"]):
            inflation_gdp.append(entry)

    return {
        "releases": ", ".join(releases[:8]) if releases else "none found",
        "tier1": ", ".join(tier1) if tier1 else "none scheduled",
        "fed_employment": ", ".join(fed_employment) if fed_employment else "none scheduled",
        "inflation_gdp": ", ".join(inflation_gdp) if inflation_gdp else "none scheduled",
    }


def _call_with_retry(fn, arg, retries: int = 3) -> str:
    """Call a tool function with retries on transient errors."""
    import time
    for attempt in range(retries):
        result = fn(arg)
        if not result.startswith("Error"):
            return result
        if attempt < retries - 1:
            print(f"    Retry {attempt+1}/{retries-1} after error: {result[:60]}")
            time.sleep(2)
    return result


def generate_fred_qa(**_kwargs) -> list[dict]:
    """Generate 5 QA pairs using the fred_releases tool. No LLM needed."""
    from react_agent import tool_fred_releases

    qa_pairs = []
    for i, template in enumerate(FRED_QA_TEMPLATES):
        tool_input = template["tool_input"]
        print(f"  [fred_releases {i+1}/5] Querying: {tool_input}")
        context = _call_with_retry(tool_fred_releases, tool_input)

        if context.startswith("Error"):
            print(f"    Skipped — {context[:80]}")
            continue

        parsed = _parse_fred_table(context)
        reference_answer = template["answer_template"].format(**parsed)

        qa_pairs.append({
            "question": template["question"],
            "expected_tool": "fred_releases",
            "reference_context": context,
            "reference_answer": reference_answer,
        })
        print(f"    Generated: {template['question'][:80]}")

    return qa_pairs


# ---------------------------------------------------------------------------
# Web search tool — 5 questions (calls Tavily API, QA crafted by Claude)
# ---------------------------------------------------------------------------
WEB_SEARCH_QA_TEMPLATES = [
    {
        "search_query": "latest Federal Reserve interest rate decision",
        "question": "What was the Federal Reserve's most recent interest rate decision?",
        "answer_extract": "summary",
    },
    {
        "search_query": "S&P 500 stock market performance today",
        "question": "How is the S&P 500 performing today?",
        "answer_extract": "summary",
    },
    {
        "search_query": "recent breakthroughs in artificial intelligence",
        "question": "What are some recent breakthroughs in artificial intelligence?",
        "answer_extract": "summary",
    },
    {
        "search_query": "global oil prices and energy market news",
        "question": "What is happening with global oil prices right now?",
        "answer_extract": "summary",
    },
    {
        "search_query": "latest unemployment rate United States",
        "question": "What is the current unemployment rate in the United States?",
        "answer_extract": "summary",
    },
]


def _extract_tavily_summary(context: str) -> str:
    """Extract the Summary line from Tavily search output."""
    match = re.search(r"Summary:\s*(.+?)(?:\n|$)", context)
    if match:
        return match.group(1).strip()
    # Fallback: first 300 chars
    return context[:300]


def generate_web_search_qa(**_kwargs) -> list[dict]:
    """Generate 5 QA pairs using the web_search tool. No LLM needed."""
    from react_agent import tool_tavily_search

    qa_pairs = []
    for i, template in enumerate(WEB_SEARCH_QA_TEMPLATES):
        query = template["search_query"]
        print(f"  [web_search {i+1}/5] Querying: {query}")
        context = _call_with_retry(tool_tavily_search, query)

        if context.startswith("Error"):
            print(f"    Skipped — {context[:80]}")
            continue

        reference_answer = _extract_tavily_summary(context)

        qa_pairs.append({
            "question": template["question"],
            "expected_tool": "web_search",
            "reference_context": context,
            "reference_answer": reference_answer,
        })
        print(f"    Generated: {template['question'][:80]}")

    return qa_pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate_all(output_path: Path | None = None) -> list[dict]:
    """Generate 10 QA pairs (5 per tool) and save to JSON."""
    print("=" * 60)
    print("Generating ground-truth QA pairs (5 per tool)")
    print("=" * 60)

    print("\n--- fred_releases ---")
    fred_qa = generate_fred_qa()

    print("\n--- web_search ---")
    web_qa = generate_web_search_qa()

    all_qa = fred_qa + web_qa

    if output_path is None:
        output_path = DATASET_DIR / "react_benchmark_qa.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_qa, indent=2))

    print(f"\n{'='*60}")
    print(f"Generated {len(all_qa)} QA pairs:")
    print(f"  fred_releases: {len(fred_qa)}")
    print(f"  web_search:    {len(web_qa)}")
    print(f"Saved to {output_path}")
    print(f"{'='*60}")

    return all_qa


def main():
    parser = argparse.ArgumentParser(description="Generate QA ground truth for ReAct benchmark")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: dataset/react_benchmark_qa.json)")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    generate_all(output_path=output_path)


if __name__ == "__main__":
    main()
