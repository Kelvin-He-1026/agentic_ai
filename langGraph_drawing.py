"""
LangGraph Architecture Diagrams
================================
Draws and saves the agent loop diagrams for three agentic RAG architectures
using LangGraph's StateGraph visualization.

1. Judge-Based Iterative Retrieval  (qaagent_benchmark.py)
   - Agent model (Qwen3-8B): retrieve, validate, find_missing, answer
   - Judge model (Qwen3-14B): score retrieval & answer quality

2. ReAct-Style Agentic RAG  (react_llm.py)
   - Single model: Thought → Action → Observe loop with retrieve tool

3. Router + Judge Agentic RAG  (react_agent_benchmark.py)
   - Router model (Qwen3-8B): ReAct-style tool selection
   - Judge model (Qwen3-14B): decides SUFFICIENT / CONTINUE after each tool call

Usage:
    python langGraph_drawing.py              # save all three diagrams
    python langGraph_drawing.py --output-dir diagrams/  # custom output dir
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

OUTPUT_DIR = Path(__file__).resolve().parent / "diagrams"


# ============================================================
# Shared state types (minimal — only for graph structure)
# ============================================================

class JudgeIterativeState(TypedDict):
    question: str
    context: str
    status: str
    answer: str


class ReActState(TypedDict):
    question: str
    messages: list
    answer: str


class RouterJudgeState(TypedDict):
    question: str
    messages: list
    observations: list
    judge_decision: str
    answer: str


# Placeholder node function
def _noop(state: dict) -> dict:
    return state


# ============================================================
# 1. Judge-Based Iterative Retrieval (qaagent_benchmark.py)
#    Agent model: retrieve, validate, find_missing, answer
#    Judge model: score retrieval & answer quality
# ============================================================

def build_judge_iterative_graph() -> StateGraph:
    """
    Flow: Retrieve → Validate (Judge) → [INCOMPLETE? → Find Missing (Judge) → Retrieve] → Answer → END
    Agent model does retrieve/answer; Judge model validates and identifies missing info.
    """
    graph = StateGraph(JudgeIterativeState)

    # --- Agent model nodes ---
    graph.add_node("Retrieve - Agent Model", _noop)
    graph.add_node("Generate Answer - Agent Model", _noop)

    # --- Judge model nodes ---
    graph.add_node("Validate Retrieval - Judge Model", _noop)
    graph.add_node("Find Missing Info - Judge Model", _noop)

    # --- Edges ---
    graph.add_edge(START, "Retrieve - Agent Model")
    graph.add_edge("Retrieve - Agent Model", "Validate Retrieval - Judge Model")

    # Conditional: COMPLETE → Answer, INCOMPLETE → Find Missing
    graph.add_conditional_edges(
        "Validate Retrieval - Judge Model",
        lambda s: "incomplete" if s.get("status") == "INCOMPLETE" else "complete",
        {
            "incomplete": "Find Missing Info - Judge Model",
            "complete": "Generate Answer - Agent Model",
        },
    )

    # Find missing loops back to Retrieve
    graph.add_edge("Find Missing Info - Judge Model", "Retrieve - Agent Model")

    # Answer → END
    graph.add_edge("Generate Answer - Agent Model", END)

    return graph


# ============================================================
# 2. ReAct-Style Agentic RAG (react_llm.py)
#    Single model does everything: Thought → Action → Observe
# ============================================================

def build_react_single_model_graph() -> StateGraph:
    """
    Flow: Think → [Action? → Execute Tool → Observe → Think] → Final Answer
    One model handles reasoning, tool selection, and answer generation.
    """
    graph = StateGraph(ReActState)

    graph.add_node("Thought - LLM", _noop)
    graph.add_node("Execute Tool - retrieve", _noop)
    graph.add_node("Observe Result", _noop)
    graph.add_node("Final Answer", _noop)

    # Start → Think
    graph.add_edge(START, "Thought - LLM")

    # Think → Action (tool) or Final Answer
    graph.add_conditional_edges(
        "Thought - LLM",
        lambda s: "action" if s.get("answer") == "" else "final",
        {
            "action": "Execute Tool - retrieve",
            "final": "Final Answer",
        },
    )

    # Tool → Observe → Think (loop)
    graph.add_edge("Execute Tool - retrieve", "Observe Result")
    graph.add_edge("Observe Result", "Thought - LLM")

    # Final Answer → END
    graph.add_edge("Final Answer", END)

    return graph


# ============================================================
# 3. Router + Judge Agentic RAG (react_agent_benchmark.py)
#    Router model (Qwen3-8B): ReAct tool selection
#    Judge model (Qwen3-14B): SUFFICIENT / CONTINUE decision
# ============================================================

def build_router_judge_graph() -> StateGraph:
    """
    Flow: Router Think → Action → Execute Tool → Judge Sufficiency
          → [SUFFICIENT → Router Final Answer] / [CONTINUE → Router Think]
    """
    graph = StateGraph(RouterJudgeState)

    graph.add_node("Thought - Router Model", _noop)
    graph.add_node("Execute Tool - retrieve, web_search, fred_releases", _noop)
    graph.add_node("Judge Sufficiency - Judge Model", _noop)
    graph.add_node("Final Answer - Router Model", _noop)

    # Start → Router thinks
    graph.add_edge(START, "Thought - Router Model")

    # Router → Action (tool) or Final Answer
    graph.add_conditional_edges(
        "Thought - Router Model",
        lambda s: "action" if s.get("answer") == "" else "final",
        {
            "action": "Execute Tool - retrieve, web_search, fred_releases",
            "final": "Final Answer - Router Model",
        },
    )

    # Tool → Judge decides
    graph.add_edge(
        "Execute Tool - retrieve, web_search, fred_releases",
        "Judge Sufficiency - Judge Model",
    )

    # Judge → SUFFICIENT (prompt Final Answer) or CONTINUE (loop back)
    graph.add_conditional_edges(
        "Judge Sufficiency - Judge Model",
        lambda s: s.get("judge_decision", "CONTINUE"),
        {
            "SUFFICIENT": "Final Answer - Router Model",
            "CONTINUE": "Thought - Router Model",
        },
    )

    # Final Answer → END
    graph.add_edge("Final Answer - Router Model", END)

    return graph


# ============================================================
# Save helpers
# ============================================================

def save_graph(graph: StateGraph, name: str, output_dir: Path) -> Path:
    """Compile a StateGraph and save its diagram as PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    compiled = graph.compile()
    png_bytes = compiled.get_graph().draw_mermaid_png()
    out_path = output_dir / f"{name}.png"
    out_path.write_bytes(png_bytes)
    print(f"  Saved: {out_path} ({len(png_bytes):,} bytes)")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Draw and save LangGraph diagrams for 3 agentic RAG architectures",
    )
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_DIR),
        help=f"Directory to save PNG diagrams (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    architectures = [
        (
            "1_judge_based_iterative_retrieval",
            "1. Judge-Based Iterative Retrieval (qaagent_benchmark.py)",
            build_judge_iterative_graph,
        ),
        (
            "2_react_style_agentic_rag",
            "2. ReAct-Style Agentic RAG (react_llm.py)",
            build_react_single_model_graph,
        ),
        (
            "3_router_judge_agentic_rag",
            "3. Router + Judge Agentic RAG (react_agent_benchmark.py)",
            build_router_judge_graph,
        ),
    ]

    print(f"Saving diagrams to {output_dir}/\n")
    for filename, label, builder in architectures:
        print(f"{label}")
        graph = builder()
        save_graph(graph, filename, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
