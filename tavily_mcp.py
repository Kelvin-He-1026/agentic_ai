"""
Tavily MCP Server with Local Qwen3-8B
======================================
An MCP (Model Context Protocol) server that exposes Tavily web search
and extraction tools, optionally augmented by local Qwen3-8B inference
for summarization and deep research.

Tools provided:
  - tavily_search          : Raw web search via Tavily API
  - tavily_extract         : Extract content from URLs via Tavily API
  - search_and_summarize   : Search + Qwen3-8B summarization
  - research_topic         : Multi-query deep research with LLM synthesis

Usage:
  # Run directly (stdio transport)
  python tavily_mcp.py

  # Test with MCP Inspector
  npx @modelcontextprotocol/inspector python tavily_mcp.py

  # Claude Desktop config (~/.config/claude/claude_desktop_config.json):
  {
    "mcpServers": {
      "tavily-qwen": {
        "command": "/home/lenovoai/kelvin/agent_env/bin/python",
        "args": ["/home/lenovoai/kelvin/part3/tavily_mcp.py"]
      }
    }
  }
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Logging — must go to stderr (stdout is reserved for MCP JSON-RPC)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv(Path(__file__).resolve().parent / ".env")

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "tavily-qwen",
    instructions="Tavily web search with local Qwen3-8B summarization",
)

# ---------------------------------------------------------------------------
# Tavily client (lazy)
# ---------------------------------------------------------------------------
_tavily_client = None


def _get_tavily_client():
    global _tavily_client
    if _tavily_client is None:
        from tavily import TavilyClient

        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise ValueError(
                "TAVILY_API_KEY not set. Add it to .env or export it."
            )
        _tavily_client = TavilyClient(api_key=api_key)
        log.info("Tavily client initialized")
    return _tavily_client


# ---------------------------------------------------------------------------
# Local LLM — Qwen3-8B (lazy loaded on first LLM tool call)
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None
_device = None


def _ensure_model_loaded():
    """Load Qwen3-8B on first use. GPU default, CPU fallback."""
    global _model, _tokenizer, _device
    if _model is not None:
        return

    # Add langGraph/ to path so we can import model_loader
    lang_graph_dir = str(Path(__file__).resolve().parent / "langGraph")
    if lang_graph_dir not in sys.path:
        sys.path.insert(0, lang_graph_dir)

    from model_loader import load_llm

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if _device == "cuda" else torch.float32
    log.info(f"Loading Qwen3-8B on {_device} ...")

    # Redirect stdout → stderr during loading to keep MCP stdio clean
    with contextlib.redirect_stdout(sys.stderr):
        _model, _tokenizer = load_llm(
            "qwen3-8b", device=_device, torch_dtype=dtype,
        )

    log.info("Qwen3-8B loaded successfully")


def _generate(system_msg: str, user_msg: str, max_new_tokens: int = 512) -> str:
    """Run a single generation turn with Qwen3-8B."""
    _ensure_model_loaded()

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = _tokenizer(text, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    result = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Strip Qwen3 thinking tags if present
    result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
    return result


# ---------------------------------------------------------------------------
# Helper — format Tavily results
# ---------------------------------------------------------------------------
def _format_search_results(results: dict) -> str:
    """Format Tavily search results into readable text."""
    lines = []

    answer = results.get("answer")
    if answer:
        lines.append(f"**Tavily Answer:** {answer}\n")

    lines.append("**Sources:**")
    for i, r in enumerate(results.get("results", []), 1):
        title = r.get("title", "No title")
        url = r.get("url", "")
        content = r.get("content", "")
        if len(content) > 500:
            content = content[:500] + "..."
        lines.append(f"\n[{i}] {title}")
        lines.append(f"    URL: {url}")
        lines.append(f"    {content}")

    return "\n".join(lines)


def _truncate_for_context(text: str, max_chars: int = 60000) -> str:
    """Truncate text to fit within Qwen3-8B context window (~20K tokens)."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n... [truncated for context length] ..."


# ---------------------------------------------------------------------------
# MCP Tool 1: tavily_search (no LLM)
# ---------------------------------------------------------------------------
@mcp.tool()
def tavily_search(
    query: str,
    search_depth: str = "basic",
    topic: str = "general",
    max_results: int = 5,
    days: int = 0,
) -> str:
    """Search the web using Tavily API.

    Args:
        query: The search query.
        search_depth: "basic" (fast) or "advanced" (thorough).
        topic: Search topic — "general", "news", or "finance".
        max_results: Number of results to return (1-20).
        days: Limit results to the last N days (0 = no limit).
    """
    client = _get_tavily_client()

    kwargs = {
        "query": query,
        "search_depth": search_depth,
        "topic": topic,
        "max_results": max_results,
        "include_answer": True,
    }
    if days > 0:
        kwargs["days"] = days

    log.info(f"tavily_search: query={query!r}, depth={search_depth}")
    results = client.search(**kwargs)
    return _format_search_results(results)


# ---------------------------------------------------------------------------
# MCP Tool 2: tavily_extract (no LLM)
# ---------------------------------------------------------------------------
@mcp.tool()
def tavily_extract(urls: str) -> str:
    """Extract content from one or more web pages using Tavily API.

    Args:
        urls: Comma-separated URLs to extract content from.
    """
    client = _get_tavily_client()
    url_list = [u.strip() for u in urls.split(",") if u.strip()]

    if not url_list:
        return "Error: No URLs provided."

    log.info(f"tavily_extract: {len(url_list)} URL(s)")
    results = client.extract(urls=url_list)

    lines = []
    for r in results.get("results", []):
        url = r.get("url", "")
        raw_content = r.get("raw_content", "")
        if len(raw_content) > 2000:
            raw_content = raw_content[:2000] + "..."
        lines.append(f"**URL:** {url}")
        lines.append(f"{raw_content}\n")

    return "\n".join(lines) if lines else "No content extracted."


# ---------------------------------------------------------------------------
# MCP Tool 3: search_and_summarize (Tavily + Qwen3-8B)
# ---------------------------------------------------------------------------
@mcp.tool()
def search_and_summarize(
    query: str,
    topic: str = "general",
    max_results: int = 5,
) -> str:
    """Search the web and summarize results using local Qwen3-8B.

    Combines Tavily web search with local LLM summarization.
    The model cites sources by number.

    Args:
        query: The search query / question to answer.
        topic: Search topic — "general", "news", or "finance".
        max_results: Number of search results to feed to the LLM.
    """
    # Step 1: Search
    client = _get_tavily_client()
    log.info(f"search_and_summarize: query={query!r}")

    results = client.search(
        query=query,
        search_depth="advanced",
        topic=topic,
        max_results=max_results,
        include_answer=True,
    )

    formatted = _format_search_results(results)
    formatted = _truncate_for_context(formatted)

    # Step 2: Summarize with Qwen3-8B
    system_prompt = (
        "You are a research assistant. Summarize the following search results "
        "to answer the user's query accurately and concisely. "
        "Cite sources by their number in brackets, e.g. [1], [2]. "
        "If the results don't contain enough information, say so."
    )
    user_prompt = f"Query: {query}\n\nSearch Results:\n{formatted}"

    summary = _generate(system_prompt, user_prompt, max_new_tokens=512)
    return f"## Summary\n\n{summary}\n\n---\n\n## Raw Sources\n\n{formatted}"


# ---------------------------------------------------------------------------
# MCP Tool 4: research_topic (multi-query Tavily + Qwen3-8B synthesis)
# ---------------------------------------------------------------------------
@mcp.tool()
def research_topic(
    topic: str,
    num_searches: int = 3,
) -> str:
    """Deep research on a topic using multiple searches and LLM synthesis.

    Uses Qwen3-8B to generate diverse sub-queries, runs Tavily searches
    for each, then synthesizes all results into a comprehensive report.

    Args:
        topic: The research topic or question.
        num_searches: Number of diverse sub-queries to generate (1-5).
    """
    num_searches = max(1, min(5, num_searches))
    log.info(f"research_topic: topic={topic!r}, num_searches={num_searches}")

    # Step 1: Generate diverse sub-queries with Qwen3-8B
    query_prompt = (
        f"Generate exactly {num_searches} diverse search queries to thoroughly "
        f"research the following topic. Each query should explore a different "
        f"angle or aspect. Return ONLY the queries, one per line, no numbering."
    )
    raw_queries = _generate(query_prompt, topic, max_new_tokens=256)
    sub_queries = [
        q.strip().strip('"').strip("'")
        for q in raw_queries.strip().split("\n")
        if q.strip() and len(q.strip()) > 5
    ][:num_searches]

    # Fallback if model didn't produce enough queries
    if len(sub_queries) < 1:
        sub_queries = [topic]

    log.info(f"Sub-queries: {sub_queries}")

    # Step 2: Run Tavily search for each sub-query
    client = _get_tavily_client()
    all_results = []

    for i, query in enumerate(sub_queries, 1):
        try:
            results = client.search(
                query=query,
                search_depth="advanced",
                max_results=3,
                include_answer=True,
            )
            section = f"### Search {i}: {query}\n\n"
            section += _format_search_results(results)
            all_results.append(section)
        except Exception as e:
            log.warning(f"Search failed for query {i}: {e}")
            all_results.append(f"### Search {i}: {query}\n\nError: {e}")

    combined = "\n\n".join(all_results)
    combined = _truncate_for_context(combined)

    # Step 3: Synthesize with Qwen3-8B
    system_prompt = (
        "You are a research analyst. Synthesize the following search results "
        "into a comprehensive, well-structured research report. "
        "Organize by themes, cite sources by their number in brackets, "
        "highlight key findings, and note any conflicting information. "
        "End with a brief conclusion."
    )
    user_prompt = f"Research Topic: {topic}\n\nSearch Results:\n{combined}"

    report = _generate(system_prompt, user_prompt, max_new_tokens=1024)

    return (
        f"# Research Report: {topic}\n\n"
        f"{report}\n\n"
        f"---\n\n"
        f"## Sub-queries Used\n\n"
        + "\n".join(f"- {q}" for q in sub_queries)
        + f"\n\n## Raw Search Results\n\n{combined}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info("Starting Tavily-Qwen MCP server (stdio transport)")
    mcp.run()
