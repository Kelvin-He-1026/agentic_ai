# agentic_rag

Agentic RAG over a local PDF corpus, with a ReAct-style tool-using agent (Router LLM + Judge LLM) and a benchmarking harness for comparing models, quantization variants, and devices.

## What's here

```
agentic_rag/
├── react_agent.py              # ReAct agent: Router (Qwen3-8B) + Judge (Qwen3-14B)
├── react_agent_benchmark.py    # Run the ReAct agent across a QA set, log per-step + per-question metrics
├── react_llm.py                # LLM-only ReAct baseline (no judge)
├── qaagent_benchmark.py        # Earlier QA-agent benchmark harness
├── qa_generation.py            # Build QA pairs from the document corpus (ground truth)
├── llm_judge.py                # LLM-as-judge evaluator (scores hypothesis vs. reference)
├── tavily_mcp.py               # MCP server wrapping Tavily web search
├── langGraph_drawing.py        # Render the LangGraph state machines as PNGs
├── djt.py                      # Misc helper (Trump truth-social puller used by one tool)
├── langGraph/
│   ├── vectorDB.py             # Chroma vector store builder over documents/
│   ├── model_loader.py         # HF + OpenVINO model loader (mirrors speech_ai/model_loader.py)
│   └── retrieval_router/       # TF-IDF / embedding routing layer
├── dataset/                    # Ground-truth JSON + accumulated benchmark CSVs
├── db/chroma/                  # Persisted vector store (gitignored, rebuild from documents/)
├── documents/                  # Source PDFs (gitignored)
├── figures/, langGraph/figures/   # Figures auto-extracted from PDFs (gitignored)
├── diagrams/                   # Architecture diagrams (PNG, committed)
├── models/                     # Auto-downloaded model weights (gitignored, ~100 GB+)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

`unstructured[pdf]` needs system packages for OCR/layout:
```bash
sudo apt install poppler-utils tesseract-ocr libgl1
```

Create `.env` (do **not** commit) with whatever the modules need:
```ini
TAVILY_API_KEY=tvly-...
HF_TOKEN=hf_...                 # for gated HF repos
# any other API keys your tools use
```

## One-time corpus prep

1. Drop the PDFs you want to query into `documents/`.
2. Build (or rebuild) the vector store:
   ```bash
   python -m langGraph.vectorDB
   ```
   This populates `db/chroma/` and `figures/` (extracted images) and writes `db/manifest.json` with content hashes.
3. (Optional) Generate a QA evaluation set against the corpus:
   ```bash
   python qa_generation.py
   ```
   Output: `dataset/rag_ground_truth_questions.json` (or similar).

## Run the agent

Single question:
```bash
python react_agent.py "What is the main contribution of the Manu RAG paper?"
```

Interactive REPL:
```bash
python react_agent.py --interactive
```

LLM-only baseline (no agent loop):
```bash
python react_llm.py "..."
```

## Benchmark

Sweep a model pair (router × judge) across the QA set:
```bash
python react_agent_benchmark.py \
    --router OpenVINO/Qwen3-8B-int4-ov \
    --judge  qwen3-14b \
    --device cpu
```
Per-question raw output → `dataset/react_agent_benchmark_raw_<router>_<judge>_<date>.csv`
Per-step trace → `dataset/react_agent_benchmark_step_<router>_<judge>_<date>.csv`

Then judge the outputs after the fact:
```bash
python llm_judge.py dataset/react_agent_benchmark_raw_<...>.csv
# → ..._judged.csv with per-row scores and rationales
```

## Architectures

`diagrams/` holds three architectures that have been built/compared:
1. `1_judge_based_iterative_retrieval.png` — Judge stops the loop when context is sufficient
2. `2_react_style_agentic_rag.png` — Tool-using ReAct agent (current main pipeline)
3. `3_router_judge_agentic_rag.png` — Router (small LLM) + Judge (larger LLM) split

## Tools available to the agent

- **vector_search** — semantic retrieval over `db/chroma`
- **tavily_search** — web search via `tavily_mcp.py`
- **calculator** / **datetime** / **truth_social** — small utilities

(See the `TOOLS` registry near the top of `react_agent.py`.)

## Notes

- `langGraph/model_loader.py` follows the same registry pattern as `speech_ai/model_loader.py`: short alias → HF repo, downloads only safetensors / OpenVINO IRs (skips `pytorch_model.bin`, `tf_model.h5`, `flax_model.msgpack`).
- `db/manifest.json` records the SHA-256 of each PDF that was ingested. The vector store builder skips files whose hash matches the manifest, so re-running ingestion is cheap.
- Benchmark CSVs accumulate in `dataset/` and are ignored by default. Remove the `dataset/*.csv` line from `.gitignore` if you want to track them.
- The 100 GB+ `models/` folder is gitignored; the loader will re-download anything missing on the first run that needs it.
