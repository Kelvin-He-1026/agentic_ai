"""
Retrieval Router Benchmark — LangGraph Single-Agent, CPU + GPU Profiling
=========================================================================
Hardware: Intel Xeon 5th Gen + NVIDIA H200
Framework: LangGraph (StateGraph)

Flow:
  interpret query → decide (retrieve or direct) → [retrieve] → generate (GPU)

Demonstrates that CPU remains critical even with GPU-accelerated LLM inference:
  - interpret node  : CPU-only  (query analysis, routing logic)
  - retrieve node   : CPU-only  (vector DB search via HNSW on CPU)
  - generate node   : GPU + CPU (LLM inference on GPU, tokenization/decode on CPU)

Metrics per node: wall time, process CPU%, GPU% (continuous sampling)
Generation metrics: TTFT (Time To First Token), TPOT (Time Per Output Token)

Usage:
  # Normal run (uses HF cache)
  python retrieval_router.py --docs ../documents

  # Use a downloaded local model
  python model_loader.py --download --preset tinyllama
  python retrieval_router.py --docs ../documents --llm-model tinyllama

  # Stress profiles (auto-configure parameters for targeted stress)
  python retrieval_router.py --profile cpu        # CPU-heavy: high concurrency, multi-step retrieval
  python retrieval_router.py --profile gpu        # GPU-heavy: long generation, large context, big batches
  python retrieval_router.py --profile balanced   # Stress both CPU and GPU

  # Override profile values with explicit args
  python retrieval_router.py --profile cpu --concurrency 128

  # Manual stress test
  python retrieval_router.py --stress --scale-factor 100 --concurrency 8

  # Stress test without batching (baseline comparison)
  python retrieval_router.py --stress --concurrency 8 --no-batch
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import operator
import os
import platform
import queue
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

import psutil
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model_loader import load_llm, load_embedding

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_LLM_MODEL = "tinyllama"       # preset name or HF model id
DEFAULT_EMBED_MODEL = "minilm"         # preset name or HF model id

# Set at runtime by main() — used by GPUSampler / _gpu_mem_used_mb
_ACTIVE_GPU_IDS: list[int] | None = None
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_TOP_K = 5
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_ITERATIONS = 3
DEFAULT_CONTEXT_DOCS = 5
DEFAULT_RETRIEVAL_STEPS = 1

SAMPLE_QUERIES = [
    "What role does the CPU play in agentic RAG pipelines?",
    "How does HNSW indexing work in vector databases?",
    "Explain the GPU-CPU data transfer overhead during inference.",
    "Compare vector similarity search algorithms for RAG systems.",
    "What are the memory bandwidth bottlenecks in transformer inference?",
    "How does KV-cache optimization reduce GPU compute requirements?",
    "Describe the orchestration overhead in multi-step agent pipelines.",
    "What preprocessing steps are needed before document embedding?",
    "How does batch inference improve GPU throughput?",
    "Explain the trade-offs between CPU and GPU for embedding generation.",
    "What is the impact of context window size on inference latency?",
    "How do concurrent requests affect CPU utilization in RAG pipelines?",
    "Describe the role of attention mechanisms in retrieval-augmented generation.",
]

# ---------------------------------------------------------------------------
# Stress profiles — preset parameter combinations for targeted stress testing
# ---------------------------------------------------------------------------
STRESS_PROFILES = {
    "cpu": {
        "description": "CPU-heavy: high concurrency, multi-step retrieval, "
                       "re-ranking, short generation",
        "why": [
            "High concurrency (64 threads) → CPU thread scheduling + "
            "orchestration overhead across 192 cores",
            "Multi-step retrieval (3 rounds) → repeated HNSW search "
            "(C++ GIL-free, truly multi-core)",
            "TF-IDF re-ranking per round → CPU-bound numpy/scipy "
            "computation (GIL-free)",
            "Large corpus (scale 300x) → bigger HNSW graph, longer "
            "search paths, more CPU cache pressure",
            "top_k=20 → more docs retrieved per round, more CPU-side "
            "filtering and scoring",
            "Short generation (32 tokens) → GPU finishes fast, CPU "
            "orchestration dominates total time",
            "Many queries (100) → sustained CPU load over minutes",
        ],
        "concurrency": 64,
        "stress_queries": 100,
        "scale_factor": 300,
        "max_tokens": 32,
        "top_k": 20,
        "context_docs": 5,
        "retrieval_steps": 3,
        "batch_size": 16,
    },
    "gpu": {
        "description": "GPU-heavy: long generation, large context prefill, "
                       "large batches",
        "why": [
            "Long generation (512 tokens) → sustained GPU decode compute, "
            "TPOT dominates total latency",
            "Large context (20 docs in prompt) → large KV-cache prefill, "
            "heavy attention computation",
            "Large batch size (32) → fills GPU memory, amortizes kernel "
            "launch overhead, high SM utilization",
            "Moderate concurrency (16) → enough threads to keep the "
            "batch pipeline continuously full",
            "Moderate corpus → enough docs for context stuffing without "
            "CPU becoming bottleneck",
        ],
        "concurrency": 16,
        "stress_queries": 50,
        "scale_factor": 100,
        "max_tokens": 512,
        "top_k": 20,
        "context_docs": 20,
        "retrieval_steps": 1,
        "batch_size": 32,
    },
    "balanced": {
        "description": "Balanced: stress both CPU and GPU simultaneously",
        "why": [
            "High concurrency (32) → CPU scheduling pressure + GPU "
            "batch filling",
            "Multi-step retrieval (2 rounds) → CPU retrieval + re-ranking "
            "work per query",
            "Large context (15 docs) → GPU prefill work per generation",
            "Medium generation (256 tokens) → meaningful GPU decode time",
            "Large corpus (200x) → CPU-heavy HNSW search on big graph",
            "Batch size 16 → GPU batching benefit without excessive "
            "memory pressure",
        ],
        "concurrency": 32,
        "stress_queries": 80,
        "scale_factor": 200,
        "max_tokens": 256,
        "top_k": 15,
        "context_docs": 15,
        "retrieval_steps": 2,
        "batch_size": 16,
    },
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class NodeMetrics:
    node: str
    duration_ms: float
    cpu_percent: float           # process-level CPU% (can exceed 100% on multi-core)
    gpu_util_percent: float      # average GPU% sampled continuously
    gpu_mem_used_mb: float       # GPU memory used at end
    device: str = "cpu"
    details: dict = field(default_factory=dict)


@dataclass
class StreamMetrics:
    ttft_ms: float
    tpot_ms: float
    total_tokens: int
    total_generation_ms: float


# ---------------------------------------------------------------------------
# Resource monitor — process-level CPU + threaded GPU sampling
# ---------------------------------------------------------------------------
class GPUSampler:
    """Background thread that samples GPU utilization every interval_ms."""

    def __init__(self, interval_s: float = 0.05, gpu_ids: list[int] | None = None):
        self.interval = interval_s
        self.gpu_ids = gpu_ids  # None = monitor GPU 0 only
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        if not torch.cuda.is_available():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[float, float, float]:
        """Returns (avg, peak, num_samples)."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        if not self.samples:
            return 0.0, 0.0, 0.0
        avg = statistics.mean(self.samples)
        peak = max(self.samples)
        return avg, peak, len(self.samples)

    def _run(self):
        # Build nvidia-smi command — optionally filter by GPU ids
        cmd = ["nvidia-smi", "--query-gpu=index,utilization.gpu",
               "--format=csv,noheader,nounits"]
        if self.gpu_ids:
            cmd += [f"--id={','.join(str(g) for g in self.gpu_ids)}"]
        target_ids = set(self.gpu_ids) if self.gpu_ids else None

        while not self._stop.is_set():
            try:
                out = subprocess.check_output(cmd, timeout=2, text=True)
                utils = []
                for line in out.strip().split("\n"):
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        gid = int(parts[0].strip())
                        util = float(parts[1].strip())
                        if target_ids is None or gid in target_ids:
                            utils.append(util)
                if utils:
                    # Average utilization across selected GPUs
                    self.samples.append(statistics.mean(utils))
            except Exception:
                pass
            self._stop.wait(self.interval)


def _gpu_mem_used_mb(gpu_ids: list[int] | None = None) -> float:
    """Total GPU memory allocated across selected GPUs."""
    if not torch.cuda.is_available():
        return 0.0
    if gpu_ids:
        return sum(torch.cuda.memory_allocated(g) / (1024 ** 2) for g in gpu_ids)
    return torch.cuda.memory_allocated(0) / (1024 ** 2)


class ResourceMonitor:
    """Process-level CPU% + continuous GPU sampling."""

    def __init__(self):
        self._proc = psutil.Process()

    def __enter__(self):
        self._proc.cpu_percent()          # prime (process-level, not system-wide)
        self._gpu_sampler = GPUSampler(interval_s=0.05, gpu_ids=_ACTIVE_GPU_IDS)
        self._gpu_sampler.start()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        # Process-level CPU%: 100% = 1 full core, 400% = 4 cores saturated
        self.cpu_pct = self._proc.cpu_percent()
        gpu_avg, gpu_peak, gpu_n = self._gpu_sampler.stop()
        self.gpu_pct = gpu_avg
        self.gpu_peak_pct = gpu_peak
        self.gpu_samples = gpu_n
        self.gpu_mem_mb = _gpu_mem_used_mb(_ACTIVE_GPU_IDS)


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------
def load_documents(docs_dir: str, file_pattern: str = "*.*") -> list:
    from langchain_community.document_loaders import (
        Docx2txtLoader, PyPDFLoader, TextLoader, UnstructuredMarkdownLoader,
    )
    loader_map = {
        ".txt": TextLoader, ".md": UnstructuredMarkdownLoader,
        ".pdf": PyPDFLoader, ".docx": Docx2txtLoader,
    }
    paths = sorted(glob.glob(os.path.join(docs_dir, "**", file_pattern), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No files matching '{file_pattern}' in {docs_dir}")

    documents = []
    for p in paths:
        ext = Path(p).suffix.lower()
        loader_cls = loader_map.get(ext)
        if loader_cls is None:
            continue
        try:
            docs = loader_cls(p).load()
            for d in docs:
                d.metadata["source_file"] = os.path.basename(p)
            documents.extend(docs)
            print(f"  [load] {os.path.basename(p)}  ({len(docs)} page(s))")
        except Exception as exc:
            print(f"  [error] {p}: {exc}")
    print(f"  Total: {len(documents)} document segment(s)")
    return documents


def scale_documents(documents: list, factor: int) -> list:
    """Duplicate documents to simulate a larger corpus for stress testing."""
    from langchain_core.documents import Document
    scaled = []
    for i in range(factor):
        for doc in documents:
            new_doc = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "copy_index": i},
            )
            scaled.append(new_doc)
    return scaled


# ---------------------------------------------------------------------------
# Vector store (pluggable)
# ---------------------------------------------------------------------------
def build_vectorstore(documents, embeddings, backend="chroma",
                      chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"  {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    if backend == "chroma":
        from langchain_community.vectorstores import Chroma
        return Chroma.from_documents(chunks, embeddings)
    elif backend == "faiss":
        from langchain_community.vectorstores import FAISS
        return FAISS.from_documents(chunks, embeddings)
    raise ValueError(f"Unsupported backend: {backend}")


# ---------------------------------------------------------------------------
# TF-IDF re-ranker (CPU-intensive, GIL-free via scipy/numpy)
# ---------------------------------------------------------------------------
class TFIDFReranker:
    """Re-rank retrieved documents by TF-IDF cosine similarity to the query.

    CPU-bound: scikit-learn's TfidfVectorizer uses scipy sparse matrices and
    numpy operations that release the GIL, enabling true multi-core parallelism
    when called from concurrent threads.
    """

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        self._TfidfVectorizer = TfidfVectorizer
        self._cosine_similarity = cosine_similarity

    def rerank(self, query: str, documents: list[str],
               top_n: int = 5) -> list[str]:
        """Return top_n documents re-ranked by TF-IDF similarity to query."""
        if not documents:
            return documents
        vectorizer = self._TfidfVectorizer(stop_words="english")
        corpus = [query] + documents
        tfidf_matrix = vectorizer.fit_transform(corpus)
        sims = self._cosine_similarity(
            tfidf_matrix[0:1], tfidf_matrix[1:],
        ).flatten()
        ranked_idx = sims.argsort()[::-1][:top_n]
        return [documents[i] for i in ranked_idx]


# ---------------------------------------------------------------------------
# Local LLM on GPU (loaded via model_loader)
# ---------------------------------------------------------------------------
class LocalLLM:
    """Wraps a HuggingFace causal-LM for token-by-token generation on GPU."""

    def __init__(self, model_name: str, device: str, max_new_tokens: int,
                 download: bool = False, gpu_ids: list[int] | None = None):
        self.max_new_tokens = max_new_tokens
        self.gpu_ids = gpu_ids
        self.model, self.tokenizer = load_llm(
            model_name, device=device, download=download,
            gpu_ids=gpu_ids,
        )
        # For multi-GPU, resolve device from model's first parameter
        self.device = str(next(self.model.parameters()).device)

    def generate_stream(self, prompt: str):
        """Yield tokens one at a time (for TTFT / TPOT measurement)."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated = inputs["input_ids"]
            past_key_values = None

            for _ in range(self.max_new_tokens):
                outputs = self.model(
                    input_ids=generated[:, -1:] if past_key_values else generated,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)

                token_id = next_token.item()
                if token_id == self.tokenizer.eos_token_id:
                    break
                yield self.tokenizer.decode(token_id, skip_special_tokens=True)

    def create_batch_engine(self, batch_size: int = 8,
                            max_wait_ms: int = 50) -> "BatchInferenceEngine":
        """Create a BatchInferenceEngine for concurrent batched inference."""
        return BatchInferenceEngine(
            self.model, self.tokenizer, self.device,
            self.max_new_tokens, batch_size, max_wait_ms,
        )


# ---------------------------------------------------------------------------
# Batched inference engine — collects prompts, runs batched forward passes
# ---------------------------------------------------------------------------
class BatchInferenceEngine:
    """
    Collects generation requests from multiple threads and runs them
    as a single batched forward pass on the GPU.

    Each thread calls submit(prompt) → Future.  A background inference
    thread collects pending requests and, when batch_size is reached or
    max_wait_ms expires, runs a batched token-by-token generation loop.
    Per-query TTFT and TPOT are measured relative to each request's
    submission time.
    """

    def __init__(self, model, tokenizer, device: str,
                 max_new_tokens: int, batch_size: int = 8,
                 max_wait_ms: int = 50):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.max_wait_s = max_wait_ms / 1000.0

        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        """Start the background inference thread."""
        self._running = True
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal stop and wait for the inference thread to finish."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

    def submit(self, prompt: str) -> concurrent.futures.Future:
        """Submit a prompt. Returns Future with (tokens, ttft_ms, tpot_ms, total_ms)."""
        fut: concurrent.futures.Future = concurrent.futures.Future()
        submit_time = time.perf_counter()
        self._queue.put((prompt, submit_time, fut))
        return fut

    # --- internal ---

    def _inference_loop(self):
        while self._running:
            batch = self._collect_batch()
            if not batch:
                continue
            self._run_batch(batch)

    def _collect_batch(self):
        """Wait for up to batch_size requests or max_wait_s, whichever first."""
        batch = []
        # Block until at least one request
        try:
            item = self._queue.get(timeout=0.1)
            batch.append(item)
        except queue.Empty:
            return []

        # Collect more until batch_size or deadline
        deadline = time.perf_counter() + self.max_wait_s
        while len(batch) < self.batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                item = self._queue.get(timeout=max(remaining, 0.001))
                batch.append(item)
            except queue.Empty:
                break
        return batch

    def _run_batch(self, batch):
        """Batched token-by-token generation with per-request metrics."""
        prompts = [b[0] for b in batch]
        submit_times = [b[1] for b in batch]
        futures = [b[2] for b in batch]
        bs = len(prompts)

        # Left-pad for batched causal LM generation
        orig_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
        ).to(self.device)
        self.tokenizer.padding_side = orig_padding_side

        first_token_times: list[float | None] = [None] * bs
        token_times_per_req: list[list[float]] = [[] for _ in range(bs)]
        tokens_per_req: list[list[str]] = [[] for _ in range(bs)]
        finished = [False] * bs

        with torch.no_grad():
            generated = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            past_key_values = None

            for step in range(self.max_new_tokens):
                if all(finished):
                    break

                if past_key_values is None:
                    outputs = self.model(
                        input_ids=generated,
                        attention_mask=attention_mask,
                        past_key_values=None,
                        use_cache=True,
                    )
                else:
                    outputs = self.model(
                        input_ids=generated[:, -1:],
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                past_key_values = outputs.past_key_values
                next_tokens = outputs.logits[:, -1, :].argmax(dim=-1)
                now = time.perf_counter()

                for i in range(bs):
                    if finished[i]:
                        continue
                    token_id = next_tokens[i].item()
                    if token_id == self.tokenizer.eos_token_id:
                        finished[i] = True
                        continue
                    if first_token_times[i] is None:
                        first_token_times[i] = now
                    token_times_per_req[i].append(now)
                    tokens_per_req[i].append(
                        self.tokenizer.decode(token_id, skip_special_tokens=True)
                    )

                generated = torch.cat(
                    [generated, next_tokens.unsqueeze(-1)], dim=-1,
                )
                attention_mask = torch.cat(
                    [attention_mask,
                     torch.ones((bs, 1), dtype=attention_mask.dtype,
                                device=self.device)],
                    dim=-1,
                )

        # Resolve futures with per-request metrics
        for i in range(bs):
            gen_end = time.perf_counter()
            total_ms = (gen_end - submit_times[i]) * 1000

            if first_token_times[i] is not None:
                ttft_ms = (first_token_times[i] - submit_times[i]) * 1000
            else:
                ttft_ms = total_ms

            times = token_times_per_req[i]
            if len(times) > 1:
                inter = [(times[j] - times[j - 1]) * 1000
                         for j in range(1, len(times))]
                tpot_ms = statistics.mean(inter)
            else:
                tpot_ms = 0.0

            futures[i].set_result((tokens_per_req[i], ttft_ms, tpot_ms, total_ms))


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    query: str
    route: str
    documents: list[str]
    answer: str
    node_metrics: Annotated[list, operator.add]
    stream_metrics: StreamMetrics | None


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------
def make_interpret_node():
    def interpret(state: dict) -> dict:
        mon = ResourceMonitor()
        with mon:
            query = state["query"]
            retrieval_signals = [
                "what", "how", "explain", "describe", "compare",
                "detail", "list", "summarize", "evidence",
            ]
            score = sum(1 for s in retrieval_signals if s in query.lower())
            route = "retrieve" if score >= 1 else "direct"

        return {
            "route": route,
            "node_metrics": [NodeMetrics(
                node="interpret", duration_ms=mon.elapsed_ms,
                cpu_percent=mon.cpu_pct, gpu_util_percent=mon.gpu_pct,
                gpu_mem_used_mb=mon.gpu_mem_mb, device="cpu",
                details={"route": route, "signal_score": score},
            )],
        }
    return interpret


def make_retrieve_node(vectorstore, top_k=DEFAULT_TOP_K):
    def retrieve(state: dict) -> dict:
        mon = ResourceMonitor()
        with mon:
            results = vectorstore.similarity_search(state["query"], k=top_k)
            doc_texts = [r.page_content for r in results]

        return {
            "documents": doc_texts,
            "node_metrics": [NodeMetrics(
                node="retrieve", duration_ms=mon.elapsed_ms,
                cpu_percent=mon.cpu_pct, gpu_util_percent=mon.gpu_pct,
                gpu_mem_used_mb=mon.gpu_mem_mb, device="cpu",
                details={"num_results": len(doc_texts)},
            )],
        }
    return retrieve


def make_enhanced_retrieve_node(vectorstore, reranker: TFIDFReranker,
                                top_k: int = DEFAULT_TOP_K,
                                context_docs: int = DEFAULT_CONTEXT_DOCS,
                                retrieval_steps: int = DEFAULT_RETRIEVAL_STEPS):
    """Multi-step retrieve node: retrieve → TF-IDF re-rank → refine → repeat.

    Each round performs:
      1. HNSW vector search (C++ — releases GIL, truly multi-core)
      2. TF-IDF re-ranking (scipy/numpy — releases GIL, multi-core)
      3. Query refinement by appending top-result keywords

    More retrieval_steps = more CPU work per query.
    """
    def retrieve(state: dict) -> dict:
        mon = ResourceMonitor()
        with mon:
            current_query = state["query"]
            ranked_docs: list[str] = []
            for step in range(retrieval_steps):
                results = vectorstore.similarity_search(
                    current_query, k=top_k,
                )
                texts = [r.page_content for r in results]
                ranked_docs = reranker.rerank(
                    current_query, texts, top_n=context_docs,
                )
                # Refine query for next round using top result keywords
                if step < retrieval_steps - 1 and ranked_docs:
                    current_query = (
                        f"{state['query']} {ranked_docs[0][:200]}"
                    )

        return {
            "documents": ranked_docs,
            "node_metrics": [NodeMetrics(
                node="retrieve", duration_ms=mon.elapsed_ms,
                cpu_percent=mon.cpu_pct, gpu_util_percent=mon.gpu_pct,
                gpu_mem_used_mb=mon.gpu_mem_mb, device="cpu",
                details={
                    "num_results": len(ranked_docs),
                    "retrieval_steps": retrieval_steps,
                    "top_k_per_step": top_k,
                    "reranked_to": context_docs,
                },
            )],
        }
    return retrieve


def make_generate_node(llm: LocalLLM, context_docs: int = DEFAULT_CONTEXT_DOCS):
    def generate(state: dict) -> dict:
        query = state["query"]
        docs = state.get("documents", [])

        if docs:
            context = "\n---\n".join(docs[:context_docs])
            prompt = (
                f"Using the following context, answer the question.\n\n"
                f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            )
        else:
            prompt = f"Answer the following question.\n\nQuestion: {query}\n\nAnswer:"

        token_times: list[float] = []
        tokens: list[str] = []
        gen_start = time.perf_counter()
        first_token_time = None

        mon = ResourceMonitor()
        with mon:
            for tok in llm.generate_stream(prompt):
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                token_times.append(now)
                tokens.append(tok)

        total_gen_ms = (time.perf_counter() - gen_start) * 1000
        ttft_ms = ((first_token_time - gen_start) * 1000
                   if first_token_time else total_gen_ms)

        if len(token_times) > 1:
            inter = [(token_times[i] - token_times[i-1]) * 1000
                     for i in range(1, len(token_times))]
            tpot_ms = statistics.mean(inter)
        else:
            tpot_ms = 0.0

        return {
            "answer": "".join(tokens),
            "stream_metrics": StreamMetrics(
                ttft_ms=ttft_ms, tpot_ms=tpot_ms,
                total_tokens=len(tokens), total_generation_ms=total_gen_ms,
            ),
            "node_metrics": [NodeMetrics(
                node="generate", duration_ms=mon.elapsed_ms,
                cpu_percent=mon.cpu_pct, gpu_util_percent=mon.gpu_pct,
                gpu_mem_used_mb=mon.gpu_mem_mb, device="gpu",
                details={"tokens": len(tokens),
                         "ttft_ms": round(ttft_ms, 2),
                         "tpot_ms": round(tpot_ms, 2),
                         "gpu_peak_pct": round(mon.gpu_peak_pct, 1),
                         "gpu_samples": mon.gpu_samples},
            )],
        }
    return generate


# ---------------------------------------------------------------------------
# Batched generate node — submits to BatchInferenceEngine
# ---------------------------------------------------------------------------
def make_batched_generate_node(batch_engine: BatchInferenceEngine,
                               context_docs: int = DEFAULT_CONTEXT_DOCS):
    """Generate node that submits to a shared batch engine for GPU inference."""
    def generate(state: dict) -> dict:
        query = state["query"]
        docs = state.get("documents", [])

        if docs:
            context = "\n---\n".join(docs[:context_docs])
            prompt = (
                f"Using the following context, answer the question.\n\n"
                f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            )
        else:
            prompt = f"Answer the following question.\n\nQuestion: {query}\n\nAnswer:"

        mon = ResourceMonitor()
        with mon:
            future = batch_engine.submit(prompt)
            tokens, ttft_ms, tpot_ms, total_gen_ms = future.result()

        return {
            "answer": "".join(tokens),
            "stream_metrics": StreamMetrics(
                ttft_ms=ttft_ms, tpot_ms=tpot_ms,
                total_tokens=len(tokens), total_generation_ms=total_gen_ms,
            ),
            "node_metrics": [NodeMetrics(
                node="generate", duration_ms=mon.elapsed_ms,
                cpu_percent=mon.cpu_pct, gpu_util_percent=mon.gpu_pct,
                gpu_mem_used_mb=mon.gpu_mem_mb, device="gpu",
                details={"tokens": len(tokens),
                         "ttft_ms": round(ttft_ms, 2),
                         "tpot_ms": round(tpot_ms, 2),
                         "gpu_peak_pct": round(mon.gpu_peak_pct, 1),
                         "gpu_samples": mon.gpu_samples,
                         "batched": True},
            )],
        }
    return generate


# ---------------------------------------------------------------------------
# Router + graph builder
# ---------------------------------------------------------------------------
def route_decision(state: dict) -> Literal["retrieve", "generate"]:
    return "retrieve" if state.get("route") == "retrieve" else "generate"


def build_graph(vectorstore, llm: LocalLLM, top_k=DEFAULT_TOP_K,
                context_docs=DEFAULT_CONTEXT_DOCS):
    g = StateGraph(GraphState)
    g.add_node("interpret", make_interpret_node())
    g.add_node("retrieve", make_retrieve_node(vectorstore, top_k))
    g.add_node("generate", make_generate_node(llm, context_docs=context_docs))
    g.set_entry_point("interpret")
    g.add_conditional_edges("interpret", route_decision,
                            {"retrieve": "retrieve", "generate": "generate"})
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


def build_batched_graph(vectorstore, batch_engine: BatchInferenceEngine,
                        top_k=DEFAULT_TOP_K,
                        context_docs=DEFAULT_CONTEXT_DOCS):
    """Build graph using batched generate node for higher throughput."""
    g = StateGraph(GraphState)
    g.add_node("interpret", make_interpret_node())
    g.add_node("retrieve", make_retrieve_node(vectorstore, top_k))
    g.add_node("generate", make_batched_generate_node(
        batch_engine, context_docs=context_docs))
    g.set_entry_point("interpret")
    g.add_conditional_edges("interpret", route_decision,
                            {"retrieve": "retrieve", "generate": "generate"})
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


def build_profiled_graph(vectorstore, llm: LocalLLM, reranker: TFIDFReranker,
                         top_k=DEFAULT_TOP_K, context_docs=DEFAULT_CONTEXT_DOCS,
                         retrieval_steps=DEFAULT_RETRIEVAL_STEPS):
    """Build graph with enhanced retrieve node (multi-step + re-ranking)."""
    g = StateGraph(GraphState)
    g.add_node("interpret", make_interpret_node())
    g.add_node("retrieve", make_enhanced_retrieve_node(
        vectorstore, reranker, top_k=top_k,
        context_docs=context_docs, retrieval_steps=retrieval_steps))
    g.add_node("generate", make_generate_node(llm, context_docs=context_docs))
    g.set_entry_point("interpret")
    g.add_conditional_edges("interpret", route_decision,
                            {"retrieve": "retrieve", "generate": "generate"})
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


def build_batched_profiled_graph(vectorstore, batch_engine: BatchInferenceEngine,
                                 reranker: TFIDFReranker,
                                 top_k=DEFAULT_TOP_K,
                                 context_docs=DEFAULT_CONTEXT_DOCS,
                                 retrieval_steps=DEFAULT_RETRIEVAL_STEPS):
    """Profiled graph with batched generate + enhanced retrieve."""
    g = StateGraph(GraphState)
    g.add_node("interpret", make_interpret_node())
    g.add_node("retrieve", make_enhanced_retrieve_node(
        vectorstore, reranker, top_k=top_k,
        context_docs=context_docs, retrieval_steps=retrieval_steps))
    g.add_node("generate", make_batched_generate_node(
        batch_engine, context_docs=context_docs))
    g.set_entry_point("interpret")
    g.add_conditional_edges("interpret", route_decision,
                            {"retrieve": "retrieve", "generate": "generate"})
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------
def collect_system_info() -> dict:
    freq = psutil.cpu_freq()
    return {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_freq_max_mhz": freq.max if freq else None,
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": (round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
                          if torch.cuda.is_available() else None),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(system_info: dict, all_results: list[dict],
                 stress_results: dict | None = None,
                 profile_name: str | None = None):
    W = 80
    print("\n" + "=" * W)
    print("  RETRIEVAL ROUTER BENCHMARK — LangGraph (CPU + GPU)")
    print("=" * W)

    print("\n--- System ---")
    for k, v in system_info.items():
        print(f"  {k:>24s}: {v}")

    if profile_name and profile_name in STRESS_PROFILES:
        prof = STRESS_PROFILES[profile_name]
        print(f"\n--- Stress Profile: {profile_name.upper()} ---")
        print(f"  {prof['description']}")
        print(f"\n  Parameter tuning and stress rationale:")
        for reason in prof["why"]:
            print(f"    - {reason}")

    total_cpu_ms = 0.0
    total_gpu_ms = 0.0

    for qr in all_results:
        metrics: list[NodeMetrics] = qr["node_metrics"]
        sm: StreamMetrics | None = qr.get("stream_metrics")

        print(f"\n--- Q: \"{qr['query'][:68]}\" ---")
        print(f"  {'Node':<12s} {'Dev':<5s} {'Time(ms)':>10s} "
              f"{'CPU%':>8s} {'GPU%':>8s} {'GPUpk%':>8s} {'GPUmem':>9s}")
        print("  " + "-" * 64)

        q_total = 0.0
        for m in metrics:
            print(f"  {m.node:<12s} {m.device:<5s} {m.duration_ms:>10.2f} "
                  f"{m.cpu_percent:>7.1f}% {m.gpu_util_percent:>7.1f}% "
                  f"{m.details.get('gpu_peak_pct', 0):>7.1f}% "
                  f"{m.gpu_mem_used_mb:>8.0f}MB")
            q_total += m.duration_ms
            if m.device == "cpu":
                total_cpu_ms += m.duration_ms
            else:
                total_gpu_ms += m.duration_ms

        print("  " + "-" * 64)
        print(f"  {'Total':<12s} {'':5s} {q_total:>10.2f}")

        if sm:
            print(f"  TTFT = {sm.ttft_ms:.2f} ms | TPOT = {sm.tpot_ms:.2f} ms | "
                  f"Tokens = {sm.total_tokens}")

    # Summary
    grand = total_cpu_ms + total_gpu_ms
    if grand > 0:
        print(f"\n--- Summary (single-query mode) ---")
        print(f"  CPU-device total : {total_cpu_ms:>10.2f} ms  ({total_cpu_ms/grand*100:.1f}%)")
        print(f"  GPU-device total : {total_gpu_ms:>10.2f} ms  ({total_gpu_ms/grand*100:.1f}%)")
        print(f"  Grand total      : {grand:>10.2f} ms")

    # Stress test results
    if stress_results:
        sr = stress_results
        mode = sr.get("mode", "non-batched")
        print(f"\n--- Stress Test Results ({mode}) ---")
        if mode == "batched":
            print(f"  Batch size       : {sr.get('batch_size', 'n/a')}")
        print(f"  Corpus scale     : {sr['scale_factor']}x "
              f"({sr['num_chunks']} chunks)")
        print(f"  Concurrency      : {sr['concurrency']} parallel queries")
        print(f"  Total queries    : {sr['total_queries']}")
        if sr.get("retrieval_steps", 1) > 1:
            print(f"  Retrieval steps  : {sr['retrieval_steps']} (with TF-IDF re-ranking)")
        if sr.get("context_docs"):
            print(f"  Context docs     : {sr['context_docs']} docs in LLM prompt")
        if sr.get("max_tokens"):
            print(f"  Max tokens       : {sr['max_tokens']}")
        print(f"  Throughput       : {sr['throughput_qps']:.2f} queries/sec")
        if sr.get("tokens_per_sec"):
            print(f"  Tokens/sec       : {sr['tokens_per_sec']:.1f}")
        print(f"  Avg latency      : {sr['avg_latency_ms']:.1f} ms")
        print(f"  P50 latency      : {sr['p50_latency_ms']:.1f} ms")
        print(f"  P95 latency      : {sr['p95_latency_ms']:.1f} ms")
        print(f"  P99 latency      : {sr['p99_latency_ms']:.1f} ms")
        if sr.get("avg_ttft_ms"):
            print(f"  Avg TTFT         : {sr['avg_ttft_ms']:.1f} ms")
            print(f"  Avg TPOT         : {sr['avg_tpot_ms']:.1f} ms")
        print(f"  Avg CPU% (proc)  : {sr['avg_cpu_pct']:.1f}%  "
              f"(= ~{sr['avg_cpu_pct']/100:.1f} cores saturated)")
        print(f"  Peak CPU% (proc) : {sr['peak_cpu_pct']:.1f}%  "
              f"(= ~{sr['peak_cpu_pct']/100:.1f} cores saturated)")
        print(f"  Avg GPU%         : {sr['avg_gpu_pct']:.1f}%")
        print(f"  Peak GPU%        : {sr['peak_gpu_pct']:.1f}%")
        print(f"  GPU mem used     : {sr['gpu_mem_mb']:.0f} MB")

        # Bottleneck analysis — thresholds scale with core count
        # psutil cpu_percent: 100% = 1 logical core, so max = num_cores * 100
        # Industry standard: sustained >=80% = HIGH, 50-80% = MODERATE, <50% = LOW
        num_cores = os.cpu_count() or 1
        max_cpu_pct = num_cores * 100  # e.g. 19200% for 192 cores
        cpu_util = sr['peak_cpu_pct'] / max_cpu_pct * 100  # 0–100 scale
        gpu_util = sr['peak_gpu_pct']                       # already 0–100
        cores_used = sr['peak_cpu_pct'] / 100

        print(f"\n--- Bottleneck Analysis ---")
        print(f"  System: {num_cores} logical cores "
              f"(max cpu_percent = {max_cpu_pct}%)")
        print(f"  Peak CPU: {sr['peak_cpu_pct']:.0f}% "
              f"= {cores_used:.1f}/{num_cores} cores "
              f"= {cpu_util:.1f}% of total CPU capacity")
        print(f"  Peak GPU: {gpu_util:.0f}%")

        cpu_high = cpu_util >= 80    # >=80% of total CPU = high
        cpu_mod  = cpu_util >= 50    # >=50% = moderate
        gpu_high = gpu_util >= 80    # >=80% GPU = high
        gpu_mod  = gpu_util >= 50    # >=50% = moderate

        if gpu_high and cpu_high:
            print("  >> BOTH SATURATED: CPU >=80% and GPU >=80%.")
            print("     System is well-balanced but at capacity. Both faster")
            print("     CPU and GPU would help for further scaling.")
        elif gpu_high and not cpu_mod:
            print("  >> GPU-BOUND: GPU >=80%, CPU <50%.")
            print("     Bottleneck is LLM inference. Larger/faster GPU or")
            print("     smaller model would help. CPU has significant headroom.")
        elif cpu_high and not gpu_mod:
            print("  >> CPU-BOUND: CPU >=80%, GPU <50%.")
            print("     Bottleneck is orchestration/retrieval/pre-processing.")
            print("     Faster CPU or optimized vector DB would help.")
        elif gpu_high and cpu_mod:
            print("  >> GPU-BOUND (approaching balanced): GPU >=80%, CPU 50-80%.")
            print("     GPU is the primary bottleneck but CPU load is moderate.")
            print("     Consider faster GPU first, then CPU if scaling further.")
        elif cpu_high and gpu_mod:
            print("  >> CPU-BOUND (approaching balanced): CPU >=80%, GPU 50-80%.")
            print("     CPU is the primary bottleneck but GPU is moderately busy.")
            print("     Consider faster CPU first, then GPU if scaling further.")
        elif cpu_mod and gpu_mod:
            print("  >> MODERATE LOAD: Both CPU 50-80% and GPU 50-80%.")
            print("     System is under moderate load. Increase --concurrency or")
            print("     --scale-factor to find which saturates first.")
        else:
            print("  >> LOW UTILIZATION: CPU <50% and GPU <50%.")
            print("     Workload is not stressing the system. Increase")
            print("     --concurrency or --scale-factor to reveal the bottleneck.")
            print(f"     Suggestion: try --concurrency {sr['concurrency'] * 2} "
                  f"--scale-factor {sr['scale_factor'] * 2}")

    print("\n" + "=" * W)


# ---------------------------------------------------------------------------
# Stress test — concurrent queries on scaled corpus
# ---------------------------------------------------------------------------
def run_stress_test(app, queries: list[str], concurrency: int,
                    total_queries: int) -> dict:
    """Fire queries concurrently, measure throughput + per-query latency."""
    proc = psutil.Process()
    latencies: list[float] = []
    cpu_samples: list[float] = []
    gpu_sampler = GPUSampler(interval_s=0.1, gpu_ids=_ACTIVE_GPU_IDS)
    gpu_sampler.start()

    # Expand query list to match total_queries
    expanded = [queries[i % len(queries)] for i in range(total_queries)]

    def run_single(query: str) -> float:
        start = time.perf_counter()
        app.invoke({
            "query": query, "documents": [],
            "node_metrics": [], "stream_metrics": None,
        })
        elapsed = (time.perf_counter() - start) * 1000
        cpu_samples.append(proc.cpu_percent())
        return elapsed

    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(run_single, q) for q in expanded]
        for f in as_completed(futures):
            latencies.append(f.result())

    wall_ms = (time.perf_counter() - wall_start) * 1000
    gpu_avg, gpu_peak, _ = gpu_sampler.stop()

    latencies.sort()
    n = len(latencies)
    return {
        "concurrency": concurrency,
        "total_queries": total_queries,
        "wall_time_ms": round(wall_ms, 1),
        "throughput_qps": round(total_queries / (wall_ms / 1000), 2),
        "avg_latency_ms": round(statistics.mean(latencies), 1),
        "p50_latency_ms": round(latencies[int(n * 0.50)], 1),
        "p95_latency_ms": round(latencies[int(n * 0.95)], 1),
        "p99_latency_ms": round(latencies[min(int(n * 0.99), n - 1)], 1),
        "avg_cpu_pct": round(statistics.mean(cpu_samples) if cpu_samples else 0, 1),
        "peak_cpu_pct": round(max(cpu_samples) if cpu_samples else 0, 1),
        "avg_gpu_pct": round(gpu_avg, 1),
        "peak_gpu_pct": round(gpu_peak, 1),
        "gpu_mem_mb": round(_gpu_mem_used_mb(_ACTIVE_GPU_IDS), 0),
    }


def run_batched_stress_test(batch_engine: BatchInferenceEngine, app,
                            queries: list[str], concurrency: int,
                            total_queries: int) -> dict:
    """Fire queries concurrently with batched GPU inference."""
    proc = psutil.Process()
    latencies: list[float] = []
    cpu_samples: list[float] = []
    per_query_metrics: list[dict] = []
    gpu_sampler = GPUSampler(interval_s=0.1, gpu_ids=_ACTIVE_GPU_IDS)
    gpu_sampler.start()

    expanded = [queries[i % len(queries)] for i in range(total_queries)]

    def run_single(query: str) -> tuple[float, dict]:
        start = time.perf_counter()
        result = app.invoke({
            "query": query, "documents": [],
            "node_metrics": [], "stream_metrics": None,
        })
        elapsed = (time.perf_counter() - start) * 1000
        cpu_samples.append(proc.cpu_percent())
        sm = result.get("stream_metrics")
        metrics = {
            "latency_ms": elapsed,
            "ttft_ms": sm.ttft_ms if sm else 0,
            "tpot_ms": sm.tpot_ms if sm else 0,
            "tokens": sm.total_tokens if sm else 0,
        }
        return elapsed, metrics

    batch_engine.start()
    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(run_single, q) for q in expanded]
        for f in as_completed(futures):
            elapsed, metrics = f.result()
            latencies.append(elapsed)
            per_query_metrics.append(metrics)

    wall_ms = (time.perf_counter() - wall_start) * 1000
    batch_engine.stop()
    gpu_avg, gpu_peak, _ = gpu_sampler.stop()

    latencies.sort()
    n = len(latencies)

    ttft_values = [m["ttft_ms"] for m in per_query_metrics if m["ttft_ms"] > 0]
    tpot_values = [m["tpot_ms"] for m in per_query_metrics if m["tpot_ms"] > 0]
    total_tokens = sum(m["tokens"] for m in per_query_metrics)

    return {
        "mode": "batched",
        "batch_size": batch_engine.batch_size,
        "concurrency": concurrency,
        "total_queries": total_queries,
        "wall_time_ms": round(wall_ms, 1),
        "throughput_qps": round(total_queries / (wall_ms / 1000), 2),
        "tokens_per_sec": round(total_tokens / (wall_ms / 1000), 2),
        "avg_latency_ms": round(statistics.mean(latencies), 1),
        "p50_latency_ms": round(latencies[int(n * 0.50)], 1),
        "p95_latency_ms": round(latencies[int(n * 0.95)], 1),
        "p99_latency_ms": round(latencies[min(int(n * 0.99), n - 1)], 1),
        "avg_ttft_ms": round(statistics.mean(ttft_values), 2) if ttft_values else 0,
        "avg_tpot_ms": round(statistics.mean(tpot_values), 2) if tpot_values else 0,
        "avg_cpu_pct": round(statistics.mean(cpu_samples) if cpu_samples else 0, 1),
        "peak_cpu_pct": round(max(cpu_samples) if cpu_samples else 0, 1),
        "avg_gpu_pct": round(gpu_avg, 1),
        "peak_gpu_pct": round(gpu_peak, 1),
        "gpu_mem_mb": round(_gpu_mem_used_mb(_ACTIVE_GPU_IDS), 0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LangGraph Retrieval Router — CPU+GPU Benchmark")
    _default_docs = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "documents"))
    parser.add_argument("--docs", default=_default_docs,
                        help="Path to documents directory (default: langGraph/documents/)")
    parser.add_argument("--file-pattern", default="*.*", help="File glob (e.g. '*.txt')")
    parser.add_argument("--vector-db", default="chroma", choices=["chroma", "faiss"])
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--queries", nargs="+", default=None)
    parser.add_argument("--output", default=None)

    # Model / GPU options
    parser.add_argument("--gpu", default=None,
                        help="GPU index or comma-separated indices "
                             "(e.g. '0', '2', '0,1,2,3' for multi-GPU)")
    parser.add_argument("--download", action="store_true",
                        help="Download models to part3/models/ for offline repeated runs")

    # Stress test options
    parser.add_argument("--stress", action="store_true",
                        help="Enable stress test mode")
    parser.add_argument("--profile", default=None,
                        choices=list(STRESS_PROFILES.keys()),
                        help="Stress profile: cpu, gpu, or balanced "
                             "(overrides defaults, explicit args take priority)")
    parser.add_argument("--scale-factor", type=int, default=None,
                        help="Multiply corpus size for stress test (default: 50)")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Concurrent queries in stress test (default: 4)")
    parser.add_argument("--stress-queries", type=int, default=None,
                        help="Total queries in stress test (default: 20)")
    parser.add_argument("--retrieval-steps", type=int, default=None,
                        help="Retrieval rounds per query with re-ranking (default: 1)")
    parser.add_argument("--context-docs", type=int, default=None,
                        help="Docs to include in LLM prompt context (default: 5)")

    # Batching options
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for batched GPU inference (default: 8)")
    parser.add_argument("--max-wait-ms", type=int, default=None,
                        help="Max wait time in ms to collect a batch (default: 50)")
    parser.add_argument("--no-batch", action="store_true",
                        help="Disable batching in stress test (baseline comparison)")
    args = parser.parse_args()

    # --- Profile resolution: apply profile defaults, CLI args override ---
    active_profile = None
    if args.profile:
        active_profile = STRESS_PROFILES[args.profile]
        args.stress = True  # profile implies stress mode

    def _resolve(attr, profile_key, fallback):
        """Use explicit CLI value > profile value > fallback default."""
        val = getattr(args, attr)
        if val is not None:
            return val
        if active_profile and profile_key in active_profile:
            return active_profile[profile_key]
        return fallback

    args.scale_factor = _resolve("scale_factor", "scale_factor", 50)
    args.concurrency = _resolve("concurrency", "concurrency", 4)
    args.stress_queries = _resolve("stress_queries", "stress_queries", 20)
    args.batch_size = _resolve("batch_size", "batch_size", 8)
    args.max_wait_ms = _resolve("max_wait_ms", "max_wait_ms", 50)
    args.retrieval_steps = _resolve("retrieval_steps", "retrieval_steps",
                                    DEFAULT_RETRIEVAL_STEPS)
    args.context_docs = _resolve("context_docs", "context_docs",
                                 DEFAULT_CONTEXT_DOCS)
    args.max_tokens = _resolve("max_tokens", "max_tokens",
                               DEFAULT_MAX_NEW_TOKENS)
    args.top_k = _resolve("top_k", "top_k", DEFAULT_TOP_K)

    # Parse --gpu into list of ints
    global _ACTIVE_GPU_IDS
    gpu_ids = None
    if args.gpu is not None:
        gpu_ids = [int(x.strip()) for x in args.gpu.split(",")]
    _ACTIVE_GPU_IDS = gpu_ids

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_label = f"GPU {gpu_ids}" if gpu_ids else device

    print("=" * 60)
    print("  LangGraph Retrieval Router — CPU + GPU Benchmark")
    print("=" * 60)
    print(f"  Device      : {gpu_label}")
    print(f"  LLM model   : {args.llm_model}")
    print(f"  Vector DB   : {args.vector_db}")
    if args.profile:
        print(f"  PROFILE     : {args.profile} — {active_profile['description']}")
    if args.stress:
        batch_label = ("OFF (baseline)" if args.no_batch
                       else f"ON (bs={args.batch_size}, wait={args.max_wait_ms}ms)")
        print(f"  STRESS MODE : scale={args.scale_factor}x, "
              f"concurrency={args.concurrency}, queries={args.stress_queries}")
        print(f"  BATCHING    : {batch_label}")
        print(f"  RETRIEVAL   : {args.retrieval_steps} step(s), "
              f"top_k={args.top_k}, context_docs={args.context_docs}")
        print(f"  GENERATION  : max_tokens={args.max_tokens}")

    if args.profile:
        print(f"\n  Why this profile stresses "
              f"{'CPU' if args.profile == 'cpu' else 'GPU' if args.profile == 'gpu' else 'both'}:")
        for reason in active_profile["why"]:
            print(f"    - {reason}")

    system_info = collect_system_info()

    # 1. Load documents
    print(f"\n[1/5] Loading documents from: {args.docs}")
    documents = load_documents(args.docs, args.file_pattern)

    # Scale corpus for stress test
    if args.stress:
        print(f"\n[1b]  Scaling corpus {args.scale_factor}x...")
        documents = scale_documents(documents, args.scale_factor)
        print(f"  Scaled to {len(documents)} document segment(s)")

    # 2. Build vector store
    print(f"\n[2/5] Building vector store ({args.vector_db})...")
    mon = ResourceMonitor()
    with mon:
        embeddings = load_embedding(
            args.embed_model, device=device, download=args.download,
            gpu_ids=gpu_ids,
        )
    print(f"  Embedding model loaded in {mon.elapsed_ms:.0f} ms")

    mon2 = ResourceMonitor()
    with mon2:
        vectorstore = build_vectorstore(
            documents, embeddings, backend=args.vector_db,
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
        )
    print(f"  Vector store built in {mon2.elapsed_ms:.0f} ms | "
          f"CPU: {mon2.cpu_pct:.1f}% | GPU avg: {mon2.gpu_pct:.1f}%")

    # 3. Load LLM
    print(f"\n[3/5] Loading LLM on {gpu_label}...")
    llm = LocalLLM(args.llm_model, device, args.max_tokens,
                    download=args.download, gpu_ids=gpu_ids)

    # Build re-ranker if multi-step retrieval is enabled
    reranker = None
    if args.retrieval_steps > 1:
        print("  Initializing TF-IDF re-ranker for multi-step retrieval...")
        reranker = TFIDFReranker()
        app = build_profiled_graph(
            vectorstore, llm, reranker,
            top_k=args.top_k, context_docs=args.context_docs,
            retrieval_steps=args.retrieval_steps,
        )
        print(f"  Graph: interpret(CPU) → [retrieve×{args.retrieval_steps}"
              f"+rerank(CPU)] → generate(GPU)")
    else:
        app = build_graph(vectorstore, llm, top_k=args.top_k,
                          context_docs=args.context_docs)
        print("  Graph: interpret(CPU) → [retrieve(CPU)] → generate(GPU)")

    # 4. Single-query profiling
    queries = args.queries or SAMPLE_QUERIES
    print(f"\n[4/5] Single-query profiling ({len(queries)} queries, "
          f"{args.iterations} iters each)...\n")

    all_results: list[dict] = []

    for qi, query in enumerate(queries, 1):
        label = f"\"{query[:60]}...\"" if len(query) > 60 else f"\"{query}\""
        print(f"  Q{qi}: {label}")
        iter_data = []

        for it in range(args.iterations):
            mon_e2e = ResourceMonitor()
            with mon_e2e:
                result = app.invoke({
                    "query": query, "documents": [],
                    "node_metrics": [], "stream_metrics": None,
                })
            iter_data.append({
                "e2e_ms": mon_e2e.elapsed_ms,
                "node_metrics": result.get("node_metrics", []),
                "stream_metrics": result.get("stream_metrics"),
            })

        last = iter_data[-1]
        e2e_times = [d["e2e_ms"] for d in iter_data]
        sm = last["stream_metrics"]
        route = last["node_metrics"][0].details.get("route", "?") if last["node_metrics"] else "?"

        print(f"      Route: {route} | "
              f"E2E median: {statistics.median(e2e_times):.1f} ms "
              f"(min={min(e2e_times):.1f}, max={max(e2e_times):.1f})")
        if sm:
            print(f"      TTFT={sm.ttft_ms:.1f} ms | TPOT={sm.tpot_ms:.2f} ms | "
                  f"Tokens={sm.total_tokens}\n")

        all_results.append({
            "query": query,
            "e2e_median_ms": round(statistics.median(e2e_times), 2),
            "node_metrics": last["node_metrics"],
            "stream_metrics": sm,
        })

    # 5. Stress test
    stress_results = None
    if args.stress:
        if args.no_batch:
            print(f"\n[5/5] Stress test (NO BATCHING — baseline): "
                  f"{args.stress_queries} queries, "
                  f"{args.concurrency} concurrent...\n")
            stress_results = run_stress_test(
                app, queries, args.concurrency, args.stress_queries,
            )
        else:
            print(f"\n[5/5] Stress test (BATCHED, bs={args.batch_size}): "
                  f"{args.stress_queries} queries, "
                  f"{args.concurrency} concurrent...\n")
            batch_engine = llm.create_batch_engine(
                batch_size=args.batch_size,
                max_wait_ms=args.max_wait_ms,
            )
            if reranker:
                batched_app = build_batched_profiled_graph(
                    vectorstore, batch_engine, reranker,
                    top_k=args.top_k, context_docs=args.context_docs,
                    retrieval_steps=args.retrieval_steps,
                )
            else:
                batched_app = build_batched_graph(
                    vectorstore, batch_engine, top_k=args.top_k,
                    context_docs=args.context_docs,
                )
            stress_results = run_batched_stress_test(
                batch_engine, batched_app, queries,
                args.concurrency, args.stress_queries,
            )

        stress_results["scale_factor"] = args.scale_factor
        stress_results["num_chunks"] = len(vectorstore.get()["ids"]) if hasattr(vectorstore, 'get') else "n/a"
        stress_results["retrieval_steps"] = args.retrieval_steps
        stress_results["context_docs"] = args.context_docs
        stress_results["max_tokens"] = args.max_tokens
        stress_results["profile"] = args.profile
        print(f"  Throughput: {stress_results['throughput_qps']:.2f} q/s | "
              f"P50: {stress_results['p50_latency_ms']:.0f} ms | "
              f"P95: {stress_results['p95_latency_ms']:.0f} ms")
        print(f"  CPU peak: {stress_results['peak_cpu_pct']:.0f}% | "
              f"GPU peak: {stress_results['peak_gpu_pct']:.0f}%")
        if stress_results.get("tokens_per_sec"):
            print(f"  Tokens/sec: {stress_results['tokens_per_sec']:.1f} | "
                  f"Avg TTFT: {stress_results.get('avg_ttft_ms', 0):.1f} ms | "
                  f"Avg TPOT: {stress_results.get('avg_tpot_ms', 0):.1f} ms")

    # Report + save
    print_report(system_info, all_results, stress_results,
                 profile_name=args.profile)

    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "retrieval_router_results.json",
    )
    with open(output_path, "w") as f:
        json.dump({
            "system_info": system_info,
            "config": {
                "llm_model": args.llm_model, "vector_db": args.vector_db,
                "embed_model": args.embed_model, "device": device,
                "gpu_ids": gpu_ids,
                "profile": args.profile,
                "max_tokens": args.max_tokens, "top_k": args.top_k,
                "context_docs": args.context_docs,
                "retrieval_steps": args.retrieval_steps,
            },
            "single_query": [{
                "query": r["query"],
                "e2e_median_ms": r["e2e_median_ms"],
                "nodes": [asdict(m) for m in r["node_metrics"]],
                "stream": asdict(r["stream_metrics"]) if r["stream_metrics"] else None,
            } for r in all_results],
            "stress_test": stress_results,
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
