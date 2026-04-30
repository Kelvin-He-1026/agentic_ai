"""
HotpotQA Vector DB Creator
==========================
Loads the HotpotQA dataset and builds a Chroma vector database of its
Wikipedia paragraphs.  Each paragraph in HotpotQA's `context` field
(title + list of sentences) is stored as a single document with the
title preserved as metadata so retrieval evaluation can compare
retrieved titles against the gold `supporting_facts` titles.

Two variants of HotpotQA are supported:
  - distractor : 10 paragraphs per question (2 gold + 8 distractors).
                 Small, self-contained — recommended for benchmarking.
  - fullwiki   : full Wikipedia retrieval pool (much larger).

Embeddings: BAAI/bge-large-en-v1.5 (re-uses local model cache when
available, falls back to HuggingFace download).

Usage as CLI:
    # Default — first 200 dev questions of the distractor variant
    python db_creator.py --build

    # Build over the entire dev split of the distractor variant
    python db_creator.py --build --num-samples 0

    # Custom collection / persist directory
    python db_creator.py --build --collection hotpotqa --persist-dir ./db/hotpotqa

Usage as a library:
    from db_creator import HotpotQAVectorDB
    builder = HotpotQAVectorDB()
    samples = builder.build(num_samples=200)   # returns the QA samples
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Iterable

import torch

# Make langGraph importable for the model_loader paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "langGraph"))
from model_loader import MODELS_DIR, load_embedding  # noqa: E402

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
# All model weights (LLMs, embeddings, VLMs) are resolved from this directory
# first; if a model is not present locally, the loader falls back to the
# HuggingFace hub.
DEFAULT_MODELS_DIR = PROJECT_DIR / "models"
DB_DIR = PROJECT_DIR / "db" / "hotpotqa_chroma"
DATASET_DIR = Path(__file__).resolve().parent / "dataset"
SAMPLES_JSON = DATASET_DIR / "hotpotqa_samples.json"

DEFAULT_COLLECTION = "hotpotqa"
DEFAULT_VARIANT = "distractor"           # "distractor" | "fullwiki"
DEFAULT_SPLIT = "validation"             # HotpotQA dev split
DEFAULT_NUM_SAMPLES = 200                # 0 = use the full split
# Use the model_loader preset name — resolves to /agentic_rag/models/BAAI--bge-large-en-v1.5
DEFAULT_EMBEDDING = "bge-large"


# ---------------------------------------------------------------------------
# Chroma collection inspection
# ---------------------------------------------------------------------------
def _inspect_chroma_collection(
    chroma_dir: Path, collection_name: str,
) -> tuple[int | None, int]:
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
            SELECT COUNT(*) FROM embeddings e
            JOIN segments s ON e.segment_id = s.id
            WHERE s.collection = ?
            """,
            (collection_id,),
        )
        return dimension, cur.fetchone()[0]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# HotpotQA loading
# ---------------------------------------------------------------------------
def load_hotpotqa(
    variant: str = DEFAULT_VARIANT,
    split: str = DEFAULT_SPLIT,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> list[dict]:
    """Load HotpotQA examples from the HuggingFace `hotpot_qa` dataset.

    Returns a list of dicts with normalized fields:
      id, question, answer, type, level, context, supporting_facts
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required. Install with: pip install datasets"
        ) from exc

    print(f"Loading HotpotQA: variant={variant}, split={split}")
    ds = load_dataset("hotpot_qa", variant, split=split, trust_remote_code=True)

    if num_samples and num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))

    samples: list[dict] = []
    for ex in ds:
        # `context` is {"title": [..], "sentences": [[..], [..]]}
        ctx = ex["context"]
        context_pairs = list(zip(ctx["title"], ctx["sentences"]))
        # supporting_facts is {"title": [..], "sent_id": [..]}
        sf = ex["supporting_facts"]
        sf_pairs = list(zip(sf["title"], sf["sent_id"]))

        samples.append({
            "id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "type": ex.get("type", ""),
            "level": ex.get("level", ""),
            "context": context_pairs,             # [(title, [sent, sent, ...]), ...]
            "supporting_facts": sf_pairs,         # [(title, sent_id), ...]
        })
    print(f"  Loaded {len(samples)} HotpotQA samples")
    return samples


def _samples_to_documents(samples: Iterable[dict]) -> list[dict]:
    """Flatten samples to one document per (title, paragraph) pair.

    Deduplicates by title — across questions, the same Wikipedia paragraph
    can appear multiple times; we keep the first occurrence.
    """
    seen_titles: set[str] = set()
    docs: list[dict] = []
    for s in samples:
        for title, sentences in s["context"]:
            if title in seen_titles:
                continue
            seen_titles.add(title)
            text = " ".join(sentences).strip()
            if not text:
                continue
            docs.append({
                "title": title,
                "text": text,
                "source": title,                  # used as identifier downstream
            })
    return docs


# ---------------------------------------------------------------------------
# Vector DB Builder
# ---------------------------------------------------------------------------
class HotpotQAVectorDB:
    """Build a Chroma vector database from HotpotQA Wikipedia paragraphs."""

    def __init__(
        self,
        persist_dir: Path | str = DB_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = DEFAULT_EMBEDDING,
        device: str | None = None,
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._embeddings = None
        self._vectorstore = None

    # -- embeddings (lazy) -------------------------------------------------
    def _get_embeddings(self):
        """Load the embedding model via model_loader so it picks up
        local weights from /agentic_rag/models/ when present."""
        if self._embeddings is None:
            print(
                f"Loading embedding from {DEFAULT_MODELS_DIR}: "
                f"{self.embedding_model_name} (device={self.device})"
            )
            self._embeddings = load_embedding(
                self.embedding_model_name, device=self.device,
            )
        return self._embeddings

    # -- build -------------------------------------------------------------
    def build(
        self,
        variant: str = DEFAULT_VARIANT,
        split: str = DEFAULT_SPLIT,
        num_samples: int = DEFAULT_NUM_SAMPLES,
        rebuild: bool = False,
        save_samples_to: Path | str | None = SAMPLES_JSON,
    ) -> list[dict]:
        """Load HotpotQA, extract Wikipedia paragraphs, and persist to Chroma.

        Returns the list of HotpotQA samples used (so the benchmark script
        can re-use them as ground-truth).
        """
        samples = load_hotpotqa(variant=variant, split=split, num_samples=num_samples)
        docs = _samples_to_documents(samples)
        print(f"  Unique paragraphs to index: {len(docs)}")

        if save_samples_to is not None:
            save_path = Path(save_samples_to)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(json.dumps(samples, indent=2))
            print(f"  Saved HotpotQA samples to {save_path}")

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        dim, count = _inspect_chroma_collection(self.persist_dir, self.collection_name)
        if count > 0 and not rebuild:
            print(
                f"  Existing Chroma collection '{self.collection_name}' "
                f"already has {count} vectors (dim={dim}). "
                f"Use --rebuild to overwrite."
            )
            return samples

        if rebuild and self.persist_dir.exists():
            import shutil
            shutil.rmtree(self.persist_dir)
            self.persist_dir.mkdir(parents=True, exist_ok=True)

        from langchain_chroma import Chroma
        from langchain_core.documents import Document

        lc_docs = [
            Document(
                page_content=d["text"],
                metadata={"title": d["title"], "source": d["source"]},
            )
            for d in docs
        ]

        print(f"  Embedding {len(lc_docs)} paragraphs...")
        self._vectorstore = Chroma.from_documents(
            documents=lc_docs,
            embedding=self._get_embeddings(),
            collection_name=self.collection_name,
            persist_directory=str(self.persist_dir),
        )
        dim, count = _inspect_chroma_collection(self.persist_dir, self.collection_name)
        print(
            f"  Vector DB ready — collection={self.collection_name}, "
            f"dim={dim}, count={count}, dir={self.persist_dir}"
        )
        return samples

    # -- load existing -----------------------------------------------------
    def load_existing(self):
        from langchain_chroma import Chroma
        self._vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._get_embeddings(),
            persist_directory=str(self.persist_dir),
        )
        return self._vectorstore

    def search(self, query: str, k: int = 5):
        if self._vectorstore is None:
            self.load_existing()
        return self._vectorstore.similarity_search(query, k=k)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build a HotpotQA vector DB")
    parser.add_argument("--build", action="store_true", help="Build the vector DB")
    parser.add_argument("--rebuild", action="store_true",
                        help="Drop and rebuild the existing collection")
    parser.add_argument("--variant", default=DEFAULT_VARIANT,
                        choices=["distractor", "fullwiki"],
                        help=f"HotpotQA variant (default: {DEFAULT_VARIANT})")
    parser.add_argument("--split", default=DEFAULT_SPLIT,
                        help=f"Dataset split (default: {DEFAULT_SPLIT})")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
                        help="Number of QA samples (0 = full split, default: 200)")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION,
                        help=f"Chroma collection name (default: {DEFAULT_COLLECTION})")
    parser.add_argument("--persist-dir", default=str(DB_DIR),
                        help=f"Chroma persist directory (default: {DB_DIR})")
    parser.add_argument("--embedding", default=DEFAULT_EMBEDDING,
                        help=f"Embedding model id (default: {DEFAULT_EMBEDDING})")
    parser.add_argument("--device", default=None,
                        help="Device for embeddings (cuda/cpu, auto-detected)")
    parser.add_argument("--search", default=None,
                        help="Run a similarity search against the existing DB")
    parser.add_argument("--search-k", type=int, default=5,
                        help="Top-k for --search (default: 5)")

    args = parser.parse_args()

    builder = HotpotQAVectorDB(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        embedding_model=args.embedding,
        device=args.device,
    )

    if args.build:
        builder.build(
            variant=args.variant,
            split=args.split,
            num_samples=args.num_samples,
            rebuild=args.rebuild,
        )
    elif args.search:
        results = builder.search(args.search, k=args.search_k)
        for i, doc in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Title:  {doc.metadata.get('title', 'unknown')}")
            print(doc.page_content[:500])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
