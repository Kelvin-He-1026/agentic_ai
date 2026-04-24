"""
VectorDB — Build and manage vector databases for multimodal RAG
================================================================

Parses documents (PDF, text, and more) using `unstructured` to extract
text, tables, and images.  Images/charts are summarised into text via a
VLM so that *all* content is searchable through a single text-embedding
vector space.

Supported backends: **Redis** (default), **Chroma**, **FAISS**.

Usage as library:
    from vectorDB import VectorDBBuilder

    # Build a new database (Redis default)
    builder = VectorDBBuilder()
    builder.build()

    # Append new documents later
    builder.build(doc_path="/part3/documents/new_paper.pdf", append=True)

    # Use Chroma instead
    builder = VectorDBBuilder(backend="chroma")
    builder.build()

    # Get a LangChain retriever for downstream RAG
    retriever = builder.as_retriever(search_kwargs={"k": 5})

    # Custom embedding model (any model_loader preset or HF model ID)
    builder = VectorDBBuilder(embedding_model="bge-base")
    builder.build()

Usage as CLI:
    # Build with defaults (Redis, bge-large, all docs)
    python vectorDB.py --build

    # Build with Chroma backend
    python vectorDB.py --build --backend chroma

    # Append a single new file
    python vectorDB.py --build --append --doc-path /path/to/new_doc.pdf

    # Rebuild from scratch
    python vectorDB.py --build --backend faiss --embedding bge-base

    # Multimodal CLIP embedding (images embedded via image encoder)
    python vectorDB.py --build --backend chroma --embedding clip-vit-large
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Make pip-installed CUDA libs visible to cuDNN (suppresses libnvrtc warning)
# ---------------------------------------------------------------------------
_VENV = Path(sys.executable).resolve().parent.parent
for _cuda_pkg in ("nvidia/cuda_nvrtc/lib", "nvidia/cu13/lib"):
    _lib_dir = _VENV / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / _cuda_pkg
    if _lib_dir.is_dir():
        os.environ["LD_LIBRARY_PATH"] = (
            str(_lib_dir) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        )

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent        # part3/
MODELS_DIR = PROJECT_DIR / "models"
DB_DIR = PROJECT_DIR / "db"
DOCS_DIR = PROJECT_DIR / "documents"


def _canonicalize_path(path: str | Path) -> Path:
    """Return a normalized absolute path for stable metadata and manifests."""
    return Path(path).expanduser().resolve()


def _set_offline_if_cached(model_id: str) -> None:
    """Enable HF offline mode only when the model already exists locally."""
    local_name = model_id.replace("/", "--")
    local_path = MODELS_DIR / local_name
    if local_path.exists() and any(local_path.iterdir()):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    else:
        # Model not cached locally — allow HF downloads
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
# BGE-large is a strong general-purpose text embedding (1024-dim) and is
# already downloaded locally.  All multimodal content (tables, images,
# charts) is converted to text descriptions before embedding so a
# high-quality text model is the best default.
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 500
# Chroma is the default — it's file-based (no server required), persists to
# disk automatically, and works on any machine.  Redis requires a running
# redis-server and is available as an opt-in backend.
DEFAULT_BACKEND: Literal["redis", "chroma", "faiss"] = "chroma"

logger = logging.getLogger(__name__)

BackendType = Literal["redis", "chroma", "faiss"]

# Known CLIP model prefixes — used to auto-detect multimodal mode
_CLIP_PREFIXES = ("openai/clip-", "clip-vit-")


def _is_clip_model(name: str) -> bool:
    """Return True if the embedding model name refers to a CLIP model."""
    return any(name.lower().startswith(p) for p in _CLIP_PREFIXES)


# ---------------------------------------------------------------------------
# CLIPEmbeddings — LangChain-compatible multimodal embeddings
# ---------------------------------------------------------------------------
class CLIPEmbeddings:
    """LangChain-compatible embedding wrapper around a CLIP model.

    Produces aligned text and image embeddings in the same 768-dim (ViT-L)
    or 512-dim (ViT-B) vector space, enabling cross-modal similarity search
    (text query → image result and vice-versa).

    Usage::

        emb = CLIPEmbeddings("openai/clip-vit-large-patch14", device="cuda:0")
        text_vecs  = emb.embed_documents(["hello world"])
        image_vecs = emb.embed_images([pil_image])
        query_vec  = emb.embed_query("a photo of a cat")
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        import torch
        from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast

        self.device = device
        self.model = CLIPModel.from_pretrained(
            model_path, torch_dtype=torch.float16,
        ).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_path)

    # -- LangChain Embeddings interface -------------------------------------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via CLIP text encoder."""
        import torch

        all_embeds = []
        # Process in batches of 64 to avoid OOM on long doc lists
        for i in range(0, len(texts), 64):
            batch = texts[i : i + 64]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=77,
            ).to(self.device)
            with torch.no_grad():
                feats = self.model.get_text_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeds.extend(feats.cpu().float().tolist())
        return all_embeds

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_documents([text])[0]

    # -- Image embedding (not part of LangChain interface) ------------------

    def embed_images(self, images: list) -> list[list[float]]:
        """Embed a list of PIL Images via CLIP image encoder.

        Returns vectors in the *same* space as ``embed_documents`` /
        ``embed_query``, so text queries naturally match image results.
        """
        import torch

        all_embeds = []
        for i in range(0, len(images), 32):
            batch = images[i : i + 32]
            inputs = self.processor(
                images=batch, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                feats = self.model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeds.extend(feats.cpu().float().tolist())
        return all_embeds

    def embed_image(self, image) -> list[float]:
        """Embed a single PIL Image."""
        return self.embed_images([image])[0]


# ---------------------------------------------------------------------------
# Document parsing helpers
# ---------------------------------------------------------------------------
def _is_tesseract_available() -> bool:
    """Check if tesseract OCR binary is on PATH."""
    import shutil
    return shutil.which("tesseract") is not None


def _parse_pdf(file_path: str | Path) -> list[dict]:
    """
    Parse a PDF into structured elements using unstructured's partition_pdf.

    Returns a list of dicts with keys: text, metadata (type, page, source).
    Handles text, tables, and images (images are base64-encoded for optional
    VLM summarisation downstream).

    Automatically falls back from ``hi_res`` to ``fast`` strategy when
    tesseract is not installed.
    """
    from unstructured.partition.pdf import partition_pdf

    file_path = _canonicalize_path(file_path)

    if _is_tesseract_available():
        strategy = "hi_res"
    else:
        strategy = "fast"
        logger.info(
            "tesseract not found — falling back to strategy='fast' "
            "(no OCR; install tesseract-ocr for hi_res mode)"
        )
        print("  ⚠ tesseract not installed — using fast PDF strategy "
              "(install tesseract-ocr for full OCR + layout detection)")

    # Build kwargs based on strategy
    kwargs: dict = {
        "filename": str(file_path),
        "strategy": strategy,
    }
    if strategy == "hi_res":
        kwargs.update(
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Figure"],
            infer_table_structure=True,
        )

    elements = partition_pdf(**kwargs)

    parsed = []
    for el in elements:
        meta = {
            "source": str(file_path),
            "element_type": type(el).__name__,
            "page_number": getattr(el.metadata, "page_number", None),
        }

        if type(el).__name__ == "Table":
            # Tables: use HTML representation for better structure retention
            text = getattr(el.metadata, "text_as_html", None) or str(el)
            meta["content_type"] = "table"
        elif type(el).__name__ in ("Image", "Figure"):
            # Images: store base64 data in metadata; text will be the
            # alt-text or a placeholder until VLM summarisation runs.
            img_b64 = getattr(el.metadata, "image_base64", None)
            text = str(el) if str(el).strip() else "[image]"
            meta["content_type"] = "image"
            if img_b64:
                meta["image_base64"] = img_b64
        else:
            text = str(el)
            meta["content_type"] = "text"

        if text and text.strip():
            parsed.append({"text": text.strip(), "metadata": meta})

    return parsed


def _parse_text(file_path: str | Path) -> list[dict]:
    """Parse a plain-text file."""
    from unstructured.partition.text import partition_text

    file_path = _canonicalize_path(file_path)

    elements = partition_text(filename=str(file_path))
    return [
        {
            "text": str(el).strip(),
            "metadata": {
                "source": str(file_path),
                "element_type": type(el).__name__,
                "content_type": "text",
            },
        }
        for el in elements
        if str(el).strip()
    ]


def _parse_file(file_path: Path) -> list[dict]:
    """Route a file to the correct parser by extension."""
    file_path = _canonicalize_path(file_path)
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(file_path)
    elif suffix in (".txt", ".md"):
        return _parse_text(file_path)
    else:
        # Fallback: try unstructured's auto-detect
        try:
            from unstructured.partition.auto import partition
            elements = partition(filename=str(file_path))
            return [
                {
                    "text": str(el).strip(),
                    "metadata": {
                        "source": str(file_path),
                        "element_type": type(el).__name__,
                        "content_type": "text",
                    },
                }
                for el in elements
                if str(el).strip()
            ]
        except Exception as exc:
            logger.warning("Skipping %s: unsupported format (%s)", file_path, exc)
            return []


def _file_hash(path: Path) -> str:
    """SHA-256 hex digest of a file (for change detection)."""
    path = _canonicalize_path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Manifest — tracks which files have already been indexed
# ---------------------------------------------------------------------------
def _load_manifest(db_path: Path) -> dict:
    manifest_path = db_path / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {}


def _save_manifest(db_path: Path, manifest: dict) -> None:
    db_path.mkdir(parents=True, exist_ok=True)
    (db_path / "manifest.json").write_text(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# VectorDBBuilder
# ---------------------------------------------------------------------------
class VectorDBBuilder:
    """Build and manage vector databases with multimodal document support.

    Parameters
    ----------
    backend : str
        Vector store backend: ``"redis"`` (default), ``"chroma"``, or ``"faiss"``.
    embedding_model : str
        Model name — can be a model_loader preset (e.g. ``"bge-base"``,
        ``"minilm"``) or any HuggingFace model ID.
    db_path : str | Path | None
        Directory for persisting the vector store. Defaults to ``part3/db``.
    chunk_size : int
        Maximum chunk size in characters (default 2000).
    chunk_overlap : int
        Overlap between consecutive chunks (default 500).
    device : str
        PyTorch device for the embedding model.
    gpu_ids : list[int] | None
        GPU indices passed through to ``model_loader.load_embedding``.
    redis_url : str
        Redis connection URL (only used when ``backend="redis"``).
    index_name : str
        Name of the vector index / collection.
    use_vlm_for_images : bool
        When True, images extracted from PDFs are summarised by a VLM
        before embedding.  Requires a VLM model available via model_loader.
    vlm_model : str
        VLM preset or HF model ID used for image summarisation.
    """

    def __init__(
        self,
        backend: BackendType = DEFAULT_BACKEND,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        db_path: str | Path | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        device: str = "cuda",
        gpu_ids: list[int] | None = None,
        redis_url: str = "redis://localhost:6379",
        index_name: str = "documents",
        use_vlm_for_images: bool = False,
        vlm_model: str = "llava-1.5-7b",
    ):
        self.backend = backend
        self.embedding_model_name = embedding_model
        self.db_path = Path(db_path) if db_path else DB_DIR
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = device
        self.gpu_ids = gpu_ids
        self.redis_url = redis_url
        self.index_name = index_name
        self.use_vlm_for_images = use_vlm_for_images
        self.vlm_model = vlm_model

        # Lazily initialised
        self._embeddings = None
        self._vectorstore = None
        self._vlm = None
        self._vlm_processor = None

    def _is_low_information_chunk(self, doc) -> bool:
        """Heuristic filter for tiny heading-like chunks that hurt retrieval."""
        text = re.sub(r"\s+", " ", (doc.page_content or "")).strip()
        if not text:
            return True

        if doc.metadata.get("content_type") != "text":
            return False

        word_count = len(text.split())
        has_sentence_punct = any(ch in text for ch in ".:;?!")
        if len(text) < 40 and word_count <= 6 and not has_sentence_punct:
            return True
        return False

    def _postprocess_search_results(self, results: list, k: int) -> list:
        """Dedupe repeated chunks and drop very low-information headings."""
        filtered = []
        seen = set()

        for doc in results:
            if self._is_low_information_chunk(doc):
                continue

            text_norm = re.sub(r"\s+", " ", (doc.page_content or "")).strip().lower()
            source = doc.metadata.get("source", "")
            source_norm = Path(source).name.lower()
            key = (source_norm, text_norm)
            if key in seen:
                continue
            seen.add(key)
            filtered.append(doc)
            if len(filtered) >= k:
                break

        return filtered

    # -- lazy loaders -------------------------------------------------------

    @property
    def is_clip(self) -> bool:
        """True when the embedding model is CLIP (multimodal)."""
        return _is_clip_model(self.embedding_model_name)

    @property
    def embeddings(self):
        """LangChain-compatible embedding model (loaded on first access).

        Returns CLIPEmbeddings for CLIP models, HuggingFaceEmbeddings otherwise.
        """
        if self._embeddings is None:
            _set_offline_if_cached(self.embedding_model_name)
            if self.is_clip:
                from model_loader import _get_preset, _resolve_model_path
                preset = _get_preset(self.embedding_model_name)
                model_id = preset["model_id"]
                path = _resolve_model_path(
                    model_id,
                    self.embedding_model_name
                    if self.embedding_model_name
                    in __import__("model_loader").MODEL_PRESETS
                    else None,
                )
                embed_device = (
                    f"cuda:{self.gpu_ids[0]}" if self.gpu_ids else self.device
                )
                print(f"  Loading CLIP embedding: {path} (device={embed_device})")
                self._embeddings = CLIPEmbeddings(path, device=embed_device)
                print(f"  CLIP embedding ready (multimodal)")
            else:
                from model_loader import load_embedding
                self._embeddings = load_embedding(
                    self.embedding_model_name,
                    device=self.device,
                    gpu_ids=self.gpu_ids,
                )
        return self._embeddings

    def _get_vlm(self):
        """Load VLM for image summarisation (only when needed)."""
        if self._vlm is None:
            from model_loader import load_vlm
            self._vlm, self._vlm_processor = load_vlm(
                self.vlm_model,
                device=self.device,
                gpu_ids=self.gpu_ids,
            )
        return self._vlm, self._vlm_processor

    # -- document loading ---------------------------------------------------

    def load_documents(
        self,
        doc_path: str | Path | None = None,
        file_filter: list[str] | None = None,
    ) -> list[dict]:
        """Parse documents from *doc_path* (file or directory).

        Parameters
        ----------
        doc_path : path, optional
            File or directory.  Defaults to ``part3/documents/``.
        file_filter : list[str], optional
            Only process files whose names are in this list.

        Returns
        -------
        list[dict]
            Each dict has ``text`` and ``metadata`` keys.
        """
        doc_path = Path(doc_path) if doc_path else DOCS_DIR

        if doc_path.is_file():
            files = [doc_path]
        elif doc_path.is_dir():
            files = sorted(
                f for f in doc_path.iterdir()
                if f.is_file() and not f.name.startswith(".")
            )
        else:
            raise FileNotFoundError(f"Document path not found: {doc_path}")

        if file_filter:
            files = [f for f in files if f.name in file_filter]

        all_parsed: list[dict] = []
        for fpath in files:
            logger.info("Parsing %s", fpath.name)
            print(f"  Parsing: {fpath.name}")
            parsed = _parse_file(fpath)
            all_parsed.extend(parsed)
            print(f"    → {len(parsed)} elements extracted")

        return all_parsed

    # -- VLM image summarisation -------------------------------------------

    def _summarise_images(self, documents: list[dict]) -> list[dict]:
        """Replace image placeholders with VLM-generated text summaries."""
        if not self.use_vlm_for_images:
            return documents

        import base64
        import torch
        from PIL import Image
        from io import BytesIO

        model, processor = self._get_vlm()

        for doc in documents:
            if doc["metadata"].get("content_type") != "image":
                continue
            img_b64 = doc["metadata"].get("image_base64")
            if not img_b64:
                continue

            try:
                img_bytes = base64.b64decode(img_b64)
                image = Image.open(BytesIO(img_bytes)).convert("RGB")

                prompt = (
                    "USER: <image>\nDescribe this image in detail, "
                    "including any text, data, charts, or diagrams.\nASSISTANT:"
                )
                inputs = processor(text=prompt, images=image, return_tensors="pt")
                input_device = next(model.parameters()).device
                inputs = {k: v.to(input_device) for k, v in inputs.items()}

                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=256)

                summary = processor.decode(output[0], skip_special_tokens=True)
                # Extract only the assistant's response
                if "ASSISTANT:" in summary:
                    summary = summary.split("ASSISTANT:")[-1].strip()

                doc["text"] = summary
                doc["metadata"]["vlm_summarised"] = True
                print(f"    → Summarised image from {doc['metadata']['source']}")
            except Exception as exc:
                logger.warning("Failed to summarise image: %s", exc)

        return documents

    # -- chunking -----------------------------------------------------------

    def _chunk_documents(self, documents: list[dict]):
        """Split parsed elements into LangChain Documents with chunking.

        Returns
        -------
        tuple(list[Document], list[dict])
            (text_chunks, image_elements) — image_elements are separated
            out for direct CLIP image embedding when in multimodal mode.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

        lc_docs = []
        image_elements = []
        for doc in documents:
            meta = {k: v for k, v in doc["metadata"].items()
                    if k != "image_base64"}

            # When using CLIP, keep image elements separate for direct
            # image-encoder embedding instead of text embedding.
            if (
                self.is_clip
                and doc["metadata"].get("content_type") == "image"
                and doc["metadata"].get("image_base64")
            ):
                image_elements.append(doc)
                continue

            lc_docs.append(Document(page_content=doc["text"], metadata=meta))

        chunks = splitter.split_documents(lc_docs)
        return chunks, image_elements

    # -- CLIP image embedding into vector store ------------------------------

    def _add_clip_images(self, vs, image_elements: list[dict]):
        """Embed images via CLIP image encoder and add to the vector store.

        Each image is embedded directly (no text conversion) into the same
        vector space as text embeddings, enabling text-query → image-result
        cross-modal retrieval.
        """
        if not image_elements:
            return

        import base64
        from io import BytesIO
        from PIL import Image
        from langchain_core.documents import Document

        pil_images = []
        docs = []
        for elem in image_elements:
            try:
                img_bytes = base64.b64decode(elem["metadata"]["image_base64"])
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                pil_images.append(img)
                meta = {k: v for k, v in elem["metadata"].items()
                        if k != "image_base64"}
                meta["clip_image_embedded"] = True
                # page_content is a description for display; the actual
                # embedding comes from the CLIP image encoder
                docs.append(Document(
                    page_content=elem["text"] or "[image]",
                    metadata=meta,
                ))
            except Exception as exc:
                logger.warning("Skipping image: %s", exc)

        if not pil_images:
            return

        print(f"  Embedding {len(pil_images)} image(s) via CLIP image encoder...")
        image_vectors = self.embeddings.embed_images(pil_images)

        # Insert pre-computed embeddings into the vector store
        texts = [d.page_content for d in docs]
        metadatas = [d.metadata for d in docs]
        text_embedding_pairs = list(zip(texts, image_vectors))

        if hasattr(vs, "add_embeddings"):
            # FAISS
            vs.add_embeddings(text_embedding_pairs, metadatas=metadatas)
        elif hasattr(vs, "_collection"):
            # Chroma — use the underlying collection API
            import uuid
            vs._collection.add(
                ids=[str(uuid.uuid4()) for _ in docs],
                embeddings=image_vectors,
                documents=texts,
                metadatas=metadatas,
            )
        else:
            # Redis / fallback: add as text (CLIP text encoder will be used,
            # which is close but not identical to image encoder)
            vs.add_documents(docs)

        print(f"    → {len(pil_images)} image embeddings added")

    # -- vector store backends ----------------------------------------------

    def _build_redis(self, chunks, append: bool):
        """Build or append to a Redis vector store.

        Requires a running Redis server with the RediSearch module (Redis Stack).
        """
        # Pre-flight check: verify Redis is reachable before doing any work
        try:
            import redis as _redis
            r = _redis.Redis.from_url(self.redis_url)
            r.ping()
            # Check for RediSearch module
            modules = [m[b"name"].decode() for m in r.module_list()]
            if "search" not in modules and "ft" not in modules:
                raise RuntimeError(
                    "Redis is running but the RediSearch module is not loaded.\n"
                    "Install Redis Stack (includes RediSearch): "
                    "https://redis.io/docs/stack/get-started/\n"
                    "Or use --backend chroma (no server required)."
                )
        except (ConnectionError, OSError, _redis.ConnectionError) as exc:
            raise RuntimeError(
                f"Cannot connect to Redis at {self.redis_url}: {exc}\n"
                "Either:\n"
                "  1. Start Redis Stack: sudo apt install redis-stack-server && redis-stack-server\n"
                "  2. Or use a file-based backend: --backend chroma  (recommended)\n"
                "  3. Or use: --backend faiss"
            ) from exc

        from langchain_community.vectorstores import Redis

        schema_path = self.db_path / "redis_schema.yaml"
        self.db_path.mkdir(parents=True, exist_ok=True)

        if append and self._vectorstore is not None:
            self._vectorstore.add_documents(chunks)
            return self._vectorstore

        # Try to load existing index for appending
        if append:
            try:
                vs = Redis.from_existing_index(
                    embedding=self.embeddings,
                    redis_url=self.redis_url,
                    index_name=self.index_name,
                    schema=str(schema_path) if schema_path.exists() else None,
                )
                vs.add_documents(chunks)
                self._vectorstore = vs
                return vs
            except Exception:
                logger.info("No existing Redis index found, creating new one")

        vs = Redis.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            redis_url=self.redis_url,
            index_name=self.index_name,
        )
        # Save schema for future reconnection
        vs.write_schema(str(schema_path))
        self._vectorstore = vs
        return vs

    def _build_chroma(self, chunks, append: bool):
        """Build or append to a Chroma vector store."""
        from langchain_community.vectorstores import Chroma

        persist_dir = str(self.db_path / "chroma")

        if append:
            try:
                vs = Chroma(
                    collection_name=self.index_name,
                    embedding_function=self.embeddings,
                    persist_directory=persist_dir,
                )
                vs.add_documents(chunks)
                self._vectorstore = vs
                return vs
            except Exception:
                logger.info("No existing Chroma collection, creating new one")

        vs = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.index_name,
            persist_directory=persist_dir,
        )
        self._vectorstore = vs
        return vs

    def _build_faiss(self, chunks, append: bool):
        """Build or append to a FAISS vector store."""
        from langchain_community.vectorstores import FAISS

        faiss_dir = self.db_path / "faiss"
        index_file = faiss_dir / "index.faiss"

        if append and index_file.exists():
            vs = FAISS.load_local(
                str(faiss_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            vs.add_documents(chunks)
            vs.save_local(str(faiss_dir))
            self._vectorstore = vs
            return vs

        vs = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings,
        )
        faiss_dir.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(faiss_dir))
        self._vectorstore = vs
        return vs

    # -- main build entry point ---------------------------------------------

    def build(
        self,
        doc_path: str | Path | None = None,
        append: bool = False,
    ):
        """Build the vector database from documents.

        Parameters
        ----------
        doc_path : path, optional
            File or directory of documents. Defaults to ``part3/documents/``.
        append : bool
            If True, add new/changed files to the existing database instead
            of rebuilding from scratch.

        Returns
        -------
        The LangChain vector store instance.
        """
        doc_path = _canonicalize_path(doc_path) if doc_path else DOCS_DIR.resolve()
        manifest = _load_manifest(self.db_path) if append else {}

        # Determine which files to process
        if doc_path.is_file():
            all_files = [_canonicalize_path(doc_path)]
        else:
            all_files = sorted(
                _canonicalize_path(f) for f in Path(doc_path).iterdir()
                if f.is_file() and not f.name.startswith(".")
            )

        if append:
            files_to_process = []
            for f in all_files:
                fhash = _file_hash(f)
                fkey = str(_canonicalize_path(f))
                if fkey not in manifest or manifest[fkey] != fhash:
                    files_to_process.append(f)
            if not files_to_process:
                print("  No new or changed files to process.")
                self.load_existing()
                return self._vectorstore
            print(f"  Found {len(files_to_process)} new/changed file(s) to index")
        else:
            files_to_process = all_files

        # Parse documents
        all_parsed = []
        for fpath in files_to_process:
            print(f"  Parsing: {fpath.name}")
            parsed = _parse_file(fpath)
            all_parsed.extend(parsed)
            print(f"    → {len(parsed)} elements extracted")

        if not all_parsed:
            print("  No content extracted from documents.")
            return None

        # Summarise images with VLM if enabled (only for non-CLIP mode;
        # CLIP embeds images directly via its image encoder)
        if not self.is_clip:
            all_parsed = self._summarise_images(all_parsed)

        # Chunk text; separate image elements for CLIP image embedding
        chunks, image_elements = self._chunk_documents(all_parsed)
        print(f"  Total text chunks: {len(chunks)}")
        if image_elements:
            print(f"  Image elements for CLIP embedding: {len(image_elements)}")

        # Build vector store from text chunks
        print(f"  Building {self.backend} vector store...")
        backend_builders = {
            "redis": self._build_redis,
            "chroma": self._build_chroma,
            "faiss": self._build_faiss,
        }
        builder_fn = backend_builders.get(self.backend)
        if builder_fn is None:
            raise ValueError(
                f"Unknown backend '{self.backend}'. "
                f"Choose from: {', '.join(backend_builders)}"
            )

        vs = builder_fn(chunks, append=append)

        # Embed images directly via CLIP image encoder (multimodal mode)
        if self.is_clip and image_elements:
            self._add_clip_images(vs, image_elements)

        # Update manifest
        for f in files_to_process:
            manifest[str(_canonicalize_path(f))] = _file_hash(f)
        _save_manifest(self.db_path, manifest)

        print(f"  Vector store ready ({self.backend}, "
              f"{len(chunks)} chunks, "
              f"embedding={self.embedding_model_name})")
        return vs

    # -- load existing store ------------------------------------------------

    def load_existing(self):
        """Load a previously built vector store from disk.

        Returns
        -------
        The LangChain vector store instance, or None if not found.
        """
        if self.backend == "redis":
            from langchain_community.vectorstores import Redis
            schema_path = self.db_path / "redis_schema.yaml"
            self._vectorstore = Redis.from_existing_index(
                embedding=self.embeddings,
                redis_url=self.redis_url,
                index_name=self.index_name,
                schema=str(schema_path) if schema_path.exists() else None,
            )
        elif self.backend == "chroma":
            from langchain_community.vectorstores import Chroma
            persist_dir = str(self.db_path / "chroma")
            self._vectorstore = Chroma(
                collection_name=self.index_name,
                embedding_function=self.embeddings,
                persist_directory=persist_dir,
            )
        elif self.backend == "faiss":
            from langchain_community.vectorstores import FAISS
            faiss_dir = self.db_path / "faiss"
            self._vectorstore = FAISS.load_local(
                str(faiss_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        print(f"  Loaded existing {self.backend} vector store")
        return self._vectorstore

    # -- retriever ----------------------------------------------------------

    def as_retriever(self, **kwargs):
        """Return a LangChain retriever from the vector store.

        Pass ``search_kwargs={"k": 5}`` etc. to control retrieval.
        """
        if self._vectorstore is None:
            self.load_existing()
        return self._vectorstore.as_retriever(**kwargs)

    # -- search convenience -------------------------------------------------

    def search(self, query: str, k: int = 5) -> list:
        """Run a similarity search and return top-k results."""
        if self._vectorstore is None:
            self.load_existing()
        raw_results = self._vectorstore.similarity_search(query, k=max(k * 3, 10))
        return self._postprocess_search_results(raw_results, k)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="VectorDB — Build and manage vector databases for multimodal RAG",
    )
    parser.add_argument("--build", action="store_true",
                        help="Build the vector database from documents")
    parser.add_argument("--append", action="store_true",
                        help="Append new/changed files instead of rebuilding")
    parser.add_argument("--backend", default=DEFAULT_BACKEND,
                        choices=["redis", "chroma", "faiss"],
                        help="Vector store backend (default: redis)")
    parser.add_argument("--embedding", default=DEFAULT_EMBEDDING_MODEL,
                        help="Embedding model preset or HF model ID")
    parser.add_argument("--doc-path", default=None,
                        help="Path to document file or directory")
    parser.add_argument("--db-path", default=None,
                        help="Path to store the vector database")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help="Chunk size in characters (default: 2000)")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                        help="Chunk overlap in characters (default: 500)")
    parser.add_argument("--redis-url", default="redis://localhost:6379",
                        help="Redis connection URL (for redis backend)")
    parser.add_argument("--index-name", default="documents",
                        help="Name of the vector index/collection")
    parser.add_argument("--device", default="cuda",
                        help="Device for embedding model (cuda/cpu)")
    parser.add_argument("--gpu", default=None,
                        help="GPU index (e.g. '0' or '0,1')")
    parser.add_argument("--use-vlm", action="store_true",
                        help="Use VLM to summarise images from PDFs")
    parser.add_argument("--vlm-model", default="llava-1.5-7b",
                        help="VLM model for image summarisation")
    parser.add_argument("--search", default=None,
                        help="Run a search query against existing DB")
    parser.add_argument("--search-k", type=int, default=5,
                        help="Number of results for search (default: 5)")

    args = parser.parse_args()

    gpu_ids = None
    if args.gpu is not None:
        gpu_ids = [int(x.strip()) for x in args.gpu.split(",")]

    builder = VectorDBBuilder(
        backend=args.backend,
        embedding_model=args.embedding,
        db_path=args.db_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        device=args.device,
        gpu_ids=gpu_ids,
        redis_url=args.redis_url,
        index_name=args.index_name,
        use_vlm_for_images=args.use_vlm,
        vlm_model=args.vlm_model,
    )

    if args.build:
        builder.build(doc_path=args.doc_path, append=args.append)
    elif args.search:
        results = builder.search(args.search, k=args.search_k)
        for i, doc in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Source: {doc.metadata.get('source', 'unknown')}")
            print(f"Type:   {doc.metadata.get('content_type', 'text')}")
            print(doc.page_content[:500])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
