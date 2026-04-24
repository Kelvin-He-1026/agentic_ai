"""
Model Loader — Download, cache, and load LLM / VLM / Embedding models
=======================================================================
Manages HuggingFace models for the Agentic RAG benchmark suite.

Two storage modes:
  --download   : Download model weights to a local directory (part3/models/)
                 for reproducible, offline-capable, repeated runs.
  (default)    : Use HuggingFace cache (~/.cache/huggingface/) — no extra
                 disk copy, auto-cached on first use.

Usage as CLI:
  # List available preset models
  python model_loader.py --list

  # Download a preset to local storage
  python model_loader.py --download --preset tinyllama

  # Download a custom model by HF ID
  python model_loader.py --download --model-id meta-llama/Llama-3.2-1B

  # Download all presets
  python model_loader.py --download --preset all

Usage as library:
  from model_loader import load_llm, load_embedding, load_vlm, MODELS_DIR
  llm, tokenizer = load_llm("tinyllama", device="cuda")
  embeddings     = load_embedding("minilm")
  vlm, processor = load_vlm("llava-1.5", device="cuda")

GPU selection:
  # Single GPU (e.g. GPU 2)
  llm, tokenizer = load_llm("tinyllama", gpu_ids=[2])
  # Multi-GPU model parallelism (across GPUs 0,1,2,3)
  llm, tokenizer = load_llm("qwen2.5-7b", gpu_ids=[0, 1, 2, 3])
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# part3/models/ — local storage for downloaded models
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# ---------------------------------------------------------------------------
# Model registry — presets with sensible defaults
# ---------------------------------------------------------------------------
MODEL_PRESETS: dict[str, dict] = {
    # --- LLMs (causal language models) ---
    "tinyllama": {
        "type": "llm",
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "1.1B param, fast, good for benchmarking",
    },
    "phi-2": {
        "type": "llm",
        "model_id": "microsoft/phi-2",
        "description": "2.7B param, strong reasoning for size",
    },
    "llama-3.2-1b": {
        "type": "llm",
        "model_id": "meta-llama/Llama-3.2-1B",
        "description": "1B param, Meta Llama 3.2 (may need HF token)",
    },
    "llama-3.2-3b": {
        "type": "llm",
        "model_id": "meta-llama/Llama-3.2-3B",
        "description": "3B param, Meta Llama 3.2 (may need HF token)",
    },
    "qwen2.5-1.5b": {
        "type": "llm",
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "description": "1.5B param, strong multilingual",
    },
    "qwen2.5-7b": {
        "type": "llm",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "description": "7B param, high quality generation",
    },
    "qwen3-8b": {
        "type": "llm",
        "model_id": "Qwen/Qwen3-8B",
        "description": "8B param, Qwen3, strong reasoning + tool use",
    },
    "qwen3-14b": {
        "type": "llm",
        "model_id": "Qwen/Qwen3-14B",
        "description": "14B param, Qwen3, strong reasoning + tool use",
    },
    "qwen3-8b-ov": {
        "type": "llm-openvino",
        "model_id": "OpenVINO/Qwen3-8B-int8-ov",
        "description": "8B param, Qwen3, INT8 OpenVINO optimized",
    },
    "qwen3-8b-ov-fp16": {
        "type": "llm-openvino",
        "model_id": "OpenVINO/Qwen3-8B-fp16-ov",
        "description": "8B param, Qwen3, FP16 OpenVINO optimized",
    },
    "qwen3-8b-ov-int4": {
        "type": "llm-openvino",
        "model_id": "OpenVINO/Qwen3-8B-int4-ov",
        "description": "8B param, Qwen3, INT4 OpenVINO optimized",
    },
    "qwen3-14b-ov": {
        "type": "llm-openvino",
        "model_id": "OpenVINO/Qwen3-14B-int8-ov",
        "description": "14B param, Qwen3, INT8 OpenVINO optimized",
    },

    # --- Embedding models ---
    "minilm": {
        "type": "embedding",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "384-dim, fast, lightweight",
    },
    "bge-small": {
        "type": "embedding",
        "model_id": "BAAI/bge-small-en-v1.5",
        "description": "384-dim, strong retrieval quality",
    },
    "bge-base": {
        "type": "embedding",
        "model_id": "BAAI/bge-base-en-v1.5",
        "description": "768-dim, higher quality retrieval",
    },
    "e5-large": {
        "type": "embedding",
        "model_id": "intfloat/e5-large-v2",
        "description": "1024-dim, top-tier embedding quality",
    },
    "bge-large": {
        "type": "embedding",
        "model_id": "BAAI/bge-large-en-v1.5",
        "description": "1024-dim, top-tier retrieval, already downloaded",
    },

    # --- Multimodal embedding (CLIP) ---
    "clip-vit-large": {
        "type": "clip",
        "model_id": "openai/clip-vit-large-patch14",
        "description": "768-dim, multimodal (text+image) CLIP ViT-L/14",
    },
    "clip-vit-base": {
        "type": "clip",
        "model_id": "openai/clip-vit-base-patch32",
        "description": "512-dim, multimodal (text+image) CLIP ViT-B/32, fast",
    },

    # --- Vision-Language Models (VLMs) ---
    "llava-1.5-7b": {
        "type": "vlm",
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "description": "7B param, multimodal (image+text)",
    },
}


# ---------------------------------------------------------------------------
# Resolve model path: local download vs HF cache
# ---------------------------------------------------------------------------
def _resolve_model_path(model_id: str, local_name: str | None = None) -> str:
    """
    If model was downloaded locally to MODELS_DIR, return that path.
    Otherwise return the HF model_id (will use HF cache).
    """
    if local_name:
        local_path = MODELS_DIR / local_name
    else:
        # Convert "org/model-name" → "org--model-name"
        local_path = MODELS_DIR / model_id.replace("/", "--")

    if local_path.exists() and any(local_path.iterdir()):
        return str(local_path)
    return model_id


def _get_preset(name: str) -> dict:
    """Look up a preset by name, or treat it as a raw HF model_id."""
    if name in MODEL_PRESETS:
        return MODEL_PRESETS[name]
    # Not a preset — assume it's a raw HF model_id
    return {"type": "llm", "model_id": name, "description": "custom model"}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_model(model_id: str, local_name: str | None = None,
                   model_type: str = "llm") -> Path:
    """
    Download model weights to MODELS_DIR for offline / repeated use.
    Returns the local path.
    """
    from huggingface_hub import snapshot_download

    if local_name:
        dest = MODELS_DIR / local_name
    else:
        dest = MODELS_DIR / model_id.replace("/", "--")

    dest.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {model_id} → {dest}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    print(f"  Done: {dest}")
    return dest


# ---------------------------------------------------------------------------
# GPU selection helpers
# ---------------------------------------------------------------------------
def _resolve_device(device: str, gpu_ids: list[int] | None) -> tuple[str, str | dict]:
    """
    Resolve device and device_map from gpu_ids.

    Returns (effective_device, device_map) where:
      - effective_device: 'cuda:N' or 'cpu' (for tokenizer .to() calls)
      - device_map: value to pass to from_pretrained()

    gpu_ids=None     → use `device` as-is (backward compatible)
    gpu_ids=[2]      → single GPU, device_map="cuda:2"
    gpu_ids=[0,1,2]  → multi-GPU model parallelism, device_map="auto"
                       with max_memory constraints
    """
    if gpu_ids is None:
        return device, device

    if not torch.cuda.is_available():
        print("  Warning: gpu_ids specified but CUDA unavailable, using CPU")
        return "cpu", "cpu"

    n_available = torch.cuda.device_count()
    for gid in gpu_ids:
        if gid >= n_available:
            raise ValueError(
                f"GPU {gid} requested but only {n_available} GPUs available "
                f"(indices 0–{n_available - 1})")

    if len(gpu_ids) == 1:
        dev = f"cuda:{gpu_ids[0]}"
        return dev, dev

    # Multi-GPU: use device_map="auto" with max_memory to restrict to
    # selected GPUs only
    max_memory = {}
    for gid in gpu_ids:
        props = torch.cuda.get_device_properties(gid)
        # Reserve 1 GB for overhead
        mem_gb = round(props.total_memory / (1024**3) - 1, 1)
        max_memory[gid] = f"{mem_gb}GiB"
    # Block all other GPUs by giving them 0
    for gid in range(n_available):
        if gid not in gpu_ids:
            max_memory[gid] = "0GiB"

    print(f"  Multi-GPU: distributing across GPUs {gpu_ids}")
    return f"cuda:{gpu_ids[0]}", {"auto": max_memory}


# ---------------------------------------------------------------------------
# Load functions — return ready-to-use model objects
# ---------------------------------------------------------------------------
def load_llm(
    name_or_id: str,
    device: str = "cuda",
    torch_dtype=torch.float16,
    max_new_tokens: int = 128,
    download: bool = False,
    gpu_ids: list[int] | None = None,
) -> tuple:
    """
    Load a causal LM. Returns (model, tokenizer).

    Args:
        name_or_id: Preset name ('tinyllama') or HF model id.
        device: 'cuda' or 'cpu' (ignored if gpu_ids is set).
        download: If True, download to MODELS_DIR first.
        gpu_ids: List of GPU indices. [2] = single GPU 2,
                 [0,1,2,3] = model parallelism across 4 GPUs.
    """
    from transformers import AutoTokenizer

    preset = _get_preset(name_or_id)
    model_id = preset["model_id"]

    if download:
        download_model(model_id, local_name=name_or_id if name_or_id in MODEL_PRESETS else None)

    path = _resolve_model_path(model_id, name_or_id if name_or_id in MODEL_PRESETS else None)

    # Detect OpenVINO models (by preset type or presence of openvino_model.xml)
    is_openvino = (
        preset.get("type") == "llm-openvino"
        or (Path(path).is_dir() and (Path(path) / "openvino_model.xml").exists())
    )

    eff_device, device_map = _resolve_device(device, gpu_ids)
    gpu_label = f"GPU {gpu_ids}" if gpu_ids else device
    print(f"  Loading LLM: {path} (device={gpu_label})")

    tokenizer = AutoTokenizer.from_pretrained(path, fix_mistral_regex=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_openvino:
        from optimum.intel import OVModelForCausalLM
        print(f"  Detected OpenVINO model format")
        model = OVModelForCausalLM.from_pretrained(path)
    elif isinstance(device_map, dict):
        from transformers import AutoModelForCausalLM
        # Multi-GPU: device_map="auto" with max_memory
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch_dtype,
            device_map="auto",
            max_memory=device_map["auto"],
            attn_implementation="eager",
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation="eager",
        )
    model.eval()
    print(f"  LLM ready on {gpu_label}")
    return model, tokenizer


def load_embedding(
    name_or_id: str = "minilm",
    device: str = "cuda",
    download: bool = False,
    gpu_ids: list[int] | None = None,
) -> object:
    """
    Load a sentence-transformers embedding model.
    Returns a LangChain HuggingFaceEmbeddings instance.

    Args:
        gpu_ids: If set, embedding runs on gpu_ids[0] (single GPU).
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    preset = _get_preset(name_or_id)
    model_id = preset["model_id"]

    if download:
        download_model(model_id, local_name=name_or_id if name_or_id in MODEL_PRESETS else None)

    path = _resolve_model_path(model_id, name_or_id if name_or_id in MODEL_PRESETS else None)

    # Embeddings always go on a single GPU
    if gpu_ids:
        embed_device = f"cuda:{gpu_ids[0]}"
    else:
        embed_device = device
    print(f"  Loading embedding: {path} (device={embed_device})")

    embeddings = HuggingFaceEmbeddings(
        model_name=path,
        model_kwargs={"device": embed_device},
    )
    print(f"  Embedding model ready")
    return embeddings


def load_clip(
    name_or_id: str = "clip-vit-large",
    device: str = "cuda",
    torch_dtype=torch.float16,
    download: bool = False,
    gpu_ids: list[int] | None = None,
) -> tuple:
    """
    Load a CLIP model for multimodal embedding. Returns (model, processor).

    The model produces aligned text and image embeddings in the same
    vector space, enabling cross-modal similarity search.

    Args:
        name_or_id: Preset name ('clip-vit-large') or HF model id.
        device: 'cuda' or 'cpu' (ignored if gpu_ids is set).
        download: If True, download to MODELS_DIR first.
        gpu_ids: List of GPU indices. Embeddings use gpu_ids[0].
    """
    from transformers import CLIPModel, CLIPProcessor

    preset = _get_preset(name_or_id)
    model_id = preset["model_id"]

    if download:
        download_model(model_id, local_name=name_or_id if name_or_id in MODEL_PRESETS else None)

    path = _resolve_model_path(model_id, name_or_id if name_or_id in MODEL_PRESETS else None)

    if gpu_ids:
        clip_device = f"cuda:{gpu_ids[0]}"
    else:
        clip_device = device
    print(f"  Loading CLIP: {path} (device={clip_device})")

    processor = CLIPProcessor.from_pretrained(path)
    model = CLIPModel.from_pretrained(path, torch_dtype=torch_dtype).to(clip_device)
    model.eval()
    print(f"  CLIP model ready ({clip_device})")
    return model, processor


def load_vlm(
    name_or_id: str = "llava-1.5-7b",
    device: str = "cuda",
    torch_dtype=torch.float16,
    download: bool = False,
    gpu_ids: list[int] | None = None,
) -> tuple:
    """
    Load a Vision-Language Model. Returns (model, processor).

    Args:
        gpu_ids: Same semantics as load_llm().
    """
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    preset = _get_preset(name_or_id)
    model_id = preset["model_id"]

    if download:
        download_model(model_id, local_name=name_or_id if name_or_id in MODEL_PRESETS else None)

    path = _resolve_model_path(model_id, name_or_id if name_or_id in MODEL_PRESETS else None)

    eff_device, device_map = _resolve_device(device, gpu_ids)
    gpu_label = f"GPU {gpu_ids}" if gpu_ids else device
    print(f"  Loading VLM: {path} (device={gpu_label})")

    processor = AutoProcessor.from_pretrained(path)
    if isinstance(device_map, dict):
        model = LlavaForConditionalGeneration.from_pretrained(
            path,
            torch_dtype=torch_dtype,
            device_map="auto",
            max_memory=device_map["auto"],
        )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
    model.eval()
    print(f"  VLM ready on {gpu_label}")
    return model, processor


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Model Loader — Download and manage HF models for benchmarking",
    )
    parser.add_argument("--list", action="store_true",
                        help="List all preset models")
    parser.add_argument("--download", action="store_true",
                        help="Download model to local storage (part3/models/)")
    parser.add_argument("--preset", default=None,
                        help="Preset name (e.g. 'tinyllama', 'minilm', 'all')")
    parser.add_argument("--model-id", default=None,
                        help="Custom HuggingFace model ID (e.g. 'org/model-name')")
    parser.add_argument("--verify", action="store_true",
                        help="Verify a model loads correctly on GPU")
    parser.add_argument("--gpu", default=None,
                        help="GPU index or comma-separated indices (e.g. '0' or '0,1,2,3')")
    args = parser.parse_args()

    # Parse --gpu into list of ints
    gpu_ids = None
    if args.gpu is not None:
        gpu_ids = [int(x.strip()) for x in args.gpu.split(",")]

    if args.list:
        print(f"\nAvailable model presets (download dir: {MODELS_DIR}):\n")
        print(f"  {'Name':<18s} {'Type':<12s} {'HuggingFace ID':<45s} Description")
        print("  " + "-" * 100)
        for name, info in MODEL_PRESETS.items():
            local_path = MODELS_DIR / name
            status = " [local]" if local_path.exists() and any(local_path.iterdir()) else ""
            print(f"  {name:<18s} {info['type']:<12s} "
                  f"{info['model_id']:<45s} {info['description']}{status}")
        print()
        return

    if args.download:
        if args.preset == "all":
            for name, info in MODEL_PRESETS.items():
                print(f"\n--- Downloading: {name} ---")
                download_model(info["model_id"], local_name=name)
        elif args.preset:
            if args.preset not in MODEL_PRESETS:
                print(f"Error: unknown preset '{args.preset}'. Use --list to see options.")
                sys.exit(1)
            info = MODEL_PRESETS[args.preset]
            download_model(info["model_id"], local_name=args.preset)
        elif args.model_id:
            download_model(args.model_id)
        else:
            print("Error: specify --preset or --model-id with --download")
            sys.exit(1)
        return

    if args.verify:
        name = args.preset or args.model_id
        if not name:
            print("Error: specify --preset or --model-id with --verify")
            sys.exit(1)
        preset = _get_preset(name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gpu_label = f"GPU {gpu_ids}" if gpu_ids else device
        print(f"\nVerifying {name} ({preset['type']}) on {gpu_label}...")
        if preset["type"] == "llm":
            model, tok = load_llm(name, device=device, gpu_ids=gpu_ids)
            # For multi-GPU, inputs go to the model's first device
            input_device = next(model.parameters()).device
            inputs = tok("Hello, world!", return_tensors="pt").to(input_device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=10)
            print(f"  Output: {tok.decode(out[0], skip_special_tokens=True)}")
        elif preset["type"] == "embedding":
            emb = load_embedding(name, device=device, gpu_ids=gpu_ids)
            result = emb.embed_query("test query")
            print(f"  Embedding dim: {len(result)}")
        elif preset["type"] == "vlm":
            model, proc = load_vlm(name, device=device, gpu_ids=gpu_ids)
            print(f"  VLM loaded successfully")
        print("  Verification passed.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
