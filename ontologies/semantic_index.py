"""
Semantic embedding index builder for ontology classes (offline stage).

This module is intentionally separated from offline_preprocessing.py so that:
- lexical bundle components remain lightweight,
- semantic embeddings (torch/transformers) are isolated and explicit dependencies.

Contract (per your pipeline):
- The unified view DataFrame used here must contain at least:
    ['iri', 'label', 'text']
  (and may also contain ['local_name'] etc.)

The semantic index structure (in-memory form):
    semantic_index = {
        "embeddings": np.ndarray (N x D, float32),
        "iris": List[str],
        "model_id": str,
        "normalized": bool,
        "max_length": int,
        "embedding_dim": int,
    }

Saving/loading options:
- embeddings saved to a separate .npy file
- semantic_index metadata stores embeddings_path (relative to bundle dir)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


# -----------------------------
# Configuration / types
# -----------------------------

@dataclass(frozen=True)
class SemanticIndexConfig:
    model_id: str
    batch_size: int = 64
    max_length: int = 256
    normalize: bool = True
    device: Optional[str] = None
    seed: int = 42


SemanticIndex = Dict[str, Any]


# -----------------------------
# Internal helpers
# -----------------------------

def _set_determinism(seed: int) -> None:
    """
    Practical determinism (best-effort):
    - fixes RNG seeds
    - disables cuDNN benchmark
    - enables deterministic cuDNN where possible
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _mean_pool_last_hidden(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling over sequence length with attention_mask.

    last_hidden_state: [B, L, H]
    attention_mask:    [B, L]
    returns:           [B, H]
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, L, 1]
    summed = (last_hidden_state * mask).sum(dim=1)                  # [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-9)                        # [B, 1]
    return summed / counts


def _validate_unified_for_semantic(df) -> None:
    required = {"iri", "label", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Unified view DataFrame missing required columns for semantic index: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )


# -----------------------------
# Public API
# -----------------------------

def build_semantic_index_from_unified(
    df_uni,
    config: SemanticIndexConfig,
) -> SemanticIndex:
    """
    Build semantic embeddings for ontology classes using a HuggingFace bi-encoder.

    Args:
        df_uni: unified view DataFrame. Must contain at least ['iri','label','text'].
        config: SemanticIndexConfig(model_id=..., batch_size=..., max_length=..., ...)

    Returns:
        semantic_index dict with in-memory embeddings (np.ndarray float32).
    """
    _validate_unified_for_semantic(df_uni)
    _set_determinism(config.seed)

    device = _resolve_device(config.device)

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModel.from_pretrained(config.model_id).to(device)
    model.eval()

    iris: List[str] = df_uni["iri"].astype(str).tolist()
    texts: List[str] = df_uni["text"].fillna("").astype(str).tolist()

    all_embs: List[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(texts), config.batch_size):
            batch_texts = texts[start : start + config.batch_size]

            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            out = model(**enc)
            pooled = _mean_pool_last_hidden(out.last_hidden_state, enc["attention_mask"])

            if config.normalize:
                pooled = F.normalize(pooled, p=2, dim=1)

            all_embs.append(pooled.detach().cpu())

    embs = torch.cat(all_embs, dim=0).numpy().astype("float32", copy=False)

    return {
        "embeddings": embs,
        "iris": iris,
        "model_id": config.model_id,
        "normalized": bool(config.normalize),
        "max_length": int(config.max_length),
        "embedding_dim": int(embs.shape[1]),
    }


def save_semantic_embeddings(
    semantic_index: SemanticIndex,
    *,
    bundle_path: str | Path,
    filename_suffix: str = ".semantic_embeddings.npy",
) -> SemanticIndex:
    """
    Saving:
    - writes embeddings to a separate .npy file next to the bundle pickle
    - returns a *metadata-only* semantic_index (embeddings removed, embeddings_path added)

    Args:
        semantic_index: dict returned by build_semantic_index_from_unified (contains 'embeddings')
        bundle_path: path of the pickle bundle this semantic index belongs to
        filename_suffix: suffix for the embeddings file

    Returns:
        semantic_index_meta: dict without 'embeddings', with 'embeddings_path' relative to bundle dir
    """
    if "embeddings" not in semantic_index or semantic_index["embeddings"] is None:
        raise ValueError("semantic_index must contain in-memory 'embeddings' to be saved (Option B).")

    embs = semantic_index["embeddings"]
    if not isinstance(embs, np.ndarray):
        raise TypeError("semantic_index['embeddings'] must be a numpy.ndarray.")

    bundle_path = Path(bundle_path)
    bundle_dir = bundle_path.parent
    bundle_dir.mkdir(parents=True, exist_ok=True)

    npy_name = bundle_path.with_suffix("").name + filename_suffix
    npy_path = bundle_dir / npy_name

    np.save(npy_path, embs)

    meta = dict(semantic_index)
    meta.pop("embeddings", None)
    meta["embeddings_path"] = npy_path.name  # store relative path for portability
    return meta


def load_semantic_embeddings(
    semantic_index_meta: SemanticIndex,
    *,
    bundle_path: str | Path,
    mmap: bool = True,
) -> SemanticIndex:
    """
    Loads embeddings back into semantic_index from embeddings_path.

    Args:
        semantic_index_meta: dict containing embeddings_path (relative to bundle dir)
        bundle_path: path of the pickle bundle
        mmap: if True, loads with np.load(..., mmap_mode="r") to avoid full RAM load

    Returns:
        semantic_index: dict with 'embeddings' inserted
    """
    if "embeddings_path" not in semantic_index_meta:
        raise ValueError("semantic_index_meta must contain 'embeddings_path' for Option B loading.")

    bundle_path = Path(bundle_path)
    npy_path = bundle_path.parent / semantic_index_meta["embeddings_path"]

    out = dict(semantic_index_meta)
    out["embeddings"] = np.load(npy_path, mmap_mode="r" if mmap else None)
    return out


def semantic_index_has_embeddings(semantic_index: SemanticIndex) -> bool:
    return isinstance(semantic_index, dict) and ("embeddings" in semantic_index) and (semantic_index["embeddings"] is not None)