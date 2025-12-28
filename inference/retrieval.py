"""
Candidate retrieval module for ontology-attribute alignment inference.

This file contains ONLY Stage-A retrieval (high recall), i.e. producing a shortlist
of candidate ontology class IRIs for each attribute string.

Retrieval sources implemented:
  1) Exact match (soft-normalized) via offline_bundle["label2classes"]
  2) Lexical retrieval via offline_bundle["inverted_index"] + offline_bundle["idf"]
     using the cross-encoder tokenizer for subword tokenization
  3) Semantic retrieval via offline_bundle["semantic_index"]["embeddings"]
     + cosine similarity against a bi-encoder query embedding.

Production constraints addressed:
  - No model/tokenizer is reloaded per attribute.
  - Bi-encoder is loaded once and cached in a BiEncoder object.
  - Lexical tokenizer is created once and reused.
  - Semantic class embeddings are assumed to be loaded once via load_offline_bundle(..., mmap=True).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from ontologies.offline_preprocessing import normalize_label

OfflineBundle = Dict[str, Any]


# -----------------------------
# Types
# -----------------------------

@dataclass(frozen=True)
class Candidate:
    """
    A candidate ontology class produced by retrieval.

    Attributes:
        iri: ontology class identifier
        score: retrieval score (source-dependent; not guaranteed comparable across sources)
        source: "exact" | "lexical" | "semantic" | "hybrid"
    """
    iri: str
    score: float
    source: str


# -----------------------------
# Exact match retrieval
# -----------------------------

def exact_match_candidates(attr_text: str, *, bundle: OfflineBundle) -> List[Candidate]:
    """
    Exact match lookup using label2classes with soft normalization.

    If normalize_label(attr_text) matches any normalized label/synonym form,
    returns all corresponding class IRIs.

    Returns:
        List[Candidate] with source="exact" and score=1.0
    """
    label2classes: Dict[str, set] = bundle["label2classes"]

    q = normalize_label(attr_text)
    if not q:
        return []

    iris = sorted(label2classes.get(q, set()))
    return [Candidate(iri=i, score=1.0, source="exact") for i in iris]


# -----------------------------
# Lexical retrieval (inverted index + IDF overlap)
# -----------------------------

def lexical_candidates(
    attr_text: str,
    *,
    bundle: OfflineBundle,
    tokenizer,  # cross-encoder tokenizer (must match offline tokenization)
    top_k: int = 50,
) -> List[Candidate]:
    """
    Retrieve candidates using lexical overlap with IDF weighting.

        score(c) = sum_{t in tokens(attr) ∩ T(c)} idf[t]

    Notes:
      - This is a high-recall heuristic retrieval score, not a calibrated probability.
      - Scores are not directly comparable to semantic cosine similarities.

    Args:
        attr_text: raw attribute text.
        bundle: offline bundle containing inverted_index and idf.
        tokenizer: tokenizer aligned with offline subword tokenization (cross-encoder tokenizer).
        top_k: maximum candidates to return.

    Returns:
        List[Candidate] sorted by descending lexical score.
    """
    inv_index: Dict[str, set] = bundle["inverted_index"]
    idf: Dict[str, float] = bundle["idf"]

    q = normalize_label(attr_text)
    if not q:
        return []

    q_tokens = tokenizer.tokenize(q)
    if not q_tokens:
        return []

    scores: Dict[str, float] = {}
    for t in q_tokens:
        cls_set = inv_index.get(t)
        if not cls_set:
            continue
        w = float(idf.get(t, 0.0))
        if w <= 0.0:
            continue
        for iri in cls_set:
            scores[iri] = scores.get(iri, 0.0) + w

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: int(top_k)]
    return [Candidate(iri=str(iri), score=float(s), source="lexical") for iri, s in ranked]


# -----------------------------
# Semantic retrieval (bi-encoder + cosine similarity)
# -----------------------------

def _mean_pool_last_hidden(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
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


class BiEncoder:
    """
    HuggingFace bi-encoder wrapper to encode texts into dense vectors.

    Key design goal: load tokenizer/model ONCE, then reuse encode() across many attributes.
    """

    def __init__(
        self,
        *,
        model_id: str,
        device: Optional[str] = None,
        max_length: int = 256,
        normalize: bool = True,
    ) -> None:
        self.model_id = str(model_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = int(max_length)
        self.normalize = bool(normalize)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
        """
        Encode a list of texts into a float32 numpy array [B, D].

        Uses mean pooling over last_hidden_state and optional L2 normalization.

        Args:
            texts: list of input strings
            batch_size: encoding batch size

        Returns:
            np.ndarray float32, shape [len(texts), D]
        """
        all_embs: List[torch.Tensor] = []

        for start in range(0, len(texts), int(batch_size)):
            batch = ["" if t is None else str(t) for t in texts[start : start + int(batch_size)]]

            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            out = self.model(**enc)
            pooled = _mean_pool_last_hidden(out.last_hidden_state, enc["attention_mask"])

            if self.normalize:
                pooled = F.normalize(pooled, p=2, dim=1)

            all_embs.append(pooled.detach().cpu())

        return torch.cat(all_embs, dim=0).numpy().astype("float32", copy=False)


def _semantic_similarity_topk(
    *,
    class_embs: np.ndarray,
    iris: Sequence[str],
    q_emb: np.ndarray,
    top_k: int,
    normalized: bool,
) -> List[Candidate]:
    """
    Compute cosine similarity top-k between a single query embedding and all class embeddings.

    If embeddings are L2-normalized, cosine similarity reduces to dot product.
    """
    E = np.asarray(class_embs)  # memmap-friendly, [N, D]

    if E.ndim != 2:
        raise ValueError(f"class_embs must be 2D [N, D], got shape={E.shape}")

    if q_emb.ndim != 1:
        raise ValueError(f"q_emb must be 1D [D], got shape={q_emb.shape}")

    if normalized:
        sims = E @ q_emb.astype(E.dtype, copy=False)  # [N]
    else:
        qn = q_emb / (np.linalg.norm(q_emb) + 1e-12)
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        sims = En @ qn.astype(En.dtype, copy=False)

    k = min(int(top_k), sims.shape[0])
    if k <= 0:
        return []

    idx = np.argpartition(-sims, kth=k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]

    return [Candidate(iri=str(iris[i]), score=float(sims[i]), source="semantic") for i in idx]


def semantic_candidates(
    attr_text: str,
    *,
    bundle: OfflineBundle,
    encoder: BiEncoder,
    top_k: int = 50,
) -> List[Candidate]:
    """
    Retrieve candidates using semantic similarity against offline class embeddings.

    Requires that the offline bundle already contains semantic embeddings, i.e.
    bundle["semantic_index"]["embeddings"] is loaded.

    Important:
      - This function does NOT reload any model. The provided `encoder` must be cached.

    Args:
        attr_text: attribute raw string
        bundle: offline bundle with semantic_index
        encoder: cached BiEncoder instance
        top_k: number of semantic candidates to return
        batch_size: encoding batch size (only relevant if you later batch queries)

    Returns:
        List[Candidate] sorted by descending cosine similarity.
    """
    sem = bundle.get("semantic_index")
    if not isinstance(sem, dict):
        return []

    class_embs = sem.get("embeddings")
    iris = sem.get("iris")
    normalized = bool(sem.get("normalized", True))

    if class_embs is None or iris is None:
        return []

    q = str(attr_text or "").strip()
    if not q:
        return []

    q_emb = encoder.encode([q], batch_size=1)[0]  # [D]
    return _semantic_similarity_topk(
        class_embs=np.asarray(class_embs),
        iris=iris,
        q_emb=q_emb,
        top_k=int(top_k),
        normalized=normalized and encoder.normalize,
    )

def semantic_candidates_batch(
    attr_texts: Sequence[str],
    *,
    bundle: OfflineBundle,
    encoder: BiEncoder,
    top_k: int = 50,
    batch_size: int = 64,
) -> List[List[Candidate]]:
    """
    Semantic retrieval for many attributes (batch queries).

    It encodes queries in batches and then performs top-k similarity per query.

    Returns:
        A list of candidate lists, same length/order as attr_texts.
    """
    sem = bundle.get("semantic_index")
    if not isinstance(sem, dict):
        return [[] for _ in attr_texts]

    class_embs = sem.get("embeddings")
    iris = sem.get("iris")
    bundle_normalized = bool(sem.get("normalized", True))

    if class_embs is None or iris is None:
        return [[] for _ in attr_texts]

    texts = ["" if t is None else str(t).strip() for t in attr_texts]
    out: List[List[Candidate]] = [[] for _ in texts]

    idx = [i for i, s in enumerate(texts) if s]
    if not idx:
        return out

    q_embs = encoder.encode([texts[i] for i in idx], batch_size=int(batch_size))  # [M, D]

    E = np.asarray(class_embs)  # memmap-friendly
    normalized = bundle_normalized and encoder.normalize

    for j, i in enumerate(idx):
        out[i] = _semantic_similarity_topk(
            class_embs=E,
            iris=iris,
            q_emb=q_embs[j],
            top_k=int(top_k),
            normalized=normalized,
        )
    return out


# -----------------------------
# Candidate merging (hybrid)
# -----------------------------

def merge_candidates(
    lexical: List[Candidate],
    semantic: List[Candidate],
    *,
    top_k: int,
) -> List[Candidate]:
    """
    Strategy:
      - take candidates in order from the first list (priority),
      - then append from the second list,
      - deduplicate by IRI,
      - cut to top_k.
    """
    top_k = max(int(top_k), 0)
    if top_k == 0:
        return []

    first = lexical
    second = semantic

    seen: set[str] = set()
    merged: List[Candidate] = []

    def _push(c: Candidate) -> None:
        iri = str(c.iri)
        if iri in seen:
            return
        seen.add(iri)
        merged.append(Candidate(iri=iri, score=float(c.score), source=str(c.source)))

    for c in first:
        if len(merged) >= top_k:
            break
        _push(c)

    if len(merged) < top_k:
        for c in second:
            if len(merged) >= top_k:
                break
            _push(c)

    return merged


def _split_budget(total_k: int, ratio_semantic: float) -> Tuple[int, int]:
    """
    Split total_k into (k_lexical, k_semantic) given a semantic ratio.
    """
    total_k = max(int(total_k), 0)
    if total_k == 0:
        return 0, 0
    r = float(ratio_semantic)
    r = min(max(r, 0.0), 1.0)
    k_sem = int(round(total_k * r))
    k_sem = min(max(k_sem, 0), total_k)
    k_lex = total_k - k_sem
    return k_lex, k_sem


# -----------------------------
# High-level retriever (production-friendly)
# -----------------------------


class CandidateRetriever:
    """
    Loads and caches:
      - the tokenizer used for lexical retrieval (cross-encoder tokenizer)
      - the bi-encoder (once) for semantic retrieval, if available

    Retrieval modes:
      - mode="lexical":
          exact -> lexical; if lexical is empty, fallback to semantic only
      - mode="hybrid":
          exact -> ALWAYS combine lexical + semantic (no fallback),
          with a fixed budget split (e.g., 50% semantic / 50% lexical)
    """

    def __init__(
        self,
        *,
        bundle: Dict[str, Any],
        cross_tokenizer_name: str,
        semantic_device: Optional[str] = None,
        semantic_batch_size: int = 64,
    ) -> None:
        self.bundle: Dict[str, Any] = bundle

        # Cache tokenizer once (used for lexical retrieval)
        self.lexical_tokenizer = AutoTokenizer.from_pretrained(str(cross_tokenizer_name))

        # Cache semantic encoder once (if semantic index exists)
        self.semantic_encoder: Optional[BiEncoder] = None
        sem = bundle.get("semantic_index")
        if isinstance(sem, dict):
            model_id = sem.get("model_id")
            if model_id is not None:
                self.semantic_encoder = BiEncoder(
                    model_id=str(model_id),
                    device=semantic_device,
                    max_length=int(sem.get("max_length", 256)),
                    normalize=bool(sem.get("normalized", True)),
                )

        self.semantic_batch_size = int(semantic_batch_size)

    # -----------------------------
    # Single attribute retrieval
    # -----------------------------

    def retrieve_one(
        self,
        attr_text: str,
        *,
        mode: str,
        lexical_top_k: int = 100,
        semantic_top_k: int = 100,
        merged_top_k: int = 150,
        hybrid_ratio_semantic: float = 0.5,
    ) -> Tuple[List[Candidate], str]:
        """
        Retrieve candidates for ONE attribute.

        Returns:
            (candidates, retrieval_source)
        """
        mode = str(mode).lower().strip()
        if mode not in {"lexical", "hybrid"}:
            raise ValueError(f"Unsupported retrieval mode: {mode}")

        # Pre-compute hybrid budget so lexical retrieval is asked for enough items
        k_lex = 0
        k_sem = 0
        lexical_top_k_eff = int(lexical_top_k)
        semantic_top_k_eff = int(semantic_top_k)
        if mode == "hybrid":
            k_lex, k_sem = _split_budget(int(merged_top_k), float(hybrid_ratio_semantic))
            lexical_top_k_eff = max(lexical_top_k_eff, k_lex)
            semantic_top_k_eff = max(semantic_top_k_eff, k_sem)

        # 1) Exact match always first
        exact = exact_match_candidates(attr_text, bundle=self.bundle)
        if exact:
            return exact, "exact"

        # 2) Lexical retrieval
        lex = lexical_candidates(
            attr_text,
            bundle=self.bundle,
            tokenizer=self.lexical_tokenizer,
            top_k=lexical_top_k_eff,
        )

        # mode == lexical: lexical first, semantic only as fallback
        if mode == "lexical":
            if lex:
                return lex, "lexical"

            if self.semantic_encoder is None:
                return [], "none"

            sem = semantic_candidates(
                attr_text,
                bundle=self.bundle,
                encoder=self.semantic_encoder,
                top_k=int(semantic_top_k_eff),
            )
            return sem, ("semantic" if sem else "none")

        # mode == hybrid: ALWAYS combine lexical + semantic (no fallback)
        if self.semantic_encoder is None:
            # cannot run semantic => return lexical only
            return lex, ("lexical" if lex else "none")

        lex_cut = lex[:k_lex] if k_lex > 0 else []

        sem = semantic_candidates(
            attr_text,
            bundle=self.bundle,
            encoder=self.semantic_encoder,
            top_k=semantic_top_k_eff,
        )
        sem_cut = sem[:k_sem] if k_sem > 0 else []

        merged = merge_candidates(
            lex_cut,
            sem_cut,
            top_k=int(merged_top_k),
        )
        return merged, ("hybrid" if merged else "none")

    # -----------------------------
    # Batch retrieval (production)
    # -----------------------------

    def retrieve_batch(
        self,
        attr_texts: Sequence[str],
        *,
        mode: str,
        lexical_top_k: int = 100,
        semantic_top_k: int = 100,
        merged_top_k: int = 150,
        hybrid_ratio_semantic: float = 0.5,
        semantic_batch_size: Optional[int] = None,
    ) -> Tuple[List[List[Candidate]], List[str]]:
        """
        Batch retrieval across many attributes.

        This is the production-friendly path:
          - exact + lexical are computed per attribute (cheap)
          - semantic queries are encoded in batch (fast) when needed

        Returns:
            (candidates_per_attr, retrieval_source_per_attr)
        """
        mode = str(mode).lower().strip()
        if mode not in {"lexical", "hybrid"}:
            raise ValueError(f"Unsupported retrieval mode: {mode}")

        # Pre-compute hybrid budget to request enough lexical/semantic items
        k_lex = 0
        k_sem = 0
        lexical_top_k_eff = int(lexical_top_k)
        semantic_top_k_eff = int(semantic_top_k)
        if mode == "hybrid":
            k_lex, k_sem = _split_budget(int(merged_top_k), float(hybrid_ratio_semantic))
            lexical_top_k_eff = max(lexical_top_k_eff, k_lex)
            semantic_top_k_eff = max(semantic_top_k_eff, k_sem)

        n = len(attr_texts)
        results: List[List[Candidate]] = [[] for _ in range(n)]
        sources: List[str] = ["none" for _ in range(n)]

        # Indices that require semantic retrieval
        need_semantic_idx: List[int] = []

        # 1) Exact + lexical pass
        for i, t in enumerate(attr_texts):
            txt = "" if t is None else str(t).strip()

            exact = exact_match_candidates(txt, bundle=self.bundle)
            if exact:
                results[i] = exact
                sources[i] = "exact"
                continue

            lex = lexical_candidates(
                txt,
                bundle=self.bundle,
                tokenizer=self.lexical_tokenizer,
                top_k=lexical_top_k_eff,
            )

            if mode == "lexical":
                if lex:
                    results[i] = lex
                    sources[i] = "lexical"
                else:
                    # fallback to semantic if possible
                    need_semantic_idx.append(i)
            else:
                # hybrid: always attempt semantic too (if possible)
                results[i] = lex
                sources[i] = "hybrid"
                need_semantic_idx.append(i)

        # If no semantic encoder, downgrade hybrid to lexical-only and/or leave lexical fallbacks empty
        if self.semantic_encoder is None:
            if mode == "hybrid":
                for i in need_semantic_idx:
                    sources[i] = "lexical" if results[i] else "none"
            return results, sources

        # 2) Semantic batch for needed indices
        if need_semantic_idx:
            bs = int(semantic_batch_size) if semantic_batch_size is not None else self.semantic_batch_size
            sem_texts = ["" if attr_texts[i] is None else str(attr_texts[i]) for i in need_semantic_idx]

            sem_lists = semantic_candidates_batch(
                sem_texts,
                bundle=self.bundle,
                encoder=self.semantic_encoder,
                top_k=semantic_top_k_eff,
                batch_size=int(bs),
            )

            if mode == "lexical":
                # fill semantic fallback when lexical is empty
                for local_j, i in enumerate(need_semantic_idx):
                    sem = sem_lists[local_j]
                    results[i] = sem
                    sources[i] = "semantic" if sem else "none"
            else:
                # hybrid: simple union + dedup (no cross-source score comparisons)
                for local_j, i in enumerate(need_semantic_idx):
                    lex = results[i][:k_lex] if k_lex > 0 else []
                    sem = sem_lists[local_j][:k_sem] if k_sem > 0 else []

                    merged = merge_candidates(
                        lex,
                        sem,
                        top_k=int(merged_top_k),
                    )

                    results[i] = merged
                    sources[i] = "hybrid" if merged else ("lexical" if lex else "none")

        return results, sources
