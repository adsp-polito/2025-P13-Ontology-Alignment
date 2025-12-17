"""
End-to-end ontology-attribute alignment inference pipeline.

Given:
  - an offline bundle (pickle) produced by offline_preprocessing.py
    (includes exact-match structures + lexical inverted index + semantic index metadata,
     and optionally semantic embeddings loaded via load_offline_bundle Option B),
  - an ontology CSV exported by build_ontology_bundle.py
    (must contain at least: iri, text; and typically: local_name, label, synonyms, text),
  - a CSV of study attributes (or any list of strings),

this script runs a two-stage inference pipeline:

  Stage A) Candidate retrieval (high recall):
    1) Exact match via label2classes (soft-normalized)
    2) Lexical retrieval via inverted index + IDF overlap (subword tokens)
    3) Semantic retrieval via bi-encoder embeddings (cosine similarity)
       - can be used as fallback or merged (hybrid)

  Stage B) Candidate scoring (high precision):
    4) Cross-encoder scoring on (attribute_text, class_text) pairs
    5) Select top-1 prediction (and optionally keep top-N)

Output:
  - a CSV file with predictions for each attribute row:
      attribute_id, attribute_text, predicted_iri, predicted_score, retrieval_source, etc.
  - optionally includes top-N candidates per attribute.

Important constraints:
  - This script does NOT train anything.
  - This script does NOT modify the offline bundle.
  - Cross-encoder scoring uses a HuggingFace sequence classification model.

Assumptions:
  - Offline bundle is built using the same cross-tokenizer used here for lexical retrieval.
  - Semantic index in the bundle was built from df_uni["text"] (RICH_TEXT).

  How to run it (example):
    python inference.py \
  --bundle data/offline_bundle.pkl \
  --ontology-csv data/internal_ontology.csv \
  --input-csv data/study_attributes.csv \
  --out-csv data/predictions.csv \
  --attr-col attribute \
  --cross-encoder-model-id <YOUR_CROSS_ENCODER_MODEL_ID> \
  --retrieval-mode hybrid \
  --cross-top-k 20
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from ontologies.offline_preprocessing import load_offline_bundle, normalize_label


# -----------------------------
# Types
# -----------------------------

OfflineBundle = Dict[str, Any]


@dataclass(frozen=True)
class Candidate:
    """A candidate ontology class produced by retrieval."""
    iri: str
    score: float
    source: str  # "exact" | "lexical" | "semantic" | "hybrid"


# -----------------------------
# Helpers: ontology text lookup
# -----------------------------

def load_ontology_text_map(ontology_csv: str | Path) -> Dict[str, str]:
    """
    Load ontology CSV and build a mapping: iri -> text (RICH_TEXT).

    The ontology CSV is produced by the framework and must include at least:
      - iri
      - text

    Returns:
        dict mapping iri to text.
    """
    df = pd.read_csv(Path(ontology_csv))
    required = {"iri", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Ontology CSV missing required columns: {sorted(missing)}. Found: {list(df.columns)}"
        )

    iri2text: Dict[str, str] = {}
    for row in df.itertuples(index=False):
        iri = getattr(row, "iri")
        txt = getattr(row, "text", "")
        if pd.isna(txt):
            txt = ""
        iri2text[str(iri)] = str(txt)

    return iri2text


# -----------------------------
# Exact match retrieval
# -----------------------------

def exact_match_candidates(attr_text: str, *, bundle: OfflineBundle) -> List[Candidate]:
    """
    Exact match lookup using label2classes with soft normalization.

    If normalized attribute text matches any normalized label/synonym form,
    returns all corresponding classes.

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
    tokenizer,  # cross-encoder tokenizer (same as offline tokenization for T(c))
    top_k: int = 50,
) -> List[Candidate]:
    """
    Retrieve candidates using lexical overlap with IDF weighting.

    score(c) = sum_{t in tokens(attr) ∩ T(c)} idf[t]

    Args:
        attr_text: raw attribute text.
        bundle: offline bundle containing inverted_index and idf.
        tokenizer: tokenizer aligned with offline subword tokenization.
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

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [Candidate(iri=iri, score=float(s), source="lexical") for iri, s in ranked]


# -----------------------------
# Semantic retrieval (bi-encoder + cosine similarity)
# -----------------------------

def _mean_pool_last_hidden(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling over sequence length with attention_mask.

    last_hidden_state: [B, L, H]
    attention_mask:    [B, L]
    returns:           [B, H]
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _encode_texts_biencoder(
    texts: Sequence[str],
    *,
    model_id: str,
    batch_size: int = 64,
    max_length: int = 256,
    normalize: bool = True,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Encode texts with a HuggingFace bi-encoder using mean pooling.

    Returns:
        float32 numpy array of shape [len(texts), D]
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModel.from_pretrained(model_id).to(dev)
    mdl.eval()

    all_embs: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = ["" if t is None else str(t) for t in texts[start:start + batch_size]]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(dev) for k, v in enc.items()}

            out = mdl(**enc)
            pooled = _mean_pool_last_hidden(out.last_hidden_state, enc["attention_mask"])
            if normalize:
                pooled = F.normalize(pooled, p=2, dim=1)
            all_embs.append(pooled.detach().cpu())

    return torch.cat(all_embs, dim=0).numpy().astype("float32", copy=False)


def semantic_candidates(
    attr_text: str,
    *,
    bundle: OfflineBundle,
    top_k: int = 50,
    device: Optional[str] = None,
) -> List[Candidate]:
    """
    Retrieve candidates using semantic similarity against offline class embeddings.

    Requires that load_offline_bundle(...) has already loaded semantic embeddings (Option B).

    Returns:
        List[Candidate] sorted by descending cosine similarity.
    """
    sem = bundle.get("semantic_index")
    if not isinstance(sem, dict):
        return []

    class_embs = sem.get("embeddings")
    iris = sem.get("iris")
    model_id = sem.get("model_id")
    normalized = bool(sem.get("normalized", True))
    max_length = int(sem.get("max_length", 256))

    if class_embs is None or iris is None or model_id is None:
        return []

    q = str(attr_text or "").strip()
    if not q:
        return []

    q_emb = _encode_texts_biencoder(
        [q],
        model_id=str(model_id),
        batch_size=1,
        max_length=max_length,
        normalize=normalized,
        device=device,
    )[0]

    E = np.asarray(class_embs)  # memmap-friendly [N, D]

    if normalized:
        sims = E @ q_emb.astype(E.dtype, copy=False)
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


# -----------------------------
# Candidate merging (hybrid)
# -----------------------------

def merge_candidates(
    lists: Sequence[List[Candidate]],
    *,
    top_k: int,
) -> List[Candidate]:
    """
    Merge multiple candidate lists by IRI, keeping the maximum score among sources.

    This is useful for hybrid retrieval (lexical + semantic).
    The returned candidates are sorted by score descending.

    Notes:
      - Scores from different retrieval sources are not strictly comparable
        (lexical score is IDF overlap, semantic score is cosine similarity).
      - This merge is best used to increase recall. The cross-encoder will
        provide final ranking.

    Args:
        lists: sequence of candidate lists.
        top_k: maximum candidates after merge.

    Returns:
        merged candidates.
    """
    best: Dict[str, Candidate] = {}

    for cand_list in lists:
        for c in cand_list:
            prev = best.get(c.iri)
            if prev is None or c.score > prev.score:
                best[c.iri] = c

    merged = sorted(best.values(), key=lambda x: x.score, reverse=True)
    return merged[:top_k]


# -----------------------------
# Cross-encoder scoring
# -----------------------------

def _scores_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert model logits to a single probability-like score per example.

    Handles common cases:
      - shape [B, 1]: applies sigmoid
      - shape [B, 2]: applies softmax and takes positive class prob
      - shape [B, K>2]: applies softmax and takes max prob (fallback)

    Returns:
        scores tensor shape [B]
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape [B, C], got {tuple(logits.shape)}")

    if logits.size(1) == 1:
        return torch.sigmoid(logits[:, 0])
    if logits.size(1) == 2:
        probs = torch.softmax(logits, dim=1)
        return probs[:, 1]
    probs = torch.softmax(logits, dim=1)
    return probs.max(dim=1).values


def score_with_cross_encoder(
    attr_text: str,
    candidates: Sequence[Candidate],
    *,
    iri2text: Dict[str, str],
    model_id: str,
    batch_size: int = 32,
    max_length: int = 256,
    device: Optional[str] = None,
) -> List[Tuple[Candidate, float]]:
    """
    Score (attribute, class_text) pairs with a cross-encoder.

    Args:
        attr_text: the study attribute string.
        candidates: candidate IRIs from retrieval.
        iri2text: mapping iri -> class RICH_TEXT (from ontology CSV).
        model_id: HF cross-encoder model id (sequence classification).
        batch_size: scoring batch size.
        max_length: tokenizer truncation length.
        device: "cuda" | "cpu" | None auto.

    Returns:
        list of (candidate, score) sorted by descending score.
        score is a probability-like value in [0,1] for typical binary models.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id).to(dev)
    mdl.eval()

    # Build pairs
    pair_a: List[str] = []
    pair_b: List[str] = []
    kept: List[Candidate] = []

    a = str(attr_text or "").strip()
    if not a:
        return []

    for c in candidates:
        txt = iri2text.get(c.iri)
        if txt is None:
            continue  # candidate not in ontology CSV mapping
        pair_a.append(a)
        pair_b.append(str(txt))
        kept.append(c)

    if not kept:
        return []

    scores_all: List[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(kept), batch_size):
            a_batch = pair_a[start:start + batch_size]
            b_batch = pair_b[start:start + batch_size]

            enc = tok(
                a_batch,
                b_batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(dev) for k, v in enc.items()}

            out = mdl(**enc)
            logits = out.logits
            batch_scores = _scores_from_logits(logits).detach().cpu()
            scores_all.append(batch_scores)

    scores = torch.cat(scores_all, dim=0).numpy().astype("float32", copy=False)

    ranked = sorted(zip(kept, scores), key=lambda x: float(x[1]), reverse=True)
    return [(c, float(s)) for c, s in ranked]


# -----------------------------
# Full pipeline per attribute
# -----------------------------

def infer_one_attribute(
    attr_text: str,
    *,
    bundle: OfflineBundle,
    iri2text: Dict[str, str],
    cross_tokenizer_name: str,
    cross_encoder_model_id: str,
    retrieval_mode: str = "fallback",  # "fallback" | "hybrid"
    lexical_top_k: int = 100,
    semantic_top_k: int = 100,
    merged_top_k: int = 150,
    cross_top_k: int = 10,
    device: Optional[str] = None,
    cross_batch_size: int = 32,
    cross_max_length: int = 256,
) -> Dict[str, Any]:
    """
    Run the complete inference pipeline for a single attribute.

    retrieval_mode:
      - "fallback": exact -> lexical; if empty then semantic
      - "hybrid": exact -> merge(lexical, semantic)

    Returns:
        dict with prediction + top candidates metadata.
    """
    # 1) Exact match
    exact = exact_match_candidates(attr_text, bundle=bundle)
    retrieval_source = "exact"
    retrieved: List[Candidate] = exact

    if not retrieved:
        # Prepare cross tokenizer once for lexical retrieval
        cross_tok = AutoTokenizer.from_pretrained(cross_tokenizer_name)

        lex = lexical_candidates(
            attr_text,
            bundle=bundle,
            tokenizer=cross_tok,
            top_k=lexical_top_k,
        )

        if retrieval_mode == "hybrid":
            sem = semantic_candidates(attr_text, bundle=bundle, top_k=semantic_top_k, device=device)
            retrieved = merge_candidates([lex, sem], top_k=merged_top_k)
            retrieval_source = "hybrid"
        else:
            # fallback
            if lex:
                retrieved = lex
                retrieval_source = "lexical"
            else:
                sem = semantic_candidates(attr_text, bundle=bundle, top_k=semantic_top_k, device=device)
                retrieved = sem
                retrieval_source = "semantic" if sem else "none"

    # 2) Cross-encoder scoring on top retrieved candidates
    retrieved_for_scoring = retrieved[: max(int(cross_top_k), 1)]

    scored = score_with_cross_encoder(
        attr_text,
        retrieved_for_scoring,
        iri2text=iri2text,
        model_id=cross_encoder_model_id,
        batch_size=cross_batch_size,
        max_length=cross_max_length,
        device=device,
    )

    if not scored:
        return {
            "attribute_text": attr_text,
            "retrieval_source": retrieval_source,
            "predicted_iri": None,
            "predicted_score": None,
            "num_retrieved": len(retrieved),
            "num_scored": 0,
        }

    best_cand, best_score = scored[0]

    return {
        "attribute_text": attr_text,
        "retrieval_source": retrieval_source,
        "predicted_iri": best_cand.iri,
        "predicted_score": float(best_score),
        "num_retrieved": len(retrieved),
        "num_scored": len(scored),
    }


# -----------------------------
# Batch inference (CSV -> CSV)
# -----------------------------

def run_inference_csv(
    *,
    bundle_path: str | Path,
    ontology_csv: str | Path,
    input_csv: str | Path,
    out_csv: str | Path,
    attr_col: str = "attribute",
    id_col: Optional[str] = None,
    cross_tokenizer_name: str = "dmis-lab/biobert-base-cased-v1.1",
    cross_encoder_model_id: str = "",
    retrieval_mode: str = "fallback",
    lexical_top_k: int = 100,
    semantic_top_k: int = 100,
    merged_top_k: int = 150,
    cross_top_k: int = 10,
    device: Optional[str] = None,
    cross_batch_size: int = 32,
    cross_max_length: int = 256,
) -> None:
    """
    Run end-to-end inference on a CSV of attributes and write predictions to CSV.

    Args:
        bundle_path: offline_bundle.pkl
        ontology_csv: exported ontology CSV with iri + text
        input_csv: CSV containing attributes to align
        out_csv: output predictions CSV
        attr_col: name of the column containing attribute text
        id_col: optional identifier column to carry through
        cross_tokenizer_name: tokenizer used for lexical retrieval (must match offline)
        cross_encoder_model_id: HF model id for cross-encoder scoring (required)
        retrieval_mode: "fallback" or "hybrid"
        lexical_top_k, semantic_top_k, merged_top_k: retrieval sizes
        cross_top_k: number of retrieved candidates to send to cross-encoder
        device: cpu/cuda/None
        cross_batch_size, cross_max_length: cross-encoder scoring params
    """
    if not cross_encoder_model_id:
        raise ValueError("cross_encoder_model_id is required for a complete inference pipeline.")

    bundle = load_offline_bundle(Path(bundle_path), load_semantic_embeddings=True, mmap=True)
    iri2text = load_ontology_text_map(Path(ontology_csv))

    df_in = pd.read_csv(Path(input_csv))
    if attr_col not in df_in.columns:
        raise ValueError(f"Input CSV missing attr_col='{attr_col}'. Found columns: {list(df_in.columns)}")

    out_rows: List[Dict[str, Any]] = []

    for idx, row in df_in.iterrows():
        attr_text = row[attr_col]
        attr_text = "" if pd.isna(attr_text) else str(attr_text)

        res = infer_one_attribute(
            attr_text,
            bundle=bundle,
            iri2text=iri2text,
            cross_tokenizer_name=cross_tokenizer_name,
            cross_encoder_model_id=cross_encoder_model_id,
            retrieval_mode=retrieval_mode,
            lexical_top_k=lexical_top_k,
            semantic_top_k=semantic_top_k,
            merged_top_k=merged_top_k,
            cross_top_k=cross_top_k,
            device=device,
            cross_batch_size=cross_batch_size,
            cross_max_length=cross_max_length,
        )

        out_row: Dict[str, Any] = {}
        if id_col is not None:
            if id_col not in df_in.columns:
                raise ValueError(f"id_col='{id_col}' not found in input CSV columns: {list(df_in.columns)}")
            out_row[id_col] = row[id_col]
        else:
            out_row["row_id"] = int(idx)

        out_row.update(res)
        out_rows.append(out_row)

    df_out = pd.DataFrame(out_rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Complete ontology alignment inference (CSV -> CSV).")

    p.add_argument("--bundle", required=True, help="Path to offline_bundle.pkl")
    p.add_argument("--ontology-csv", required=True, help="Path to ontology export CSV (iri,text,...)")
    p.add_argument("--input-csv", required=True, help="Path to input CSV containing study attributes")
    p.add_argument("--out-csv", required=True, help="Path to output predictions CSV")

    p.add_argument("--attr-col", default="attribute", help="Column name for attribute text in input CSV")
    p.add_argument("--id-col", default=None, help="Optional identifier column to carry through")

    p.add_argument(
        "--cross-tokenizer-name",
        default="dmis-lab/biobert-base-cased-v1.1",
        help="Tokenizer name used for lexical retrieval (must match offline preprocessing).",
    )

    p.add_argument(
        "--cross-encoder-model-id",
        required=True,
        help="HuggingFace model id of the cross-encoder (sequence classification).",
    )

    p.add_argument(
        "--retrieval-mode",
        choices=["fallback", "hybrid"],
        default="fallback",
        help="Candidate retrieval strategy: fallback (lexical then semantic) or hybrid (merge lexical+semantic).",
    )

    p.add_argument("--lexical-top-k", type=int, default=100)
    p.add_argument("--semantic-top-k", type=int, default=100)
    p.add_argument("--merged-top-k", type=int, default=150)
    p.add_argument("--cross-top-k", type=int, default=10)

    p.add_argument("--device", default=None, help="cpu | cuda | None (auto)")
    p.add_argument("--cross-batch-size", type=int, default=32)
    p.add_argument("--cross-max-length", type=int, default=256)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_inference_csv(
        bundle_path=args.bundle,
        ontology_csv=args.ontology_csv,
        input_csv=args.input_csv,
        out_csv=args.out_csv,
        attr_col=args.attr_col,
        id_col=args.id_col,
        cross_tokenizer_name=args.cross_tokenizer_name,
        cross_encoder_model_id=args.cross_encoder_model_id,
        retrieval_mode=args.retrieval_mode,
        lexical_top_k=args.lexical_top_k,
        semantic_top_k=args.semantic_top_k,
        merged_top_k=args.merged_top_k,
        cross_top_k=args.cross_top_k,
        device=args.device,
        cross_batch_size=args.cross_batch_size,
        cross_max_length=args.cross_max_length,
    )

    print(f"[DONE] Predictions written to: {args.out_csv}")


if __name__ == "__main__":
    main()