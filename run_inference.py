"""
Production-ready CLI entrypoint for ontology-attribute alignment inference.

Pipeline (no training):
  Stage A) Retrieval (high recall) via inference/retrieval.py
    - exact match
    - lexical retrieval (IDF overlap)
    - semantic retrieval (bi-encoder + cosine) either as fallback (lexical mode)
      or always combined (hybrid mode)

  Stage B) Scoring (high precision) via inference/scoring.py
    - cross-encoder scores (attribute_text, class_text) pairs in BATCH
    - selects best prediction per attribute
    - optionally keeps top-N scored candidates

Input:
  - offline bundle pickle (built offline)
  - ontology CSV (iri, text, ...)
  - input CSV with attribute column

Output:
  - predictions CSV with one row per attribute (+ optional top-N columns)

Key production properties:
  - tokenizer/model loaded ONCE:
      CandidateRetriever caches lexical tokenizer + optional bi-encoder
      CrossEncoderScorer caches cross-encoder tokenizer + model
  - retrieval is batched (semantic queries encoded in batch)
  - scoring is batched across ALL pairs (chunked) to avoid OOM

Run example:
  python run_inference.py \
    --bundle data/offline_bundle.pkl \
    --ontology-csv data/internal_ontology.csv \
    --input-csv data/study_attributes.csv \
    --out-csv data/predictions.csv \
    --attr-col attribute \
    --id-col attribute_id \
    --mode hybrid \
    --cross-encoder-model-id <YOUR_CE_MODEL> \
    --cross-batch-size 32 \
    --retrieval-lexical-top-k 120 \
    --retrieval-semantic-top-k 120 \
    --retrieval-merged-top-k 200 \
    --hybrid-ratio-semantic 0.5 \
    --keep-top-n 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ontologies.offline_preprocessing import load_offline_bundle

from inference.retrieval import CandidateRetriever, Candidate
from inference.scoring import CrossEncoderScorer, ScoredCandidate


# -----------------------------
# IO helpers
# -----------------------------

def load_ontology_text_map(ontology_csv: str | Path) -> Dict[str, str]:
    """
    Load ontology CSV and build mapping: iri -> text.

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


def _safe_str(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x)


# -----------------------------
# Scoring utilities (batched across all pairs)
# -----------------------------

def build_pairs_for_scoring(
    attr_texts: Sequence[str],
    candidates_per_attr: Sequence[List[Candidate]],
    *,
    iri2text: Dict[str, str],
    cross_top_k: int,
    mode: str = "lexical",            # "lexical" | "hybrid"
    hybrid_ratio_semantic: float = 0.5,
) -> Tuple[List[str], List[str], List[Tuple[int, str]]]:
    """
    Build global pair lists for cross-encoder scoring.

    - Skips attributes whose retrieval already returned an EXACT match.
    - In lexical mode: score up to cross_top_k candidates (lexical or semantic fallback).
    - In hybrid mode: score up to cross_top_k candidates with a lexical/semantic split.
    """
    left: List[str] = []
    right: List[str] = []
    meta: List[Tuple[int, str]] = []

    mode = str(mode).lower().strip()
    if mode not in {"lexical", "hybrid"}:
        raise ValueError(f"Unsupported mode: {mode}")

    k = max(int(cross_top_k), 1)
    r = float(hybrid_ratio_semantic)
    r = min(max(r, 0.0), 1.0)

    for i, a in enumerate(attr_texts):
        a_clean = _safe_str(a).strip()
        if not a_clean:
            continue

        all_cands = list(candidates_per_attr[i])

        # 1) If EXACT exists, do NOT score this attribute with the cross-encoder.
        if any(str(c.source) == "exact" for c in all_cands):
            continue

        # 2) Select candidates to score
        if mode == "lexical":
            # lexical mode: just take the first k non-exact candidates (usually all lexical, or semantic fallback list)
            cands = all_cands[:k]

        else:  # hybrid
            k_sem = int(round(k * r))
            k_lex = k - k_sem

            lex_all = [c for c in all_cands if str(c.source) == "lexical"]
            sem_all = [c for c in all_cands if str(c.source) == "semantic"]

            lex_list = lex_all[:k_lex]
            sem_list = sem_all[:k_sem]

            cands = lex_list + sem_list

            # fill if short
            if len(cands) < k:
                need = k - len(cands)
                # fill semantic first, then lexical (o viceversa: scegli e basta)
                sem_extra = sem_all[len(sem_list):len(sem_list) + need]
                cands += sem_extra
                need = k - len(cands)
                if need > 0:
                    lex_extra = lex_all[len(lex_list):len(lex_list) + need]
                    cands += lex_extra

        if not cands:
            continue

        # 3) Build pairs (dedup by iri to avoid scoring duplicates)
        seen_iri: set[str] = set()
        for c in cands:
            iri = str(c.iri)
            if iri in seen_iri:
                continue
            seen_iri.add(iri)

            txt = iri2text.get(iri)
            if txt is None:
                continue

            left.append(a_clean)
            right.append(str(txt))
            meta.append((i, iri))

    return left, right, meta


def reduce_scores_to_predictions(
    meta: Sequence[Tuple[int, str]],
    scores: np.ndarray,
    *,
    n_attrs: int,
    keep_top_n: int = 0,
) -> Tuple[List[Optional[ScoredCandidate]], Optional[List[List[ScoredCandidate]]]]:
    """
    Reduce scored pairs back to per-attribute predictions.

    Args:
        meta: list of (attr_index, iri) for each scored pair
        scores: float scores aligned with meta
        keep_top_n: if >0, also keep ranked top-N per attribute

    Returns:
        best_per_attr: list length = num_attributes, each is ScoredCandidate or None
        topn_per_attr: optional list length = num_attributes, each is a list of ScoredCandidate
    """
    if scores.ndim != 1 or len(scores) != len(meta):
        raise ValueError("scores must be 1D and aligned with meta")

    n_attrs = int(n_attrs)
    if n_attrs < 0:
        raise ValueError(f"n_attrs must be >= 0, got {n_attrs}")

    # group by attribute index
    by_attr: Dict[int, List[ScoredCandidate]] = {}
    for (i, iri), s in zip(meta, scores):
        i = int(i)
        if i < 0 or i >= n_attrs:
            raise ValueError(f"Attribute index out of range: i={i}, n_attrs={n_attrs}")
        by_attr.setdefault(i, []).append(ScoredCandidate(iri=str(iri), score=float(s)))

    best: List[Optional[ScoredCandidate]] = [None for _ in range(n_attrs)]
    topn: Optional[List[List[ScoredCandidate]]] = None
    if keep_top_n > 0:
        topn = [[] for _ in range(n_attrs)]

    # fill only indices that have scored pairs
    for i, lst in by_attr.items():
        lst_sorted = sorted(lst, key=lambda x: x.score, reverse=True)
        best[i] = lst_sorted[0] if lst_sorted else None
        if topn is not None:
            topn[i] = lst_sorted[: int(keep_top_n)]

    return best, topn


# -----------------------------
# Main inference (CSV -> CSV)
# -----------------------------

def run_inference_csv(
    *,
    bundle_path: str | Path,
    ontology_csv: str | Path,
    input_csv: str | Path,
    out_csv: str | Path,
    attr_col: str = "attribute",
    id_col: Optional[str] = None,
    mode: str = "lexical",  # "lexical" | "hybrid"
    cross_tokenizer_name: str = "dmis-lab/biobert-base-cased-v1.1", # must match offline preprocessing
    cross_encoder_model_id: str = "",
    device: Optional[str] = None,
    retrieval_lexical_top_k: int = 100,
    retrieval_semantic_top_k: int = 100,
    retrieval_merged_top_k: int = 150,
    hybrid_ratio_semantic: float = 0.5,
    semantic_batch_size: int = 64,
    cross_top_k: int = 20,
    cross_batch_size: int = 32,
    cross_max_length: int = 256,
    keep_top_n: int = 0,
) -> None:
    """
    End-to-end inference on a CSV of attributes and write predictions to CSV.
    """
    if not cross_encoder_model_id:
        raise ValueError("--cross-encoder-model-id is required (cross-encoder scoring stage).")

    mode = str(mode).lower().strip()
    if mode not in {"lexical", "hybrid"}:
        raise ValueError(f"Unsupported mode: {mode}")

    # Load offline bundle (semantic embeddings should be loaded/mmap'ed for semantic retrieval)
    bundle = load_offline_bundle(Path(bundle_path), load_semantic_embeddings=True, mmap=True)
    iri2text = load_ontology_text_map(Path(ontology_csv))

    # Load input
    df_in = pd.read_csv(Path(input_csv))
    if attr_col not in df_in.columns:
        raise ValueError(f"Input CSV missing attr_col='{attr_col}'. Found columns: {list(df_in.columns)}")
    if id_col is not None and id_col not in df_in.columns:
        raise ValueError(f"id_col='{id_col}' not found. Found columns: {list(df_in.columns)}")

    attr_texts: List[str] = [_safe_str(x) for x in df_in[attr_col].tolist()]
    n = len(attr_texts)

    # Stage A: Retrieval (batch)
    retriever = CandidateRetriever(
        bundle=bundle,
        cross_tokenizer_name=cross_tokenizer_name,
        semantic_device=device,
        semantic_batch_size=int(semantic_batch_size),
    )

    candidates_per_attr, retrieval_sources = retriever.retrieve_batch(
        attr_texts,
        mode=mode,
        lexical_top_k=int(retrieval_lexical_top_k),
        semantic_top_k=int(retrieval_semantic_top_k),
        merged_top_k=int(retrieval_merged_top_k),
        hybrid_ratio_semantic=float(hybrid_ratio_semantic),
        semantic_batch_size=int(semantic_batch_size),
    )

    # Stage B: Scoring (batched across ALL (attr, cand) pairs)
    scorer = CrossEncoderScorer(
        model_id=cross_encoder_model_id,
        device=device,
        max_length=int(cross_max_length),
    )

    left, right, meta = build_pairs_for_scoring(
        attr_texts,
        candidates_per_attr,
        iri2text=iri2text,
        cross_top_k=int(cross_top_k),
        mode=mode,
        hybrid_ratio_semantic=float(hybrid_ratio_semantic),
    )

    if not meta:
        # No pairs to score => empty predictions or exact matches only
        out_rows: List[Dict[str, Any]] = []
        for i in range(n):
            row: Dict[str, Any] = {}
            if id_col is not None:
                row[id_col] = df_in.iloc[i][id_col]
            else:
                row["row_id"] = int(i)

            exact_iris = [c.iri for c in candidates_per_attr[i] if str(c.source) == "exact"]
            if exact_iris:
                row["predicted_iri"] = exact_iris[0]  # oppure li concatenI, ma meglio 1
                row["predicted_score"] = 1.0
                row["num_scored"] = 0
            else:
                row["predicted_iri"] = None
                row["predicted_score"] = None
                row["num_scored"] = 0

            row["attribute_text"] = attr_texts[i]
            row["retrieval_source"] = retrieval_sources[i] if i < len(retrieval_sources) else "none"
            row["num_retrieved"] = len(candidates_per_attr[i]) if i < len(candidates_per_attr) else 0
            out_rows.append(row)

        df_out = pd.DataFrame(out_rows)
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)
        return

    scores = scorer.score_pairs(left, right, batch_size=int(cross_batch_size))

    scored_count = np.zeros(n, dtype=int)
    for (i, _iri) in meta:
        scored_count[int(i)] += 1

    # Reduce scores to best prediction (+ optional top-N)
    best_per_attr, topn_per_attr = reduce_scores_to_predictions(
    meta,
    scores,
    n_attrs=n,
    keep_top_n=int(keep_top_n),
    )

    # Build output rows
    out_rows: List[Dict[str, Any]] = []
    for i in range(n):
        row: Dict[str, Any] = {}
        if id_col is not None:
            row[id_col] = df_in.iloc[i][id_col]
        else:
            row["row_id"] = int(i)
        
        row["attribute_text"] = attr_texts[i]
        row["retrieval_source"] = retrieval_sources[i] if i < len(retrieval_sources) else "none"
        row["num_retrieved"] = len(candidates_per_attr[i]) if i < len(candidates_per_attr) else 0

        exact_iris = [c.iri for c in candidates_per_attr[i] if str(c.source) == "exact"]
        if exact_iris:
            row["predicted_iri"] = exact_iris[0]
            row["predicted_score"] = 1.0
            row["num_scored"] = 0
        else:
            best_i = best_per_attr[i] if i < len(best_per_attr) else None
            row["num_scored"] = int(scored_count[i])
            if best_i is None:
                row["predicted_iri"] = None
                row["predicted_score"] = None
            else:
                row["predicted_iri"] = best_i.iri
                row["predicted_score"] = float(best_i.score)

        # Optional top-N columns
        if topn_per_attr is not None and i < len(topn_per_attr):
            topn = topn_per_attr[i]
            # store as repeated columns to keep CSV simple for downstream consumers
            for rank, sc in enumerate(topn, start=1):
                row[f"top{rank}_iri"] = sc.iri
                row[f"top{rank}_score"] = float(sc.score)

        out_rows.append(row)

    df_out = pd.DataFrame(out_rows)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ontology alignment inference (retrieval + cross-encoder scoring).")

    p.add_argument("--bundle", required=True, help="Path to offline_bundle.pkl")
    p.add_argument("--ontology-csv", required=True, help="Path to ontology export CSV (iri,text,...)")
    p.add_argument("--input-csv", required=True, help="Path to input CSV containing attributes")
    p.add_argument("--out-csv", required=True, help="Path to output predictions CSV")

    p.add_argument("--attr-col", default="attribute", help="Column name for attribute text in input CSV")
    p.add_argument("--id-col", default=None, help="Optional identifier column to carry through")

    p.add_argument(
        "--mode",
        choices=["lexical", "hybrid"],
        default="lexical",
        help=(
            "Retrieval mode: "
            "lexical = exact -> lexical, fallback to semantic if lexical empty; "
            "hybrid = exact -> always combine lexical+semantic (no fallback)."
        ),
    )

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

    p.add_argument("--device", default=None, help="cpu | cuda | None (auto)")

    # Retrieval params
    p.add_argument("--retrieval-lexical-top-k", type=int, default=100)
    p.add_argument("--retrieval-semantic-top-k", type=int, default=100)
    p.add_argument("--retrieval-merged-top-k", type=int, default=150)
    p.add_argument("--hybrid-ratio-semantic", type=float, default=0.5)
    p.add_argument("--semantic-batch-size", type=int, default=64,
                   help= "32 if GPU is small and texts are long. 128 if GPU is large and texts are short.")

    # Scoring params
    p.add_argument(
        "--cross-top-k",
        type=int,
        default=20,
        help="How many retrieved candidates per attribute to send to the cross-encoder.",
    )
    p.add_argument("--cross-batch-size", type=int, default=32)
    p.add_argument("--cross-max-length", type=int, default=256)

    # Output detail
    p.add_argument(
        "--keep-top-n",
        type=int,
        default=0,
        help="If >0, store top-N scored candidates in extra CSV columns.",
    )

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
        mode=args.mode,
        cross_tokenizer_name=args.cross_tokenizer_name,
        cross_encoder_model_id=args.cross_encoder_model_id,
        device=args.device,
        retrieval_lexical_top_k=args.retrieval_lexical_top_k,
        retrieval_semantic_top_k=args.retrieval_semantic_top_k,
        retrieval_merged_top_k=args.retrieval_merged_top_k,
        hybrid_ratio_semantic=args.hybrid_ratio_semantic,
        semantic_batch_size=args.semantic_batch_size,
        cross_top_k=args.cross_top_k,
        cross_batch_size=args.cross_batch_size,
        cross_max_length=args.cross_max_length,
        keep_top_n=args.keep_top_n,
    )

    print(f"[DONE] Predictions written to: {args.out_csv}")


if __name__ == "__main__":
    main()