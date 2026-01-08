"""
new_evaluate_inference.py

Evaluate run_inference.py output against a ground-truth CSV.

Original intent (legacy):
- join predictions to test_split.csv via row_id (preferred) or explicit ids or (rare) text join
- compute coverage + ranking metrics on positives only (match==1)
- optional breakdown by retrieval_source

Compatibility patch (minimal, to support your current inference outputs):
Your inference produces:
- predictions.csv
- training_dataset.test.gold.csv

Those two files are typically compatible via:
- join on source_iri
- gold labels stored as `gold_target_iris` (often a stringified Python list, sometimes with >1 IRI)
- the gold file may not have a `match` column (it is effectively all positives)

Everything else is left unchanged.
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Helpers
# -------------------------

def _find_topk_iri_cols(df_pred: pd.DataFrame) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for c in df_pred.columns:
        m = re.fullmatch(r"top(\d+)_iri", str(c))
        if m:
            out.append((int(m.group(1)), c))
    out.sort(key=lambda x: x[0])
    return out


def _as_int01(x) -> Optional[int]:
    if pd.isna(x):
        return None
    try:
        v = float(x)
    except Exception:
        return None
    return 1 if v >= 0.5 else 0


@dataclass
class JoinPlan:
    method: str
    details: str


def join_predictions_to_ground_truth(
    df_gt: pd.DataFrame,
    df_pred: pd.DataFrame,
    *,
    gt_id_col: Optional[str] = None,
    pred_id_col: Optional[str] = None,
    gt_text_col: str = "source_text",
    pred_text_col: str = "attribute_text",
) -> Tuple[pd.DataFrame, JoinPlan]:
    """
    Join strategies (in order):
      A) explicit id join (if provided)
      A2) source_iri join (your current inference + gold format)
      B) row_id join (legacy default)
      C) text join (only if almost-unique)
    """
    # A) explicit id join
    if gt_id_col and pred_id_col and (gt_id_col in df_gt.columns) and (pred_id_col in df_pred.columns):
        merged = df_pred.merge(
            df_gt,
            left_on=pred_id_col,
            right_on=gt_id_col,
            how="left",
            suffixes=("_pred", "_gt"),
        )
        return merged, JoinPlan("id_col", f"pred[{pred_id_col}] == gt[{gt_id_col}]")

    # A2) source_iri join (current pipeline)
    if ("source_iri" in df_pred.columns) and ("source_iri" in df_gt.columns):
        merged = df_pred.merge(
            df_gt,
            on="source_iri",
            how="left",
            suffixes=("_pred", "_gt"),
        )
        return merged, JoinPlan("source_iri", "pred[source_iri] == gt[source_iri]")

    # B) row_id join (LEGACY DEFAULT)
    if "row_id" in df_pred.columns:
        df_gt_idx = df_gt.reset_index(drop=False).rename(columns={"index": "__gt_index"})
        merged = df_pred.merge(
            df_gt_idx,
            left_on="row_id",
            right_on="__gt_index",
            how="left",
            suffixes=("_pred", "_gt"),
        ).drop(columns=["__gt_index"])
        return merged, JoinPlan("row_id", "pred[row_id] == gt[row_index]")

    # C) text join fallback (only if safe)
    if (pred_text_col in df_pred.columns) and (gt_text_col in df_gt.columns):
        gt_texts = df_gt[gt_text_col].astype(str)
        uniq_ratio = gt_texts.nunique(dropna=True) / max(len(gt_texts), 1)
        if uniq_ratio > 0.98:
            merged = df_pred.merge(
                df_gt,
                left_on=pred_text_col,
                right_on=gt_text_col,
                how="left",
                suffixes=("_pred", "_gt"),
            )
            return merged, JoinPlan(
                "text_col",
                f"pred[{pred_text_col}] == gt[{gt_text_col}] (uniq_ratio={uniq_ratio:.3f})",
            )

    raise ValueError(
        "Cannot join predictions to ground truth.\n"
        "Predictions CSV has no 'row_id' and no explicit id columns were provided,\n"
        "and source_iri-based join was not possible.\n"
        "Fix: ensure both files share a stable key (e.g., source_iri), or run inference without --id-col "
        "(so it emits row_id), or pass --gt-id-col/--pred-id-col."
    )


def compute_basic_stats(df_pred: pd.DataFrame) -> Dict[str, Any]:
    n = len(df_pred)
    out: Dict[str, Any] = {"n": int(n)}
    if n == 0:
        out["coverage"] = float("nan")
        return out

    if "predicted_iri" in df_pred.columns:
        out["coverage"] = float(df_pred["predicted_iri"].notna().mean())
    else:
        out["coverage"] = 0.0

    if "retrieval_source" in df_pred.columns:
        vc = df_pred["retrieval_source"].fillna("none").astype(str).value_counts(normalize=True)
        out["retrieval_source_dist"] = {k: float(v) for k, v in vc.items()}
    return out


def compute_ranking_metrics_on_positives(
    merged: pd.DataFrame,
    *,
    gold_col: str = "gold_target_iris",
    match_col: str = "match",
    k: int = 10,
) -> Dict[str, float]:
    """
    Metrics computed ONLY on positive test examples (match==1):
      - Precision@1 (whether predicted_iri matches gold)
      - Hits@K (whether gold appears in top-K list)
      - MRR@K

    Compatibility:
      - gold_col may contain:
          * a single IRI string, OR
          * a stringified Python list of IRIs, e.g. "['iri1']" or "['iri1','iri2']"
        In the multi-gold case, the prediction is correct if it matches ANY gold IRI,
        and MRR uses the best (lowest) rank among the gold IRIs.
    """
    if gold_col not in merged.columns:
        raise ValueError(f"Missing gold column: {gold_col}")
    if match_col not in merged.columns:
        raise ValueError(f"Missing match column: {match_col}")
    if "predicted_iri" not in merged.columns:
        raise ValueError("Predictions missing 'predicted_iri'.")

    y = merged[match_col].apply(_as_int01)
    df_pos = merged.loc[y == 1].copy()
    n_pos = len(df_pos)
    if n_pos == 0:
        return {
            "n_pos": 0.0,
            "precision_at_1_pos": float("nan"),
            "hits_at_k_pos": float("nan"),
            "mrr_at_k_pos": float("nan"),
        }

    use_k = max(int(k), 1)

    def _parse_gold(v: Any) -> List[str]:
        if isinstance(v, list):
            return [str(x) for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
        if pd.isna(v):
            return []
        if isinstance(v, str):
            s = v.strip()
            # try parse stringified list
            if s.startswith("[") and s.endswith("]"):
                try:
                    vv = ast.literal_eval(s)
                    if isinstance(vv, list):
                        return [str(x) for x in vv if x is not None]
                except Exception:
                    # fall back to treating it as a single string
                    pass
            return [s]
        return [str(v)]

    # Build per-row ranked list (length <= K) and per-row gold list(s)
    ranked_lists: List[List[str]] = []
    gold_lists: List[List[str]] = []

    for _, r in df_pos.iterrows():
        ranks: List[str] = []
        ranks.append("" if pd.isna(r["predicted_iri"]) else str(r["predicted_iri"]))
        for kk in range(2, use_k + 1):
            c = f"top{kk}_iri"
            if c in df_pos.columns:
                ranks.append("" if pd.isna(r[c]) else str(r[c]))
        ranks = ranks[:use_k]
        ranked_lists.append(ranks)

        gold_lists.append(_parse_gold(r[gold_col]))

    pred1 = df_pos["predicted_iri"].astype(str).tolist()

    # Precision@1 on positives: correct if pred1 equals ANY gold
    correct1 = 0
    for g_list, p in zip(gold_lists, pred1):
        if p in set(g_list):
            correct1 += 1
    p_at_1 = correct1 / n_pos

    # Hits@K + MRR@K: hit if ANY gold appears; RR uses best (smallest) rank
    hits = 0
    rr_sum = 0.0
    for g_list, ranks in zip(gold_lists, ranked_lists):
        gset = set(g_list)
        best_rank = None
        for idx, iri in enumerate(ranks, start=1):
            if iri in gset:
                best_rank = idx
                break
        if best_rank is not None:
            hits += 1
            rr_sum += 1.0 / best_rank

    return {
        "n_pos": float(n_pos),
        "precision_at_1_pos": float(p_at_1),
        "hits_at_k_pos": float(hits / n_pos),
        "mrr_at_k_pos": float(rr_sum / n_pos),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate run_inference.py outputs vs test split.")
    ap.add_argument("--test-split", required=True, help="Path to ground-truth CSV (e.g., training_dataset.test.gold.csv).")
    ap.add_argument("--predictions", required=True, help="Path to predictions.csv from run_inference.py")
    ap.add_argument("--k", type=int, default=10, help="K for Hits@K and MRR@K")

    ap.add_argument("--gt-id-col", default=None, help="Optional GT id col for join (rare).")
    ap.add_argument("--pred-id-col", default=None, help="Optional pred id col for join (rare).")

    # default adjusted to your gold file format
    ap.add_argument("--gt-gold-col", default="gold_target_iris", help="Gold target IRI col in ground truth.")
    ap.add_argument("--gt-match-col", default="match", help="Match col in ground truth (if missing, assumed all-positives).")
    ap.add_argument("--gt-text-col", default="source_text", help="Attribute text col in ground truth (legacy).")
    ap.add_argument("--pred-text-col", default="attribute_text", help="Attribute text col in predictions.")

    ap.add_argument("--out-merged", default=None, help="Optional path to save merged CSV (pred+gt).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    df_gt = pd.read_csv(args.test_split)
    df_pred = pd.read_csv(args.predictions)

    # If the gold file has no match column, treat it as all positives.
    if args.gt_match_col not in df_gt.columns:
        df_gt[args.gt_match_col] = 1

    stats = compute_basic_stats(df_pred)

    merged, plan = join_predictions_to_ground_truth(
        df_gt,
        df_pred,
        gt_id_col=args.gt_id_col,
        pred_id_col=args.pred_id_col,
        gt_text_col=args.gt_text_col,
        pred_text_col=args.pred_text_col,
    )

    # sanity: how many rows got a gold label
    gold_present = merged[args.gt_gold_col].notna().mean() if args.gt_gold_col in merged.columns else 0.0

    metrics = compute_ranking_metrics_on_positives(
        merged,
        gold_col=args.gt_gold_col,
        match_col=args.gt_match_col,
        k=int(args.k),
    )

    print("\n=== Evaluation Report ===")
    print(f"Join method: {plan.method} | {plan.details}")
    print(f"Pred rows: {stats['n']}")
    print(f"Coverage (predicted_iri != null): {stats.get('coverage', float('nan')):.4f}")
    print(f"GT attach rate (gold present after join): {gold_present:.4f}")

    rs = stats.get("retrieval_source_dist")
    if isinstance(rs, dict):
        print("\nRetrieval source distribution:")
        for k, v in sorted(rs.items(), key=lambda x: -x[1]):
            print(f"  {k:>8s}: {v:.4f}")

    print("\nMetrics on POSITIVES only (match==1):")
    print(f"  n_pos:             {int(metrics['n_pos'])}")
    print(f"  Precision@1 (pos): {metrics['precision_at_1_pos']:.4f}")
    print(f"  Hits@{int(args.k)} (pos):     {metrics['hits_at_k_pos']:.4f}")
    print(f"  MRR@{int(args.k)} (pos):      {metrics['mrr_at_k_pos']:.4f}")

    # --- INIZIO NUOVO BLOCCO ---
    if "retrieval_source" in merged.columns:
        print("\n=== Breakdown by Retrieval Source ===")
        sources = merged["retrieval_source"].dropna().unique()

        for src in sorted(sources):
            subset = merged[merged["retrieval_source"] == src].copy()

            sub_metrics = compute_ranking_metrics_on_positives(
                subset,
                gold_col=args.gt_gold_col,
                match_col=args.gt_match_col,
                k=int(args.k),
            )

            if sub_metrics["n_pos"] > 0:
                print(f"\nSource: {src}")
                print(f"  n_pos:             {int(sub_metrics['n_pos'])}")
                print(f"  Precision@1 (pos): {sub_metrics['precision_at_1_pos']:.4f}")
                print(f"  Hits@{int(args.k)} (pos):     {sub_metrics['hits_at_k_pos']:.4f}")
    # --- FINE NUOVO BLOCCO ---

    if args.out_merged:
        outp = Path(args.out_merged)
        outp.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(outp, index=False)
        print(f"\nSaved merged CSV to: {outp}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()