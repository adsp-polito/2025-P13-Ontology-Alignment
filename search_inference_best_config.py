from run_inference import run_inference_csv
import optuna
import argparse
import pandas as pd
from testing.new_evaluate_inference import join_predictions_to_ground_truth, compute_ranking_metrics_on_positives
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def objective(trial, args):

    if args.mode == "hybrid":
        retrieval_lexical_top_k = trial.suggest_categorical("retrieval_lexical_top_k", [50, 100, 200, 300])
        retrieval_semantic_top_k = trial.suggest_categorical("retrieval_semantic_top_k", [50, 100, 200, 300])
        retrieval_merged_top_k = trial.suggest_categorical("retrieval_merged_top_k", [100, 150, 200, 300])
        hybrid_ratio_semantic = trial.suggest_categorical("hybrid_ratio_semantic", [0.3, 0.5, 0.7])
        cross_top_k = trial.suggest_categorical("cross_top_k", [20, 50, 75, 100, 150])
    else:
        retrieval_lexical_top_k = trial.suggest_categorical("retrieval_lexical_top_k", [50, 100, 200, 300])
        retrieval_semantic_top_k = 100 # default value in run_inference.py
        retrieval_merged_top_k = 150 # default value in run_inference.py
        hybrid_ratio_semantic = 0.5 # default value in run_inference.py
        cross_top_k = trial.suggest_categorical("cross_top_k", [20, 50, 75, 100, 150])

    if retrieval_merged_top_k > (retrieval_lexical_top_k + retrieval_semantic_top_k):
         raise optuna.TrialPruned()
    
    if cross_top_k > retrieval_merged_top_k:
        raise optuna.TrialPruned()

    run_inference_csv(
        bundle_path=args.bundle,
        ontology_csv=args.ontology_csv,
        input_csv=args.input_csv,
        out_csv=args.out_csv,
        retrieval_col=args.retrieval_col,
        scoring_col=args.scoring_col,
        mode=args.mode,
        cross_tokenizer_name=args.cross_tokenizer_name,
        cross_encoder_model_id=args.cross_encoder_model_id,
        device=args.device,
        retrieval_lexical_top_k=retrieval_lexical_top_k,
        retrieval_semantic_top_k=retrieval_semantic_top_k,
        retrieval_merged_top_k=retrieval_merged_top_k,
        hybrid_ratio_semantic=hybrid_ratio_semantic,
        semantic_batch_size=args.semantic_batch_size,
        cross_top_k=cross_top_k,
        cross_batch_size=args.cross_batch_size,
        cross_max_length=args.cross_max_length,
        keep_top_n=1,
    )

    df_gt = pd.read_csv(args.input_csv)
    df_pred = pd.read_csv(args.out_csv)

    # If the gold file has no match column, treat it as all positives.
    if args.gt_match_col not in df_gt.columns:
        df_gt[args.gt_match_col] = 1


    merged, _ = join_predictions_to_ground_truth(
        df_gt,
        df_pred,
        gt_id_col=args.gt_id_col,
        pred_id_col=args.pred_id_col,
        gt_text_col=args.gt_text_col,
        pred_text_col=args.pred_text_col,
    )

    metrics = compute_ranking_metrics_on_positives(
        merged,
        gold_col=args.gt_gold_col,
        match_col=args.gt_match_col,
        k=1,
    )

    """
    "n_pos": float(n_pos),
    "precision_at_1_pos": float(p_at_1),
    "hits_at_k_pos": float(hits / n_pos),
    "mrr_at_k_pos": float(rr_sum / n_pos),
    """

    return metrics["precision_at_1_pos"]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search Inference Best Config")

    p.add_argument("--bundle", required=True, help="Path to offline_bundle.pkl")
    p.add_argument("--ontology-csv", required=True, help="Path to ontology export CSV (iri,text,...)")
    p.add_argument("--input-csv", required=True, help="Path to input CSV containing attributes")
    p.add_argument("--out-csv", required=True, help="Path to output predictions CSV")
    p.add_argument(
        "--retrieval-col",
        default="attribute",
        help="Column name used for retrieval (exact match + lexical). Typically the attribute LABEL.",
    )
    p.add_argument(
        "--scoring-col",
        default=None,
        help="Column name used for scoring (cross-encoder) and semantic retrieval (bi-encoder). "
             "If not provided, defaults to --retrieval-col.",
    )
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
    p.add_argument("--semantic-batch-size", type=int, default=64,
                   help= "32 if GPU is small and texts are long. 128 if GPU is large and texts are short.")
    p.add_argument("--cross-batch-size", type=int, default=32)
    p.add_argument("--cross-max-length", type=int, default=256)
    # p.add_argument("--keep-top-n", type=int, default=0)

    # p.add_argument("--k", type=int, default=10, help="K for Hits@K and MRR@K")

    p.add_argument("--gt-id-col", default=None, help="Optional GT id col for join (rare).")
    p.add_argument("--pred-id-col", default=None, help="Optional pred id col for join (rare).")

    p.add_argument("--gt-gold-col", default="gold_target_iris", help="Gold target IRI col in ground truth.")
    p.add_argument("--gt-match-col", default="match", help="Match col in ground truth (if missing, assumed all-positives).")
    p.add_argument("--gt-text-col", default="source_text", help="Attribute text col in ground truth (legacy).")
    p.add_argument("--pred-text-col", default="attribute_text", help="Attribute text col in predictions.")

    return p.parse_args()

def main() -> None:
    args = parse_args()

    set_seed(42)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=100)

    print(study.best_params)

    # To run with best params:
    # python search_inference_best_config.py --bundle outputs/offline_bundle.pkl --ontology-csv outputs/internal_ontology.csv --input-csv datasets/val_split_inference.csv --out-csv outputs/test_predictions.csv --cross-encoder-model-id outputs/cross_encoder_model_scibert/final_cross_encoder_model --mode hybrid --retrieval-col source_label --scoring-col source_text --gt-gold-col target_iri --gt-match-col match --gt-text-col source_text --pred-text-col attribute_text --device cuda --semantic-batch-size 32 --cross-batch-size 32 --cross-max-length 256

if __name__ == "__main__":
    main()