from __future__ import annotations
from sklearn.model_selection import train_test_split
from pathlib import Path
from datasets import Dataset
import pandas as pd

def stratified_split(
    dataset: pd.DataFrame,
    *,
    split_ratios: tuple[float, float, float] = (0.75, 0.15, 0.10),
    seed: int = 42,
):
    """
    Stratified split into train/val/test using GLOBAL ratios.

    split_ratios = (train, val, test) must sum to 1.0 (within tolerance).
    """
    if "match" not in dataset.columns:
        raise ValueError("Dataset must contain a 'match' column for stratification.")

    train_r, val_r, test_r = map(float, split_ratios)
    s = train_r + val_r + test_r
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"split_ratios must sum to 1.0, got sum={s} with {split_ratios}")
    if any(r < 0.0 for r in (train_r, val_r, test_r)):
        raise ValueError(f"split_ratios must be non-negative, got {split_ratios}")
    if test_r <= 0.0 or val_r <= 0.0 or train_r <= 0.0:
        raise ValueError(f"All split ratios must be > 0. Got {split_ratios}")

    # 1) Take test directly as GLOBAL ratio
    df_train_val, df_test = train_test_split(
        dataset,
        test_size=test_r,
        random_state=seed,
        shuffle=True,
        stratify=dataset["match"],
    )

    # 2) Take val as GLOBAL ratio from the remaining pool
    # remaining ratio = train_r + val_r
    remaining = train_r + val_r
    val_size_within_remaining = val_r / remaining  # e.g., 0.15 / 0.90 = 0.166666...

    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_size_within_remaining,
        random_state=seed,
        shuffle=True,
        stratify=df_train_val["match"],
    )

    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )
    

def convert_df_to_dataset(
        df: pd.DataFrame
) -> Dataset:
    """
    Convert pandas dataframe to HuggingFace Dataset.

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        Dataset: HuggingFace Dataset
    """
    return Dataset.from_dict({
        "source_text": df["source_text"].tolist(),
        "target_text": df["target_text"].tolist(),
        "label": df["match"].astype(float).tolist()
    })



def build_queries_and_gold_from_pairwise_test(
    df_test: pd.DataFrame,
    *,
    id_col: str = "source_iri",
    retrieval_col: str = "source_label",
    scoring_col: str = "source_text",
    target_id_col: str = "target_iri",
    match_col: str = "match",
):
    required = {id_col, retrieval_col, scoring_col, target_id_col, match_col}
    missing = required - set(df_test.columns)
    if missing:
        raise ValueError(f"Test split missing columns for Option B: {sorted(missing)}")

    # Keep only evaluable queries: those that have at least one positive
    df_pos = df_test[df_test[match_col] == 1]
    valid_ids = df_pos[id_col].dropna().unique()
    if len(valid_ids) == 0:
        raise ValueError("No positive matches found in test split (match==1). Cannot build queries+gold.")

    # Queries: one row per source_iri
    queries = (
        df_test[df_test[id_col].isin(valid_ids)][[id_col, retrieval_col, scoring_col]]
        .drop_duplicates(subset=[id_col], keep="first")
        .reset_index(drop=True)
    )

    # Gold: list of all positive targets per source_iri
    gold = (
        df_pos[df_pos[id_col].isin(valid_ids)][[id_col, target_id_col]]
        .groupby(id_col)[target_id_col]
        .apply(lambda s: list(pd.unique(s.dropna())))
        .reset_index()
        .rename(columns={target_id_col: "gold_target_iris"})
    )

    # Align gold row-by-row with queries order
    gold_aligned = queries[[id_col]].merge(gold, on=id_col, how="left")
    gold_aligned["gold_target_iris"] = gold_aligned["gold_target_iris"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    return queries, gold_aligned


def save_split_and_eval_artifacts(
    base_dataset_csv_path: str,
    *,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
):
    base = Path(base_dataset_csv_path)
    stem = base.with_suffix("")

    train_path = stem.with_suffix(".train.csv")
    val_path = stem.with_suffix(".val.csv")
    test_path = stem.with_suffix(".test.csv")

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    # Option B artifacts from pairwise test
    queries, gold = build_queries_and_gold_from_pairwise_test(df_test)

    queries_path = stem.with_suffix(".test.queries.csv")
    gold_path = stem.with_suffix(".test.gold.csv")
    queries.to_csv(queries_path, index=False)
    gold.to_csv(gold_path, index=False)

    return {
        "train": str(train_path),
        "val": str(val_path),
        "test": str(test_path),
        "queries": str(queries_path),
        "gold": str(gold_path),
    }