from sklearn.model_selection import train_test_split
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
        "source_text": [r["source_text"] for _, r in df.iterrows()],
        "target_text": [r["target_text"] for _, r in df.iterrows()],
        "label": df["match"].astype(float).tolist()
    })