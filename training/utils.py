from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd


def stratified_split(
        dataset: pd.DataFrame,
        val_size: float = 0.15,
        test_size: float = 0.1
):
    """
    Stratified split of the dataset into train, val and test sets.

    Args:
        dataset (pd.DataFrame): input dataframe
        val_size (float, optional): size of the validation set. Defaults to 0.1.
        test_size (float, optional): size of the test set. Defaults to 0.15.

    Returns:
        _type_: _tuple: train, val, test dataframes
    """
    
    df_train_val, df_test = train_test_split(
        dataset,
        test_size=test_size,
        random_state=42,
        shuffle=True,
        stratify=dataset["match"]
    )

    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_size,
        random_state=42,
        shuffle=True,
        stratify=df_train_val["match"]
    )

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)
    

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
