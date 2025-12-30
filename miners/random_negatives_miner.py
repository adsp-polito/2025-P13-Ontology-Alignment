import pandas as pd
import numpy as np

def generate_random_negatives(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        df_samples: pd.DataFrame,
        num_random_negatives: int = 100,
) -> pd.DataFrame:
    
    """_summary_

    Returns:
        df1 (pd.DataFrame): first dataframe
        df2 (pd.DataFrame): second dataframe
        df_samples (pd.DataFrame): dataframe with existing samples to avoid sampling them again
        num_random_negatives (int, optional): number of random negatives to generate. Defaults to 100.
    """
    if num_random_negatives <= 0:
        return pd.DataFrame(columns=["source_iri", "target_iri", "match"])

    if df1.empty or df2.empty:
        return pd.DataFrame(columns=["source_iri", "target_iri", "match"])
    
    random_negatives = []
    total_pairs = len(df1) * len(df2)
    sampled_indices = set()
    alignment_set = set(zip(df_samples["source_iri"], df_samples["target_iri"]))
    np.random.seed(42)  # For reproducibility

    while len(random_negatives) < num_random_negatives and len(sampled_indices) < total_pairs:
        idx1 = np.random.randint(0, len(df1)) # random index for df1
        idx2 = np.random.randint(0, len(df2)) # random index for df2
        source_iri = df1.iloc[idx1]["source_iri"]
        target_iri = df2.iloc[idx2]["target_iri"]

        if (idx1, idx2) not in sampled_indices \
            and (source_iri, target_iri) not in alignment_set:
            # if (idx1, idx2) not in sampled_indices and the pair is not in the samples (positive samples + hard negatives)
            sampled_indices.add((idx1, idx2))
            source_iri = df1.iloc[idx1]["source_iri"]
            target_iri = df2.iloc[idx2]["target_iri"]
            random_negatives.append({"source_iri": source_iri, "target_iri": target_iri, "sample_type": "random_negative", "match": 0.0}) # non-match

    df_random_negatives = pd.DataFrame(random_negatives)
    return df_random_negatives
