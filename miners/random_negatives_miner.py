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
    
    random_negatives = []
    total_pairs = len(df1) * len(df2)
    sampled_indices = set()
    while len(random_negatives) < num_random_negatives and len(sampled_indices) < total_pairs:
        idx1 = np.random.randint(0, len(df1))
        idx2 = np.random.randint(0, len(df2))
        if (idx1, idx2) not in sampled_indices \
            and (df_samples.empty or not ((df_samples["source_iri"] == df1.iloc[idx1]["source_iri"]) & (df_samples["target_iri"] == df2.iloc[idx2]["target_iri"])).any()):
            # if (idx1, idx2) not in sampled_indices and the pair is not in the samples (positive sammples + hard negatives)
            sampled_indices.add((idx1, idx2))
            source_iri = df1.iloc[idx1]["source_iri"]
            target_iri = df2.iloc[idx2]["target_iri"]
            random_negatives.append({"source_iri": source_iri, "target_iri": target_iri, "match": 0.0}) # non-match
    df_random_negatives = pd.DataFrame(random_negatives)
    # print("\nGenerated random negatives:")
    # print(df_random_negatives.head())
    return df_random_negatives
