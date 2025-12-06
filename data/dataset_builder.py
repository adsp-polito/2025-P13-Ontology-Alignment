from miners.hard_negatives_miner import generate_hard_negatives
from miners.random_negatives_miner import generate_random_negatives
import pandas as pd


def build_training_dataset(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        df_align: pd.DataFrame
) -> pd.DataFrame:
    
    num_alignments = len(df_align)
    # Generate hard negatives -> 50% of the number of alignments
    df_hard_negatives = generate_hard_negatives(df1, df2, df_align, num_hard_negatives = int(num_alignments/2), top_n=5)

    # Combine positives and negatives
    df_training = pd.concat([df_align, df_hard_negatives], axis=0, ignore_index=True)

    # Generate random negatives -> 50% of the number of alignments
    df_random_negatives = generate_random_negatives(df1, df2, df_training, num_random_negatives = int(num_alignments/2))

    # Final training dataset (50% positives + 25% hard negatives + 25% random negatives)
    df_training_final = pd.concat([df_training, df_random_negatives], axis=0, ignore_index=True)

    # Merge with source and target texts
    merged = df_training_final.merge(df1, on="source_iri", how="left")
    merged = merged.merge(df2, on="target_iri", how="left")

    # Return final dataset with relevant columns
    return merged[["source_iri", "target_iri", "source_text", "target_text", "match"]]
