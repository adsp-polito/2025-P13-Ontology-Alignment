from miners.hard_negatives_miner import generate_hard_negatives
from miners.random_negatives_miner import generate_random_negatives
import pandas as pd

def build_training_dataset(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        df_align: pd.DataFrame
) -> pd.DataFrame:
    
    # Filter alignment to only include valid IRIs present in source and target dataframes
    df_align = df_align[df_align["source_iri"].isin(df1["source_iri"]) & df_align["target_iri"].isin(df2["target_iri"])]
    print(f"Filtered alignment to {len(df_align)} valid correspondences.")

    num_alignments = len(df_align)
    # Generate hard negatives -> 50% of the number of alignments
    df_hard_negatives = generate_hard_negatives(df1, df2, df_align, num_hard_negatives = int(num_alignments/2))
    print(f"Generated {len(df_hard_negatives)} hard negatives.")

    # Combine positives and negatives
    df_training = pd.concat([df_align, df_hard_negatives], axis=0, ignore_index=True)

    # Generate random negatives -> 50% of the number of alignments
    df_random_negatives = generate_random_negatives(df1, df2, df_training, num_random_negatives = int(num_alignments/2))
    print(f"Generated {len(df_random_negatives)} random negatives.")

    # Final training dataset (50% positives + 25% hard negatives + 25% random negatives)
    df_training_final = pd.concat([df_training, df_random_negatives], axis=0, ignore_index=True)

    # Merge with source and target texts
    merged = df_training_final.merge(df1, on="source_iri", how="left")
    merged = merged.merge(df2, on="target_iri", how="left")

    # Return final dataset with relevant columns
    return merged[["source_iri", "target_iri", "source_label", "target_label", "source_text", "target_text", "sample_type", "match"]]
