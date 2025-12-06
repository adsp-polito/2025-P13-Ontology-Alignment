import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch

def generate_hard_negatives(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        df_align: pd.DataFrame,
        num_hard_negatives: int = 100,
        top_n: int = 5,
) -> pd.DataFrame:
    """_summary_

    Args:
        df1 (pd.DataFrame): fist dataframe
        df2 (pd.DataFrame): second dataframe
        df_align (pd.DataFrame): alignment dataframe to avoid sampling alignments
        num_hard_negatives (int, optional): number of hard negatives to generate. Defaults to 100.
        top_n (int, optional): number of top similar candidates to consider. Defaults to 5.

    Returns:
        pd.DataFrame: hard negatives dataframe
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb', device=device)
    embeddings1 = model.encode(df1["source_text"].tolist(), convert_to_tensor=True, device=device) # short_text
    embeddings2 = model.encode(df2["target_text"].tolist(), convert_to_tensor=True, device=device) # rich_text
    hard_negatives = []
    threshold1 = 0.5
    threshold2 = 0.65
    
    for idx, row in df1.iterrows():
        emb1 = embeddings1[idx]
        similarities = util.cos_sim(emb1, embeddings2)[0].detach().cpu().numpy() # first row, as cosine_similarity returns a 2D tensor
        # Get top-k similar indices
        top_k = np.argsort(-similarities)[:top_n]

        for idx2 in top_k:
            if idx2 != idx and \
                (df_align.empty or not ((df_align["source_iri"] == row["source_iri"]) & (df_align["target_iri"] == df2.iloc[idx2]["target_iri"])).any()) \
                and similarities[idx2] > threshold1 and similarities[idx2] < threshold2:
                # If not the same index and the pair is not in the alignment and within thresholds
                source_iri = row["source_iri"]
                target_iri = df2.iloc[idx2]["target_iri"]
                hard_negatives.append({"source_iri": source_iri, "target_iri": target_iri, "match": 0.0}) # non-match
    
    df_hard_negatives = pd.DataFrame(hard_negatives)
    df_hard_negatives = df_hard_negatives.sample(n=min(num_hard_negatives, len(df_hard_negatives)), random_state=42).reset_index(drop=True)

    return df_hard_negatives
            
