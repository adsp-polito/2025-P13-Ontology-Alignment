import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import re
from nltk.metrics.distance import jaro_winkler_similarity
from Levenshtein import distance as lev_distance

def normalize(
        text: str
) -> str:
    """Normalize text for lexical similarity calculations.

    Parameters:
        text (str): input text
    Returns:
        str: normalized text
    """
    if pd.isna(text):
        return ""

    # split camelCase: "PrecipitableWater" -> "precipitable water"
    text = re.sub(r'(?<!^)(?=[A-Z])', ' ', text)
    # substitute non-alphanumeric characters with space: wave-number -> wave number
    text = re.sub(r'[^a-zA-Z0-9]', " ", text)

    text = text.lower().strip()

    # normalize whitespace: wave    number -> wave number
    text = re.sub(r'\s+', ' ', text)
    return text

def label_containment(a: str, b: str) -> bool:
    """
    True if one label contains the other as a complete phrase,
    not just as a substring.
    Avoids false positives when similarity is superficial.
    Parameters:
        a (str): string 1
        b (str): string 2
    Returns:
        bool: True if one label contains the other
    """

    # token sets
    A_tokens = set(a.split())
    B_tokens = set(b.split())

    if not A_tokens or not B_tokens:
        return False

    # hard containment: all tokens of A are in B or viceversa
    return A_tokens.issubset(B_tokens) or B_tokens.issubset(A_tokens)

def token_overlap(
        a: str,
        b: str
)-> float:
    """
    Compute token overlap as Intersection / max(tokens).
    Parameters:
        a (str): string 1
        b (str): string 2
    Returns:
        float: token overlap ratio
    """
    A = set(a.split())
    B = set(b.split())
    if not A or not B:
        return 0
    return len(A & B) / max(len(A), len(B))
    
def lexical_similarity(
        a: str,
        b: str
) -> tuple:
    """Compute lexical similarities between two strings: Jaro-Winkler, Levenshtein, Label Containment, and Token Overlap.

    Parameters:
        a (str): string 1
        b (str): string 2
    Returns:
        tuple: (jaro-winkler similarity, levenshtein similarity, label containment, token overlap)
    """
    a = normalize(a)
    b = normalize(b)

    jw = jaro_winkler_similarity(a, b)
    # Levenshtein similarity
    lev = 1 - (lev_distance(a, b) / max(len(a), len(b), 1))
    # Label containment
    lc = label_containment(a, b)
    # Tokens overlap
    to = token_overlap(a, b)

    return jw, lev, lc, to

def apply_lexical_filters(
        jw: float,
        lev: float,
        lc: float,
        to: float
) -> bool:
    """Filter hard negative candidates based on similarity thresholds.

    Parameters:
        jw (float): jaro-winkler similarity
        lev (float): levenshtein similarity
        lc (float): label containment
        to (float): token overlap
        sim_emb (float): embedding cosine similarity
    Returns:
        bool: True if the candidate is a valid hard negative, False otherwise
    """

    # 1 - Label containment => false negative (parent-child or variant)
    if lc:
        return False

    # 2 - Token overlap too high
    if to >= 0.50:
        return False
    
    # 3 - too similar -> possible actual match
    if jw > 0.85 or lev > 0.85:
        return False

    # 4 - too dissimilar -> not a hard negative
    if jw < 0.40 or lev < 0.20:
        return False

    # 5 - good hard negative
    return True


def generate_hard_negatives(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        df_align: pd.DataFrame,
        num_hard_negatives: int = 100,
        top_n: int = 20,
) -> pd.DataFrame:
    """Generate hard negatives using SBERT embeddings and lexical similarity filtering.

    Parameters:
        df1 (pd.DataFrame): fist dataframe
        df2 (pd.DataFrame): second dataframe
        df_align (pd.DataFrame): alignment dataframe to avoid sampling alignments
        num_hard_negatives (int, optional): number of hard negatives to generate. Defaults to 100.
        top_n (int, optional): number of top similar candidates to consider. Defaults to 5.

    Returns:
        pd.DataFrame: hard negatives dataframe
    """
    if num_hard_negatives <= 0:
        return pd.DataFrame(columns=["source_iri", "target_iri", "match"])

    if df1.empty or df2.empty:
        return pd.DataFrame(columns=["source_iri", "target_iri", "match"])

    top_n = int(top_n)
    if top_n <= 0:
        return pd.DataFrame(columns=["source_iri", "target_iri", "match"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb').to(device) # or pritamdeka/S-Scibert-snli-multinli-stsb
    embeddings1 = model.encode(df1["source_text"].tolist(), convert_to_tensor=True) # short_text
    embeddings2 = model.encode(df2["target_text"].tolist(), convert_to_tensor=True) # rich_text
    top_k_count = min(top_n, len(df2))
    if top_k_count <= 0:
        return pd.DataFrame(columns=["source_iri", "target_iri", "match"])
    hard_negatives = []
    threshold_min = 0.55 # minimum similarity threshold
    threshold_max = 0.75 # maximum similarity threshold

    alignment_set = set(zip(df_align["source_iri"], df_align["target_iri"]))
    
    for idx, row in df1.iterrows():
        emb1 = embeddings1[idx]
        similarities = util.cos_sim(emb1, embeddings2)[0] # first row, as cosine_similarity returns a 2D tensor
        # Get top-k similar indices
        top_k_idx = torch.topk(similarities, k=top_k_count).indices.cpu().numpy()

        for idx2 in top_k_idx:
            sim = similarities[idx2].item()
            if not (sim >= threshold_min and sim <= threshold_max): # too dissimilar
                continue

            source_label = row["source_label"]
            target_label = df2.iloc[idx2]["target_label"]
            jw, lev, lc, to = lexical_similarity(source_label, target_label)
            is_valid = apply_lexical_filters(jw, lev, lc, to)

            if not is_valid:
                continue

            source_iri = row["source_iri"]
            target_iri = df2.iloc[idx2]["target_iri"]

            if (source_iri, target_iri) not in alignment_set:
                # If the pair is not in the alignment and similarity is above threshold
                hard_negatives.append({"source_iri": source_iri, "target_iri": target_iri, "match": 0.0}) # non-match
    
    df_hard_negatives = pd.DataFrame(hard_negatives)
    df_hard_negatives = df_hard_negatives.sample(n=min(num_hard_negatives, len(df_hard_negatives)), random_state=42).reset_index(drop=True)

    return df_hard_negatives
            
