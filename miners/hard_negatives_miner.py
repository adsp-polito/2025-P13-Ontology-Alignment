import pandas as pd
import re
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import spmatrix

def normalize(
        text: str
) -> str:
    """Normalize text for lexical similarity calculations.

    Parameters:
        text (str): input text

    Returns:
        str: normalized text
    """
    if pd.isna(text): return ""

    # split camelCase: "PrecipitableWater" -> "precipitable water"
    text = re.sub(r'(?<!^)(?=[A-Z])', ' ', text)
    # substitute non-alphanumeric characters with space: wave-number -> wave number
    text = re.sub(r'[^a-zA-Z0-9]', " ", text)
    # normalize whitespace: wave    number -> wave number
    text = re.sub(r'\s+', ' ', text)

    return text.lower().strip()


def normalize_synonyms(
        synonyms: str
) -> list[str]:
    """
    Normalize synonyms string into a list of normalized synonyms.

    Parameters:
        synonyms (str): synonyms string separated by "|"

    Returns:
        list[str]: list of normalized synonyms
    """
    if pd.isna(synonyms):
        return []
    return [normalize(s) for s in synonyms.split("|") if s.strip()]


def is_bad_hard_negative(
        s1: str,
        s2: str,
        tfidf_matrix: spmatrix,
        label2idx: dict[str, int],
        source_synonyms: list[str],
        target_synonyms: list[str]
) -> bool:
    """
    Returns True if the two strings are too similar (Risk of False Negative) or too dissimilar.

    Parameters:
        s1 (str): first string
        s2 (str): second string,
        tfidf_matrix (spmatrix): TF-IDF matrix of all labels
        label2idx (dict[str, int]): mapping from label to index in the TF-IDF matrix
        source_synonyms (list[str]): list of synonyms for the source label
        target_synonyms (list[str]): list of synonyms for the target label

    Returns:
        bool: True if the strings are too similar, False otherwise
    """
    set1, set2 = set(s1.split()), set(s2.split())
    if s1 in s2 or s2 in s1:
        return True

    # 1. Label Containment
    # Examples: "cancer" in "lung cancer", "heart disease" in "disease of the heart"
    if not set1 or not set2:
        return False

    if set1.issubset(set2) or set2.issubset(set1):
        return True

    s1_compact = s1.replace(" ", "")
    s2_compact = s2.replace(" ", "")
    # Example: "mudflat" vs "mud flat"
    if s1_compact in s2_compact or s2_compact in s1_compact:
        return True
    
    # 2. Synonym Containment
    if s1 in target_synonyms or s2 in source_synonyms:
        return True

    # TF-IDF
    idx1 = label2idx.get(s1)
    idx2 = label2idx.get(s2)

    vec1 = tfidf_matrix[idx1]
    vec2 = tfidf_matrix[idx2]

    sim = cosine_similarity(vec1, vec2)[0][0]

    # too similar
    if sim >= 0.8:
        return True

    return False


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

    top_k_count = min(int(top_n), len(df2))
    if top_k_count <= 0:
        return pd.DataFrame(columns=["source_iri", "target_iri", "match"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb').to(device) # or pritamdeka/S-Scibert-snli-multinli-stsb

    embeddings1 = model.encode(df1["source_text"].tolist(), show_progress_bar=True, convert_to_tensor=True) # short_text
    embeddings2 = model.encode(df2["target_text"].tolist(), show_progress_bar=True, convert_to_tensor=True) # rich_text

    # util.semantic_search is optimized for large-scale search
    # Returns a list of lists of hits for each embedding in embeddings1, where each hit is a dict with 'corpus_id' and 'score'
    hits_list = util.semantic_search(embeddings1, embeddings2, top_k=top_n)

    hard_negatives = []
    threshold_min, threshold_max = 0.65, 0.80 # minimum and maximum similarity threshold

    # Precompute alignment set for fast lookup
    alignment_set = set(zip(df_align["source_iri"], df_align["target_iri"]))

    # Precompute normalized labels for lexical similarity filtering
    df1["norm_label"] = df1["source_label"].apply(normalize)
    df2["norm_label"] = df2["target_label"].apply(normalize)
    df1["norm_synonyms"] = df1["source_synonyms"].apply(normalize_synonyms)
    df2["norm_synonyms"] = df2["target_synonyms"].apply(normalize_synonyms)

    # TF-IDF initalization
    all_labels = df1["norm_label"].tolist() + df2["norm_label"].tolist()
    tfidf_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1)
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_labels)
    label2idx = {label: idx for idx, label in enumerate(all_labels)}

    for idx1, hits in enumerate(hits_list):
        src_row = df1.iloc[idx1]
        src_norm = src_row["norm_label"]
        src_iri = src_row["source_iri"]

        for hit in hits:
            score = hit['score']
            idx2 = hit['corpus_id']
            tgt_row = df2.iloc[idx2]
            tgt_iri = tgt_row["target_iri"]

            # If the pair is not in the alignment and similarity is outside the desired range, skip it
            if ((src_iri, tgt_iri) in alignment_set) or (score < threshold_min or score > threshold_max):
                continue

            # Lexical Similarity Filtering
            if is_bad_hard_negative(src_norm, tgt_row["norm_label"], tfidf_matrix, label2idx, src_row["norm_synonyms"], tgt_row["norm_synonyms"]):
                continue

            hard_negatives.append({
                "source_iri": src_iri,
                "target_iri": tgt_iri,
                "sample_type": "hard_negative",
                "match": 0.0
            })

    df_hard_negatives = pd.DataFrame(hard_negatives)
    df_hard_negatives = df_hard_negatives.sample(n=min(num_hard_negatives, len(df_hard_negatives)), random_state=42).reset_index(drop=True)

    return df_hard_negatives