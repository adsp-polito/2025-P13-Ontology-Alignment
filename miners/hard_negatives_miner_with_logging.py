import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import re
from nltk.metrics.distance import jaro_winkler_similarity
from Levenshtein import distance as lev_distance
import logging
from datetime import datetime

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

    # normalize whitespace
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

    # containment “forte”: tutti i token di A sono in B o viceversa
    return A_tokens.issubset(B_tokens) or B_tokens.issubset(A_tokens)

def token_overlap(s, t):
    """
    Compute token overlap as Intersection / max(tokens).
    Parameters:
        a (str): string 1
        b (str): string 2
    Returns:
        float: token overlap ratio
    """
    A = set(s.split())
    B = set(t.split())
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

    # Jaccard similarity: to
    # jt = jaccard_tokens(a, b)

    return jw, lev, lc, to #, jt

def apply_lexical_filters(jw: float,
           lev: float,
           lc: float,
           to: float,
           sim_emb: float,
           source_label: str,
           target_label: str,
           logger
) -> tuple:
    """Filter hard negative candidates based on similarity thresholds.

    Parameters:
        jw (float): jaro-winkler similarity
        lev (float): levenshtein similarity
        lc (float): label containment
        to (float): token overlap
        sim_emb (float): embedding cosine similarity
    Returns:
        tuple: (is_valid_hard_negative, reason) where reason is a string explaining the filtering decision
    """

    # 0 - SBERT bypass: semantically strong candidates
    # if sim_emb >= 0.78 and not lc and to < 0.4:
    #    return True

    # 1 - Label containment => false negative (parent-child or variant)
    if lc:
        logger.debug(f"Filtered [LC]: '{source_label}' - '{target_label}'")
        return False, "label_containment"

    # 2 - Token overlap too high
    if to >= 0.50:
        logger.debug(f"Filtered [TO]: '{source_label}' - '{target_label}'")
        return False, "high_token_overlap"
    
    # 3 - too similar -> possible actual match
    if jw > 0.85 or lev > 0.85:
        logger.debug(f"Filtered [SIMILAR]: '{source_label}' - '{target_label}'")
        return False, "potential_actual_match"

    # 4 - too dissimilar -> not a hard negative
    if jw < 0.40 or lev < 0.20:
        logger.debug(f"Filtered [DISSIMILAR]: '{source_label}' - '{target_label}'")
        return False, "too_dissimilar"

    # 5 - good hard negative
    logger.debug(f"Accepted [HARD_NEGATIVE]: '{source_label}' - '{target_label}'")
    return True, "valid_hard_negative"

def log_setup():
    """
    Setup logging for hard negatives mining.
    """
    log_filename = f"hard_negatives_mining_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log" # logging filename with timestamp
    logging.basicConfig( # Basic configuration for the logging system.
        level=logging.DEBUG, # This sets the threshold for what information is captured. It will capture significant milestones and errors but ignore low-level technical noise.
        format='%(asctime)s - %(levelname)s - %(message)s', # This specifies the layout of each log message, including the timestamp, severity level, and the actual message.
        handlers=[
            logging.FileHandler(log_filename), # This handler writes log messages to a file, preserving a record of the program's execution for later review.
            logging.StreamHandler() # This handler outputs log messages to the console, allowing real-time monitoring of the program's progress and issues.
        ]
    )
    logger = logging.getLogger() # Create a logger object to be used throughout the module.
    return logger


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
    
    logger = log_setup()
    logger.info("Starting hard negatives mining")
    stats = {"total_processed": 0, "label_containment": 0, "high_token_overlap": 0, 
             "potential_actual_match": 0, "too_dissimilar": 0, "valid_found": 0} # statistics tracking

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb').to(device) # pritamdeka/S-Scibert-snli-multinli-stsb
    embeddings1 = model.encode(df1["source_text"].tolist(), convert_to_tensor=True) # short_text
    embeddings2 = model.encode(df2["target_text"].tolist(), convert_to_tensor=True) # rich_text
    hard_negatives = []
    threshold_min = 0.55 # minimum similarity threshold
    threshold_max = 0.75 # maximum similarity threshold

    alignment_set = set(zip(df_align["source_iri"], df_align["target_iri"]))
    
    for idx, row in df1.iterrows():
        emb1 = embeddings1[idx]
        similarities = util.cos_sim(emb1, embeddings2)[0] # first row, as cosine_similarity returns a 2D tensor
        # Get top-k similar indices
        top_k = torch.topk(similarities, k=top_n).indices.cpu().numpy()
        # sim_values = similarities[top_k].cpu().numpy()
        # threshold_local = np.percentile(sim_values, 75) # 75th percentile of top-n similarities

        #if threshold_local < threshold_min:
        #    continue # skip concept if similarity is too low

        for idx2 in top_k:
            stats["total_processed"] += 1
            sim = similarities[idx2].item()
            if not (#sim >= threshold_local and \
                sim >= threshold_min and \
                sim <= threshold_max):
                stats["too_dissimilar"] += 1
                logger.debug(f"Filtered [OUT_OF_SIM_RANGE]: '{row['source_label']}' - '{df2.iloc[idx2]['target_label']}' with sim {sim:.4f}")
                continue

            source_label = row["source_label"]
            target_label = df2.iloc[idx2]["target_label"]
            jw, lev, lc, to = lexical_similarity(source_label, target_label)
            
            is_valid, reason = apply_lexical_filters(jw, lev, lc, to, sim, source_label, target_label, logger)

            if not is_valid:
                stats[reason] += 1
                continue

            source_iri = row["source_iri"]
            target_iri = df2.iloc[idx2]["target_iri"]

            if (source_iri, target_iri) not in alignment_set:
                # If the pair is not in the alignment and similarity is above threshold
                hard_negatives.append({"source_iri": source_iri, "target_iri": target_iri, "match": 0.0}) # non-match
                stats["valid_found"] += 1
    
    logger.info(f"Hard negatives mining completed. Stats: {stats}")
    df_hard_negatives = pd.DataFrame(hard_negatives)
    df_hard_negatives = df_hard_negatives.sample(n=min(num_hard_negatives, len(df_hard_negatives)), random_state=42).reset_index(drop=True)

    return df_hard_negatives