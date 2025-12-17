from __future__ import annotations

from typing import Dict, Any, Iterable
from collections import defaultdict
from pathlib import Path
import math
import pickle
import re

import pandas as pd

from ontologies.semantic_index import save_semantic_embeddings
from ontologies.semantic_index import load_semantic_embeddings

def normalize_label(text: str) -> str:
    """
    Soft normalization used both for building Ω(c) and for exact-match lookup.

    Steps:
      - convert to lowercase
      - strip leading/trailing whitespace
      - collapse multiple internal whitespace into a single space

    No aggressive punctuation removal, no stemming.
    """
    if text is None:
        return ""

    # Lowercase + strip
    s = str(text).lower().strip()
    if not s:
        return ""

    # Collapse multiple whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def _split_synonyms_field(synonyms: str) -> Iterable[str]:
    """
    Utility to split the 'synonyms' field from the unified view.

    The unified view typically uses ' | ' as the separator between synonyms.
    This helper handles the standard case as well as variants with or without surrounding spaces.
    """
    if not isinstance(synonyms, str) or not synonyms.strip():
        return []

    parts = [p.strip() for p in synonyms.split("|")]
    return [p for p in parts if p]

def build_class_data(df_tgt_uni: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Costruisce Ω(c) a partire dalla unified view del target.

    Per ogni classe (riga di df_tgt_uni) estrae:
      - iri
      - label principale
      - tutti i sinonimi (campo 'synonyms' pipe-separated)

    e costruisce una lista di stringhe normalizzate (soft) che rappresentano
    tutte le forme testuali per cui, se un attributo di studio coincide
    dopo normalizzazione, consideriamo un exact-match lessicale.
    Output:
        class_data[c_iri] = {
            "labels": [norm_label_1, norm_label_2, ...],
        }
    """
    if "iri" not in df_tgt_uni.columns:
        raise ValueError("df_tgt_uni must contain an 'iri' column.")
    if "label" not in df_tgt_uni.columns:
        raise ValueError("df_tgt_uni must contain a 'label' column.")
    if "synonyms" not in df_tgt_uni.columns:
        df_tgt_uni = df_tgt_uni.copy()
        df_tgt_uni["synonyms"] = ""

    class_data: Dict[str, Dict[str, Any]] = {}

    for row in df_tgt_uni.itertuples(index=False):
        iri = getattr(row, "iri")
        raw_label = getattr(row, "label", "") or ""
        raw_synonyms = getattr(row, "synonyms", "") or ""

        label_candidates = []

        # True label
        if raw_label:
            label_candidates.append(raw_label)

        # Synonyms
        for syn in _split_synonyms_field(raw_synonyms):
            label_candidates.append(syn)

        # Soft normalization + deduplication
        norm_labels = []
        seen = set()
        for lbl in label_candidates:
            norm = normalize_label(lbl)
            if norm and norm not in seen:
                seen.add(norm)
                norm_labels.append(norm)

        class_data[iri] = {
            "labels": norm_labels,
        }

    return class_data


def build_label2classes(class_data: Dict[str, Dict[str, Any]]) -> Dict[str, set]:
    """
    Builds the dictionary for exact match:

        label2classes[norm_label] -> set of class_iri that use that label form.

    Note: this assumes that class_data[c_iri]["labels"] already contains
    label forms normalized through `normalize_label`.
    """
    label2classes: Dict[str, set] = defaultdict(set)

    for c_iri, data in class_data.items():
        labels = data.get("labels", []) or []
        for lbl in labels:
            if not lbl:
                continue
            label2classes[lbl].add(c_iri)

    # Convert to a regular dict to avoid serializing the defaultdict
    return dict(label2classes)


def build_token_sets(
    class_data: Dict[str, Dict[str, Any]],
    tokenizer,
) -> Dict[str, set]:
    """
    Builds T(c) for each class, using the tokenizer of the cross-encoder
    (e.g., BioBERT).

    It uses the forms in class_data[c_iri]["labels"], which are already
    normalized using the soft normalization pipeline (lowercase, cleaned
    whitespace). This ensures symmetry between the preprocessing of study
    attributes and ontology class labels.

    Output:
        T = {
            c_iri: {token_1, token_2, ...},
            ...
        }
    """
    T: Dict[str, set] = {}

    for c_iri, data in class_data.items():
        labels = data.get("labels", []) or []
        token_set: set = set()
        for lbl in labels:
            if not lbl:
                continue
            # tokenizer.tokenize returns a subword list
            tokens = tokenizer.tokenize(lbl)
            for t in tokens:
                token_set.add(t)

        T[c_iri] = token_set

    return T


def build_inverted_index(T: Dict[str, set]) -> Dict[str, set]:
    """
    Builds the inverted index:

        inverted_index[token] -> set of class_iri that contain that token.
    """
    inverted_index: Dict[str, set] = defaultdict(set)

    for c_iri, tokens in T.items():
        for t in tokens:
            inverted_index[t].add(c_iri)

    return dict(inverted_index)


def compute_idf(inverted_index: Dict[str, set], num_classes: int) -> Dict[str, float]:
    """
    Computes the IDF for each token:

        idf[t] = log10(num_classes / df(t))

    where df(t) is the number of classes in which token t appears.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    idf: Dict[str, float] = {}

    for token, cls_set in inverted_index.items():
        df = len(cls_set)
        if df <= 0:
            continue
        idf[token] = math.log10(num_classes / df)

    return idf


def build_offline_bundle_from_class_data(
    class_data: Dict[str, Dict[str, Any]],
    tokenizer,
) -> Dict[str, Any]:
    """
    Main entrypoint starting from class_data (Ω(c)).

    We assume that class_data has the structure:

        class_data[c_iri] = {
            "labels": [...],  # already soft-normalized forms
        }

    This function builds:
      - label2classes
      - T(c)
      - inverted_index
      - idf
      - num_classes

    and returns them as a single dictionary called offline_bundle.
    """
    num_classes = len(class_data)

    label2classes = build_label2classes(class_data)
    T = build_token_sets(class_data, tokenizer)
    inverted_index = build_inverted_index(T)
    idf = compute_idf(inverted_index, num_classes)

    offline_bundle: Dict[str, Any] = {
        "class_data": class_data,
        "label2classes": label2classes,
        "T": T,
        "inverted_index": inverted_index,
        "idf": idf,
        "num_classes": num_classes,
    }

    return offline_bundle


def build_offline_bundle_from_unified(
    df_tgt_uni: pd.DataFrame,
    tokenizer,
) -> Dict[str, Any]:
    """
    High-level entrypoint that starts directly from the target unified view
    (i.e., the DataFrame produced by build_unified_view for the target ontology).

    This matches the typical use-case where:
      - df_tgt_uni is already in memory (notebook / training pipeline),
      - and you want to build the offline_bundle for the target ontology.

    Steps:
      1. build class_data (Ω(c)) via build_class_data
      2. build the offline_bundle using build_offline_bundle_from_class_data
    """
    class_data = build_class_data(df_tgt_uni)
    offline_bundle = build_offline_bundle_from_class_data(class_data, tokenizer)
    return offline_bundle


def save_offline_bundle(offline_bundle: Dict[str, Any], path: str | Path) -> None:
    """
    Saves the offline_bundle to disk using pickle.

    If offline_bundle contains semantic_index with in-memory embeddings,
    we save embeddings separately and store only metadata + embeddings_path in pickle.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if "semantic_index" in offline_bundle and isinstance(offline_bundle["semantic_index"], dict):
        sem = offline_bundle["semantic_index"]
        # If embeddings are present, move them to disk next to bundle
        if "embeddings" in sem and sem["embeddings"] is not None:
            offline_bundle["semantic_index"] = save_semantic_embeddings(
                sem,
                bundle_path=path,
            )

    with path.open("wb") as f:
        pickle.dump(offline_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_offline_bundle(
    path: str | Path,
    *,
    load_semantic_embeddings: bool = True,
    mmap: bool = True,
) -> Dict[str, Any]:
    """
    Loads an offline_bundle previously saved with save_offline_bundle.

    If semantic embeddings were saved separately and requested,
    they are loaded back into semantic_index['embeddings'].

    Args:
        path: path to the offline bundle pickle.
        load_semantic_embeddings: whether to load semantic embeddings (.npy).
        mmap: if True, load embeddings with numpy memmap (recommended).

    Returns:
        offline_bundle dict with semantic_index['embeddings'] available.
    """
    path = Path(path)

    with path.open("rb") as f:
        bundle = pickle.load(f)

    if (
        load_semantic_embeddings
        and "semantic_index" in bundle
        and isinstance(bundle["semantic_index"], dict)
    ):
        sem = bundle["semantic_index"]
        if "embeddings_path" in sem and "embeddings" not in sem:
            bundle["semantic_index"] = load_semantic_embeddings(
                sem,
                bundle_path=path,
                mmap=mmap,
            )

    return bundle