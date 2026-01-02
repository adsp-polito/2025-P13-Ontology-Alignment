from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from transformers import AutoTokenizer

from ontologies.raw_loader import ontology_classes_to_dataframe
from ontologies.unified_view import build_unified_view
from ontologies.offline_preprocessing import (
    build_offline_bundle_from_unified,
    save_offline_bundle,
)
from data.text_encoding import build_target_text

from ontologies.semantic_index import SemanticIndexConfig, build_semantic_index_from_unified


def parse_args() -> argparse.Namespace:
    """
    Command-line interface for building:
      1) the internal ontology CSV (iri, local_name, label, text)
      2) the offline_bundle.pkl for candidate selection and exact match.

    This script is meant to be used by Repertorio to "register" or update
    their internal ontology in an offline step.
    """
    parser = argparse.ArgumentParser(
        description="Build internal ontology CSV and offline bundle from an OWL/RDF file."
    )

    parser.add_argument(
    "--export-csv",
    default=None,
    help=(
        "Optional path to a CSV already in the final format. "
        "If provided, the ontology OWL/RDF will NOT be loaded."
    ),
    )

    parser.add_argument(
        "--ont-path",
        default=None,
        help="Internal ontology path/IRI (OWL/RDF).",
    )

    parser.add_argument(
        "--prefix",
        default=None,
        help=(
            "IRI prefix used to filter target classes (optional), "
            "e.g. 'http://purl.obolibrary.org/obo/ENVO_'."
        ),
    )

    parser.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV path for the internal ontology (iri, local_name, label, text).",
    )

    parser.add_argument(
        "--out-bundle",
        required=True,
        help="Output path for the offline bundle (pickle file).",
    )

    parser.add_argument(
        "--tokenizer-name",
        default="dmis-lab/biobert-base-cased-v1.1",
        help=(
            "Hugging Face tokenizer name to use for subword tokenization. "
            "Must match the tokenizer used by the cross-encoder."
        ),
    )

    parser.add_argument(
    "--bi-encoder-model-id",
    required=True,
    help="HuggingFace model id for the bi-encoder used to build semantic embeddings.",
    )

    parser.add_argument(
        "--semantic-batch-size",
        type=int,
        default=64,
        help="Batch size for bi-encoder embedding computation.",
    )

    parser.add_argument(
        "--semantic-max-length",
        type=int,
        default=256,
        help="Max token length for bi-encoder tokenization.",
    )

    parser.add_argument(
        "--no-semantic-normalize",
        action="store_true",
        help="Disable L2 normalization for semantic embeddings.",
    )

    return parser.parse_args()

def _validate_export_view_columns(df: pd.DataFrame) -> None:
    required = {"iri", "local_name", "label", "synonyms", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Export CSV must come from this framework and contain columns: "
            f"{sorted(required)}. Missing: {sorted(missing)}"
        )

def main() -> None:
    args = parse_args()

    ont_path_or_iri: str = args.ont_path
    class_prefix: Optional[str] = args.prefix

    out_csv = Path(args.out_csv)
    out_bundle_path = Path(args.out_bundle)

    # Ensure output directories exist
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_bundle_path.parent.mkdir(parents=True, exist_ok=True)

    df_uni: pd.DataFrame

    if args.export_csv is not None:
        # CASE A: export view already exists
        export_csv_path = Path(args.export_csv)
        print(f"[INFO] Loading export view from CSV: {export_csv_path}")
        df_uni = pd.read_csv(export_csv_path)
        _validate_export_view_columns(df_uni)
        print(f"[INFO] Loaded export view with {len(df_uni)} rows.")

        # Ensure minimal schema and save internal CSV also in CASE A
        required_min = {"iri", "local_name", "label", "text", "synonyms"}
        missing_min = required_min - set(df_uni.columns)
        if missing_min:
            raise ValueError(f"Export CSV missing required minimal columns: {sorted(missing_min)}")

        df_export = df_uni[["iri","local_name","label","synonyms","text"]].copy()
        print(f"[INFO] Saving internal ontology CSV to: {out_csv}")
        df_export.to_csv(out_csv, index=False)
        print("[INFO] Internal CSV saved successfully.")
    else:
        # CASE B: build unified/export view from ontology OWL/RDF
        if args.ont_path is None:
            raise ValueError("You must provide either --export-csv or --ont-path.")

        # 1) Load the internal ontology as raw DataFrame
        print(f"[INFO] Loading internal ontology from: {ont_path_or_iri}")
        df_raw: pd.DataFrame = ontology_classes_to_dataframe(
            path=ont_path_or_iri,
            class_iri_prefix=class_prefix,
        )

        print(f"[INFO] Loaded {len(df_raw)} classes from internal ontology.")

        # 2) Build unified view for the internal ontology
        print("[INFO] Building unified view for internal ontology...")
        df_uni = build_unified_view(df_raw)
        print(f"[INFO] Unified view has {len(df_uni)} rows.")

        # 3) Build the text representation (RICH_TEXT) for each class
        print("[INFO] Building text representation...")
        df_uni["text"] = df_uni.apply(build_target_text, axis=1)

        # Reduce to the minimal schema
        df_export = df_uni[["iri", "local_name", "label", "synonyms", "text"]].copy()

        # 4) Save the internal ontology CSV
        print(f"[INFO] Saving internal ontology CSV to: {out_csv}")
        df_export.to_csv(out_csv, index=False)
        print("[INFO] Internal CSV saved successfully.")

    # 5) Load the tokenizer to be used for subword tokenization
    print(f"[INFO] Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # 6) Build the offline bundle from the unified view (lexical structures)
    print("[INFO] Building offline bundle (class_data, label2classes, T, inverted_index, idf)...")
    offline_bundle = build_offline_bundle_from_unified(df_uni, tokenizer)

    # 6b) Build semantic index (bi-encoder embeddings) - ALWAYS
    print(f"[INFO] Building semantic index with bi-encoder: {args.bi_encoder_model_id}")
    cfg = SemanticIndexConfig(
        model_id=args.bi_encoder_model_id,
        batch_size=args.semantic_batch_size,
        max_length=args.semantic_max_length,
        normalize=not args.no_semantic_normalize,
    )
    semantic_index = build_semantic_index_from_unified(df_uni, cfg)
    offline_bundle["semantic_index"] = semantic_index

    # 7) Save the offline bundle to disk
    print(f"[INFO] Saving offline bundle to: {out_bundle_path}")
    save_offline_bundle(offline_bundle, out_bundle_path)
    print("[INFO] Offline bundle saved successfully.")

    print("[DONE] Internal ontology CSV and offline bundle are ready.")


if __name__ == "__main__":
    main()