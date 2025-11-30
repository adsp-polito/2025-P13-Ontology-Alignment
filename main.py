# main.py

from __future__ import annotations
from pathlib import Path

import argparse

from ontologies.facade import ontology_loader, SourceTextConfig


def parse_args() -> argparse.Namespace:
    """
    Command-line argument parser for running the end-to-end ontology loader.

    Example usage:

        python main.py \
            --src ./data/envo.owl \
            --tgt ./data/sweet.owl \
            --src-prefix "http://purl.obolibrary.org/obo/ENVO_" \
            --tgt-prefix "http://sweetontology.net/" \
            --src-use-description \
            --src-use-synonyms \
            --out-src ./outputs/envo_text.csv \
            --out-tgt ./outputs/sweet_text.csv
    """

    parser = argparse.ArgumentParser(description="Ontology loading pipeline")

    parser.add_argument(
        "--src",
        required=True,
        help="Source ontology path/IRI",
    )
    parser.add_argument(
        "--tgt",
        required=True,
        help="target ontology path/IRI",
    )
    parser.add_argument(
        "--src-prefix",
        default=None,
        help="IRI prefix used to filter source classes (optional)",
    )
    parser.add_argument(
        "--tgt-prefix",
        default=None,
        help="IRI prefix used to filter target classes (optional)",
    )

    parser.add_argument(
        "--src-use-description",
        action="store_true",
        help="Include the description field in the source SHORT_TEXT",
    )

    parser.add_argument(
        "--src-use-synonyms",
        action="store_true",
        help="Include the synonyms in the source SHORT_TEXT",
    )

    parser.add_argument(
        "--src-use-parents",
        action="store_true",
        help="Include parent labels in the source SHORT_TEXT",
    )

    parser.add_argument(
        "--src-use-equivalent",
        action="store_true",
        help="Include equivalent class axioms (equivalent_to) in the source SHORT_TEXT",
    )

    parser.add_argument(
        "--src-use-disjoint",
        action="store_true",
        help="Include disjoint class axioms (disjoint_with) in the source SHORT_TEXT",
    )

    parser.add_argument(
        "--out-src",
        required=True,
        help="Source output CSV path",
    )
    parser.add_argument(
        "--out-tgt",
        required=True,
        help="Target output CSV path",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Build the configuration for source SHORT_TEXT
    src_text_cfg = SourceTextConfig(
        use_label=True,  # always true
        use_description=args.src_use_description,
        use_synonyms=args.src_use_synonyms,
        use_parents=args.src_use_parents,
        use_equivalent=args.src_use_equivalent,
        use_disjoint=args.src_use_disjoint,
    )

    # Execute the ontology loading pipeline
    df_src, df_tgt = ontology_loader(
        src_path_or_iri=args.src,
        tgt_path_or_iri=args.tgt,
        src_class_prefix=args.src_prefix,
        tgt_class_prefix=args.tgt_prefix,
        src_text_config=src_text_cfg,
    )

    out_src_path = Path(args.out_src)
    out_src_path.parent.mkdir(parents=True, exist_ok=True)

    out_tgt_path = Path(args.out_tgt)
    out_tgt_path.parent.mkdir(parents=True, exist_ok=True)

    # Save teh DataFrames (iri, local_name, text)
    df_src.to_csv(args.out_src, index=False)
    df_tgt.to_csv(args.out_tgt, index=False)


if __name__ == "__main__":
    main()