from __future__ import annotations
from pathlib import Path

import argparse

from ontologies.facade import ontology_loader, SourceTextConfig
from ontologies.alignment_loader import load_alignment_file
from data.dataset_builder import build_training_dataset
from visualization.alignment_visualization import visualize_alignments

def parse_args() -> argparse.Namespace:
    """
    Command-line argument parser for running the end-to-end ontology loader.

    Example usage:

        python main.py \
            --src ./datasets/envo.owl \
            --tgt ./datasets/sweet.owl \
            --align ./datasets/envo-sweet.rdf \
            --src-prefix http://purl.obolibrary.org/obo/ENVO_ \
            --tgt-prefix http://sweetontology.net/ \
            --src-use-description \
            --src-use-synonyms \
            --out-src ./outputs/envo_text.csv  \
            --out-tgt ./outputs/sweet_text.csv \
            --out-dataset ./outputs/envo_sweet_training.csv
            --visualize-alignments
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
        "--align",
        required=True,
        help="alignment RDF file path",
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

    parser.add_argument(
        "--out-dataset",
        required=False,
        help="Final training dataset output CSV path",
    )

    parser.add_argument(
        "--visualize-alignments",
        action="store_true",
        help="Visualize the alignments using a graph",
    )

    return parser.parse_args()


def main() -> None:
    # 1. Parse Arguments
    args = parse_args()

    # 2. Dispatch Commands
    
    # --- RUN BUILD ---
    # if args.command == 'build':
    #      print("--- Running Dataset Builder ---")
        
    src_text_cfg = SourceTextConfig(
        use_label=True,
        use_description=args.src_use_description,
        use_synonyms=args.src_use_synonyms,
        use_parents=args.src_use_parents,
        use_equivalent=args.src_use_equivalent,
        use_disjoint=args.src_use_disjoint,
    )

    df_src, df_tgt = ontology_loader(
        src_path_or_iri=args.src,
        tgt_path_or_iri=args.tgt,
        src_class_prefix=args.src_prefix,
        tgt_class_prefix=args.tgt_prefix,
        src_text_config=src_text_cfg,
    )

    df_alignment = load_alignment_file(args.align)
    df_training_final = build_training_dataset(df_src, df_tgt, df_alignment)

    # Create directories
    Path(args.out_src).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_tgt).parent.mkdir(parents=True, exist_ok=True)
    if args.out_dataset:
        Path(args.out_dataset).parent.mkdir(parents=True, exist_ok=True)

    # Save files
    df_src.to_csv(args.out_src, index=False)
    df_tgt.to_csv(args.out_tgt, index=False)
    
    if args.out_dataset:
        df_training_final.to_csv(args.out_dataset, index=False)
        print(f"Dataset saved to: {args.out_dataset}")

    if args.visualize_alignments:
        df_alignments = df_training_final[df_training_final["match"]==1.0]
        visualize_alignments(
            df_alignments,
            animated=True,
            source_ontology_name=Path(args.src).stem,
            target_ontology_name=Path(args.tgt).stem
        )

if __name__ == "__main__":
    main()