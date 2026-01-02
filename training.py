from __future__ import annotations
from pathlib import Path

import argparse
import pandas as pd

from ontologies.facade import ontology_loader, SourceTextConfig
from ontologies.alignment_loader import load_alignment_file
from data.dataset_builder import build_training_dataset
from visualization.alignment_visualization import visualize_alignments
from training.train import train_model

def parse_args() -> argparse.Namespace:
    """
    Command-line argument parser for running the end-to-end ontology loader.
    """

    parser = argparse.ArgumentParser(description="Ontology loading pipeline")

    parser.add_argument(
        "--src",
        default=None,
        help="Source ontology path/IRI",
    )
    parser.add_argument(
        "--tgt",
        default=None,
        help="target ontology path/IRI",
    )

    parser.add_argument(
        "--align",
        default=None,
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
        default=None,
        help="Source output CSV path",
    )
    parser.add_argument(
        "--out-tgt",
        default=None,
        help="Target output CSV path",
    )

    parser.add_argument(
        "--out-dataset",
        default=None,
        help="Final training dataset output CSV path",
    )

    parser.add_argument(
        "--visualize-alignments",
        action="store_true",
        help="Visualize the alignments using a graph",
    )

    parser.add_argument(
        "--model-type",
        choices=["bi-encoder", "cross-encoder"],
        help="Type of model to train, either bi-encoder or cross-encoder"
    )

    parser.add_argument(
        "--model-name",
        help="Pretrained model name or path from HuggingFace"
    )

    parser.add_argument(
        "--model-output-dir",
        help="Directory to save the trained model"
    )

    parser.add_argument(
    "--mode",
    choices=["full", "build-dataset", "train-only"],
    default="full",
    help="full: build dataset + train; build-dataset: only build; train-only: only train from CSV"
    )

    parser.add_argument(
        "--dataset-csv",
        default=None,
        help="Path to an existing training dataset CSV (required for --mode train-only)."
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of epochs to train the model."
    )

    return parser.parse_args()


def main() -> None:
    # 1. Parse Arguments
    args = parse_args()
    if args.mode in {"full", "build-dataset"}:
        if not (args.src and args.tgt and args.align):
            raise ValueError("--src, --tgt and --align are required for modes: full, build-dataset")

    # 2. Dispatch Commands
    if args.mode == "train-only":
        if not args.dataset_csv:
            raise ValueError("--dataset-csv is required for --mode train-only")
        if not (args.model_type and args.model_name and args.model_output_dir):
            raise ValueError("--model-type, --model-name, --model-output-dir are required for --mode train-only")

        p = Path(args.dataset_csv)
        if not p.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {p}")

        df_training_final = pd.read_csv(p)

        required_cols = {"source_text", "target_text", "match"}
        missing = required_cols - set(df_training_final.columns)
        if missing:
            raise ValueError(f"Dataset CSV missing required columns: {sorted(missing)}")
    else:   
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

        # Create directories and save files if specified
        if not (args.out_src and args.out_tgt and args.out_dataset):
            raise ValueError("--out-src, --out-tgt and --out-dataset are required for dataset-building modes")
        Path(args.out_src).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_tgt).parent.mkdir(parents=True, exist_ok=True)

        # Save files
        df_src.to_csv(args.out_src, index=False)
        df_tgt.to_csv(args.out_tgt, index=False)
    
        Path(args.out_dataset).parent.mkdir(parents=True, exist_ok=True)
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

        if args.mode == "build-dataset":
            print("Dataset built. Exiting (mode=build-dataset).")
            return
    
    if args.model_type and args.model_name and args.model_output_dir:

        print(f"--- Running {args.model_type} Training ---")

        train_model(
            df_training=df_training_final,
            model_type=args.model_type,
            model_name=args.model_name,
            output_dir=args.model_output_dir,
            num_epochs=args.num_epochs
        )

        print(f"Model saved to: {args.model_output_dir}")

if __name__ == "__main__":
    main()