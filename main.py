# main.py

from __future__ import annotations
from pathlib import Path

import argparse

from ontologies.facade import ontology_loader, SourceTextConfig
from ontologies.alignment_loader import load_alignment_file
from data.dataset_builder import build_training_dataset

from compile_inverted_index import compile_inverted_index
from training.trainer import run_trainer
from inference.matcher_inverted_index import run_pipeline

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
   
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # 2. INDEX COMMAND (Offline Preprocessing)
    parser_index = subparsers.add_parser('index', help='Step 2: Compile Inverted Index & Exact Match Map')
    parser_index.add_argument('--ontology', required=True, help="Path to Target Ontology (.owl)")
    parser_index.add_argument('--out-index', required=True, help="Path to save .pkl index")

    # 3. TRAIN COMMAND (Fine-Tuning)
    parser_train = subparsers.add_parser('train', help='Step 3: Fine-Tune BERT Model')
    parser_train.add_argument('--train-file', required=True, help="Path to training CSV")
    parser_train.add_argument('--model-out', required=True, help="Directory to save model")
    parser_train.add_argument('--epochs', type=int, default=6)

    # 4. INFER COMMAND (The Pipeline)
    parser_infer = subparsers.add_parser('infer', help='Step 4: Run Inference Pipeline')
    parser_infer.add_argument('--input-csv', required=True, help="CSV with attributes to align")
    parser_infer.add_argument('--index-file', required=True, help="Path to .pkl index")
    parser_infer.add_argument('--model-path', required=True, help="Path to trained model")
    parser_infer.add_argument('--out-results', required=True, help="Output CSV results")

    return parser.parse_args()


def main() -> None:
    # 1. Parse Arguments
    args = parse_args()

    # 2. Dispatch Commands
    
    # --- RUN BUILD ---
    if args.command == 'build':
        print("--- Running Dataset Builder ---")
        
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


# --- RUN INDEX ---
    elif args.command == 'index':
        print(f"--- [Step 2] Compiling Inverted Index ---")
        compile_inverted_index(args.ontology, args.out_index)

    # --- RUN TRAIN ---
    elif args.command == 'train':
        print(f"--- [Step 3] Training BERT Model ---")
        run_trainer(args.train_file, args.model_out, args.epochs)

    # --- RUN INFER ---
    elif args.command == 'infer':
        print(f"--- [Step 4] Running Inference Pipeline ---")
        run_pipeline(args.input_csv, args.index_file, args.model_path, args.out_results)

if __name__ == "__main__":
    main()