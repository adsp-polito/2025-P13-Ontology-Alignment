# Ontology Alignment Framework

Production-oriented framework for aligning free-text attributes (survey fields, study variables, metadata) to ontology classes using a two-stage retrieval + reranking pipeline. The repo covers dataset construction, training (bi-encoder and cross-encoder), offline preprocessing, and scalable inference.

## Table of contents
- [Key ideas](#key-ideas)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Data formats](#data-formats)
- [Training pipeline](#training-pipeline)
- [Offline preprocessing (bundle)](#offline-preprocessing-bundle)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Inference config search](#inference-config-search)
- [Notebooks and visualization](#notebooks-and-visualization)
- [Reproducibility and performance](#reproducibility-and-performance)
- [License](#license)

## Key ideas
- Two-stage inference: high-recall candidate retrieval (exact + lexical + semantic) followed by cross-encoder scoring.
- Unified ontology schema (label, description, synonyms, hierarchy context) used consistently across training and inference.
- Offline bundle for production: lexical indices + semantic embeddings stored once, reused across runs.
- Reproducible, modular training with dataset splits and Optuna tuning.

## Architecture
```
Raw ontologies (.owl/.rdf) + alignment (.rdf)
        |            
        v
  Raw loader -> unified view -> text encoding
        |                       (source SHORT_TEXT, target RICH_TEXT)
        v
  Dataset builder (positives + hard negatives + random negatives)
        v
  Train bi-encoder / cross-encoder

Target ontology only (offline):
  unified view -> offline bundle (lexical index + semantic embeddings)

Inference (online):
  attributes -> retrieval (exact/lexical/semantic) -> cross-encoder rerank -> predictions
```

## Repository layout
- `ontologies/`: ontology loaders, unified view, offline preprocessing, semantic index
- `data/`: dataset builder, source/target text encoding
- `miners/`: hard-negative mining (SBERT + lexical filters), random negatives
- `training/`: bi-encoder and cross-encoder training, Optuna optimization
- `inference/`: Stage-A retrieval and Stage-B scoring
- `testing/`: inference evaluation and model sanity tests
- `visualization/`: alignment graph visualization
- `datasets/`: sample ontologies and inference splits
- `outputs/`: recommended location for artifacts (models, bundles, predictions)

## Installation
Python dependencies are pinned in `requirements.txt`.

```bash
python -m venv OAvenv
source OAvenv/bin/activate
pip install -r requirements.txt
```

## Data formats

### 1) Raw ontology input
- Supported: OWL/RDF ontologies (local file or URL). SKOS is also supported if no OWL classes are found.
- Loader output (raw): includes `iri`, `local_name`, `label`, `parents`, `equivalent_to`, `disjoint_with`, and ontology-specific annotation columns.

### 2) Unified view schema
The unified view normalizes every class to:
- `iri`, `local_name`, `label`
- `description`
- `synonyms`
- `parents_label`
- `equivalent_to`
- `disjoint_with`

### 3) Text encoding
- **Source SHORT_TEXT** (configurable): built from label, description, synonyms, parents, equivalents, disjoints.
- **Target RICH_TEXT** (fixed): label + description + synonyms + parents + equivalents + disjoints.

### 4) Training dataset
Produced by `data/dataset_builder.py`:
- `source_iri`, `target_iri`
- `source_label`, `target_label`
- `source_text`, `target_text`
- `sample_type`: `positive`, `hard_negative`, `random_negative`
- `match`: 1.0 for positive, 0.0 for negatives

### 5) Inference input
`run_inference.py` expects a CSV with at least one column:
- `retrieval_col` (default `attribute`): used for exact + lexical retrieval
- `scoring_col` (optional, default to retrieval_col): used for semantic retrieval + cross-encoder scoring

### 6) Inference output
`run_inference.py` writes a predictions CSV with:
- `row_id` or your `--id-col`
- `attribute_text`
- `retrieval_source` (`exact`, `lexical`, `semantic`, `hybrid`, `none`)
- `num_retrieved`, `num_scored`
- `predicted_iri`, `predicted_score`
- optional `topN_iri`, `topN_score` columns if `--keep-top-n` > 0

## Training pipeline
Entry point: `training.py` with 3 modes.

### Mode 1: Full pipeline (build dataset + train)
```bash
python training.py \
  --mode full \
  --src datasets/sweet.owl \
  --tgt datasets/envo.owl \
  --align datasets/envo-sweet.rdf \
  --out-src outputs/source.csv \
  --out-tgt outputs/target.csv \
  --out-dataset outputs/alignment_dataset.csv \
  --model-type cross-encoder \
  --model-name pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb \
  --model-output-dir outputs/cross_encoder_model
```

### Mode 2: Build dataset only
```bash
python training.py \
  --mode build-dataset \
  --src datasets/sweet.owl \
  --tgt datasets/envo.owl \
  --align datasets/envo-sweet.rdf \
  --out-src outputs/source.csv \
  --out-tgt outputs/target.csv \
  --out-dataset outputs/alignment_dataset.csv
```
Outputs also include stratified splits and evaluation artifacts:
- `alignment_dataset.train.csv`, `.val.csv`, `.test.csv`
- `alignment_dataset.test.queries.csv`, `.test.gold.csv`

### Mode 3: Train only
```bash
python training.py \
  --mode train-only \
  --dataset-csv outputs/alignment_dataset.csv \
  --model-type bi-encoder \
  --model-name allenai/scibert_scivocab_uncased \
  --model-output-dir outputs/bi_encoder_model
```

### Source text configuration
Control the SHORT_TEXT features (for ablations or production constraints):
- `--src-use-description`
- `--src-use-synonyms`
- `--src-use-parents`
- `--src-use-equivalent`
- `--src-use-disjoint`

### Hyperparameter tuning (Optuna)
```bash
python training.py \
  --mode full \
  --tune --n-trials 20 \
  --model-type cross-encoder \
  --model-name pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb \
  --model-output-dir outputs/cross_encoder_model_tuned \
  --src datasets/sweet.owl \
  --tgt datasets/envo.owl \
  --align datasets/envo-sweet.rdf \
  --out-src outputs/source.csv \
  --out-tgt outputs/target.csv \
  --out-dataset outputs/alignment_dataset.csv
```

## Offline preprocessing (bundle)
Build the offline bundle for the target ontology. This is required for inference.

```bash
python build_ontology_bundle.py \
  --ont-path datasets/envo.owl \
  --prefix "http://purl.obolibrary.org/obo/ENVO_" \
  --out-csv outputs/internal_ontology.csv \
  --out-bundle outputs/offline_bundle.pkl \
  --tokenizer-name dmis-lab/biobert-base-cased-v1.1 \
  --bi-encoder-model-id allenai/scibert_scivocab_uncased
```

What is stored in the bundle:
- `label2classes` for exact match
- subword token sets + inverted index + IDF for lexical retrieval
- semantic index (bi-encoder embeddings + metadata)

Semantic embeddings are stored in a separate `.semantic_embeddings.npy` next to the bundle and loaded with memory mapping at inference time.

## Inference
Entry point: `run_inference.py`.

```bash
python run_inference.py \
  --bundle outputs/offline_bundle.pkl \
  --ontology-csv outputs/internal_ontology.csv \
  --input-csv datasets/val_split_inference.csv \
  --out-csv outputs/predictions.csv \
  --retrieval-col attribute \
  --scoring-col attribute \
  --mode hybrid \
  --cross-encoder-model-id outputs/cross_encoder_model/final_cross_encoder_model \
  --cross-top-k 50 \
  --keep-top-n 5
```

Retrieval modes:
- `lexical`: exact -> lexical; fallback to semantic only if lexical is empty.
- `hybrid`: exact -> always merge lexical + semantic with a fixed budget split.

Key knobs:
- `--retrieval-lexical-top-k`, `--retrieval-semantic-top-k`, `--retrieval-merged-top-k`
- `--hybrid-ratio-semantic` (split between lexical/semantic in hybrid)
- `--cross-top-k`, `--cross-batch-size`, `--cross-max-length`

## Evaluation
Evaluate predictions against the gold test split produced by training.

```bash
python testing/new_evaluate_inference.py \
  --test-split outputs/alignment_dataset.test.gold.csv \
  --predictions outputs/predictions.csv \
  --k 10 \
  --out-merged outputs/predictions_merged.csv \
  --out-metrics outputs/metrics.csv
```

Reported metrics (on positives only):
- Precision@1
- Hits@K
- MRR@K
- Coverage (overall)

## Inference config search
Use Optuna to search inference parameters that maximize Precision@1.

```bash
python search_inference_best_config.py \
  --bundle outputs/offline_bundle.pkl \
  --ontology-csv outputs/internal_ontology.csv \
  --input-csv datasets/val_split_inference.csv \
  --out-csv outputs/tmp_predictions.csv \
  --mode hybrid \
  --cross-encoder-model-id outputs/cross_encoder_model/final_cross_encoder_model \
  --retrieval-col source_label \
  --scoring-col source_text \
  --gt-gold-col target_iri \
  --gt-match-col match
```

## Notebooks and visualization
- `notebooks/launcher_fullPipeline_colab.ipynb`: end-to-end pipeline in Colab.
- `local_laauncher_fullPipeline.ipynb`: local launcher notebook.
- `visualization/alignment_visualization.py`: static or interactive alignment graph.

## Reproducibility and performance
- Deterministic seeds for semantic index and Optuna search.
- Tokenizers and models are loaded once per run (retrieval and scoring).
- Semantic embeddings can be memory-mapped to avoid large RAM usage.
- All offline-heavy steps are isolated from online inference.

## License
No license file is included in the repository. Add one if you plan to redistribute or publish.
