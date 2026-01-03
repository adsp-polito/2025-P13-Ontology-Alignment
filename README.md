# 2025-P13-Ontology-Alignment

# Ontology Alignment Framework

A modular and production-oriented framework for **ontology alignment**, designed to align free-text attributes (e.g. study variables, survey fields, metadata) to ontology classes using a **two-stage architecture**:

1. **High-recall candidate retrieval** (Stage A)
2. **High-precision cross-encoder scoring** (Stage B)

The framework supports dataset construction, model training, offline preprocessing, and scalable inference, with a strong focus on reproducibility and deployment constraints.

---

## Overview

Ontology alignment is framed as a **binary classification / ranking problem** between:
- a *source text* (attribute, variable, or field name)
- a *target ontology class* (label + semantic context)

This repository provides:
- tools to **build training datasets** with positives and structured negatives
- **bi-encoder and cross-encoder training pipelines**
- an **offline bundle** for fast inference (lexical + semantic indices)
- a **production-ready inference pipeline** with caching and batching

The design follows a strict separation between:
- **offline steps** (heavy, one-time computation)
- **online inference** (fast, memory-efficient, reusable components)

---

## Architecture Overview

![Architecture diagram](docs/images/architecture.png)

**High-level flow:**
1. Ontologies are normalized into a unified internal representation
2. Training datasets are built from reference alignments + negative mining
3. Models are trained (bi-encoder or cross-encoder)
4. Target ontologies are preprocessed into an offline bundle
5. Inference runs in two stages: retrieval → scoring

---

## Key Concepts

### Unified View
All ontologies are normalized into a common schema containing:
- `iri`
- `label`
- `synonyms`
- hierarchical and logical context
- derived textual representations

### SHORT_TEXT vs RICH_TEXT
- **SHORT_TEXT** (source): compact, configurable text built from labels and optional metadata  
- **RICH_TEXT** (target): richer semantic description used for scoring and embedding

### Two-Stage Inference
- **Stage A (Retrieval):** maximize recall, cheap heuristics
- **Stage B (Scoring):** maximize precision, expensive neural model

---

## Candidate Retrieval — Stage A

![Retrieval pipeline](docs/images/retrieval_pipeline.png)

Stage A produces a shortlist of candidate ontology classes for each attribute.

### Retrieval Sources
1. **Exact Match**
   - Soft-normalized label and synonym matching
2. **Lexical Retrieval**
   - Inverted index + IDF-weighted subword overlap
3. **Semantic Retrieval**
   - Bi-encoder embeddings + cosine similarity
4. **Hybrid Mode**
   - Controlled merge of lexical and semantic candidates

### Design Principles
- No model or tokenizer is reloaded per attribute
- Bi-encoder is loaded once and reused
- Semantic embeddings are memory-mapped when possible
- Retrieval scores are *source-specific* and not directly compared

---

## Cross-Encoder Scoring — Stage B

Stage B re-ranks candidates using a HuggingFace **cross-encoder** trained for sequence classification.

Given:
- an attribute text
- a shortlist of candidate IRIs
- a mapping `iri → RICH_TEXT`

The cross-encoder outputs a probability-like alignment score.

### Features
- Batched scoring
- Cached tokenizer and model
- Supports binary and multi-class heads
- Clean separation from retrieval logic

---

## Dataset Construction

Training datasets are built by combining:

- **Positive samples**
  - From reference alignment files
- **Hard negatives**
  - Mined using semantic similarity (SBERT)
  - Filtered with lexical heuristics to avoid false negatives
- **Random negatives**
  - Uniformly sampled non-aligned pairs

Each sample includes a `sample_type` field:
- `positive`
- `hard_negative`
- `random_negative`

This allows:
- debugging
- ablation studies
- curriculum or weighted training strategies

---

## Training

Training is orchestrated via a single entry point supporting **three modes**:

1. **Full pipeline**
   - Load ontologies
   - Build dataset
   - Train model
2. **Dataset-only**
   - Build and inspect training CSV
3. **Train-only**
   - Train from an existing dataset CSV

Supported models:
- **Bi-encoder** (for semantic retrieval)
- **Cross-encoder** (for final scoring)

Splits are **stratified** to preserve label balance.

---

## Offline Preprocessing

Target ontologies are preprocessed once into an **offline bundle** containing:

- lexical structures:
  - `label2classes`
  - inverted index
  - IDF statistics
- semantic index:
  - bi-encoder embeddings
  - metadata (model id, normalization, dimensions)

Semantic embeddings are optionally stored in a separate `.npy` file and loaded via memory-mapping for scalability.

---

## Inference

Inference is fully decoupled from training and dataset construction.

Typical flow:
1. Load offline bundle
2. Initialize `CandidateRetriever`
3. Retrieve candidates (Stage A)
4. Score candidates with `CrossEncoderScorer` (Stage B)
5. Return best prediction or ranked list

Both **single-attribute** and **batch inference** are supported.

---

## Repository Structure

├── ontologies/
│   ├── raw_loader.py
│   ├── unified_view.py
│   ├── facade.py
│   ├── offline_preprocessing.py
│   └── semantic_index.py
│
├── data/
│   ├── dataset_builder.py
│   └── text_encoding.py
│
├── miners/
│   ├── hard_negatives_miner.py
│   └── random_negatives_miner.py
│
├── training/
│   ├── train.py
│   ├── bi_encoder_training.py
│   ├── cross_encoder_training.py
│   └── utils.py
│
├── inference/
│   ├── retrieval.py
│   └── scoring.py
│
├── notebooks/
│   └── launcher_dataset&training_colab.ipynb
│
└── docs/ ??
    └── images/

---

## Design Philosophy

- **Separation of concerns**
  - dataset ≠ training ≠ inference
- **Offline first**
  - heavy computation is pushed offline
- **Production constraints**
  - caching, batching, memory-mapping
- **Reproducibility**
  - deterministic seeds
  - explicit configuration objects
- **Inspectability**
  - CSV-based datasets
  - explicit sample types
  - transparent heuristics

---

## Project Status

The framework is **feature-complete** for:
- dataset construction
- model training
- offline preprocessing
- scalable inference

Ongoing work focuses on:
- extensive testing
- model training and evaluation
- downstream integration

---

## License

Specify your license here. !!!!!!

---

## Acknowledgements

???