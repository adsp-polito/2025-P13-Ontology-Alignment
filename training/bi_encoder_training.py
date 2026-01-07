from __future__ import annotations
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.losses import OnlineContrastiveLoss
from sentence_transformers.util import cos_sim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score
import pandas as pd
import os
import wandb
import optuna
import torch
from .utils import convert_df_to_dataset
from typing import Optional
from pathlib import Path


def evaluate_bi_encoder(
        model: SentenceTransformer,
        df: pd.DataFrame,
        threshold: float = 0.5
) -> dict:
    """
    Evaluate bi-encoder model on val/test dataframe.

    Parameters:
        model (SentenceTransformer): bi-encoder model
        df (pd.DataFrame): dataframe with source_text, target_text and match columns
        threshold (float, optional): similarity threshold to classify as match. Defaults to 0.5.

    Returns:
        dict: evaluation metrics
    """
    
    # Encode source and target texts
    souce_embeddings = model.encode(df["source_text"].tolist(), convert_to_tensor=True)
    target_embeddings = model.encode(df["target_text"].tolist(), convert_to_tensor=True)

    # Compute cosine similarities and binarize based on threshold
    cosine_scores = cos_sim(souce_embeddings, target_embeddings).diagonal().cpu().numpy()
    y_pred = (cosine_scores > threshold).astype(int)

    # True labels
    y_true = df["match"].astype(int).values

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "avg_precision": average_precision_score(y_true, cosine_scores)
    }

    return metrics


def print_metrics(
        metrics: dict
) -> None:
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


def bi_objective(
        trial: optuna.Trial,
        df_train, df_val,
        model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
) -> float:
    """


    Args:
        trial (optuna.Trial): _description_
        df_train (_type_): _description_
        df_val (_type_): _description_
        model_name (str, optional): _description_. Defaults to "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb".

    Returns:
        float: _description_
    """
    # 1. Hyperparameter suggestions
    # Define the hyperparameter search space
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("per_device_train_batch_size", [16, 32])
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name).to(device)

    dataset_train = convert_df_to_dataset(df_train)

    # Evaluator for validation
    evaluator = BinaryClassificationEvaluator(
        sentences1=df_val["source_text"].tolist(),
        sentences2=df_val["target_text"].tolist(),
        labels=df_val["match"].astype(int).tolist(),
        name=f"optuna_eval_trial_{trial.number}"
    )

    train_loss = OnlineContrastiveLoss(model)

    # --- 2. Training Arguments con parametri Optuna ---
    args = SentenceTransformerTrainingArguments(
        output_dir=f"./optuna_outputs/trial_{trial.number}",
        num_train_epochs=3, # Lower epochs for faster optimization
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_steps=100,
        fp16=True,
        eval_strategy="no",
        save_strategy="no",
        report_to="none"
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset_train,
        loss=train_loss,
    )

    trainer.train()

    # 3. Evaluation on validation set
    eval_results = evaluator(model)

    print(evaluator.primary_metric)
    return eval_results[evaluator.primary_metric]


def train_bi_encoder(
    df_train: pd.DataFrame,
    df_val: Optional[pd.DataFrame],
    df_test: Optional[pd.DataFrame],
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.0,
    num_epochs: int = 10,
    model_name: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    project_name: str = "bi-encoder-alignment",
    output_dir: str = "outputs/bi_encoder_model"
) -> None:
    """
    Train a bi-encoder model. df_val/df_test can be None (train-only mode).
    """

    if df_train is None or len(df_train) == 0:
        raise ValueError("df_train is empty.")

    has_val = df_val is not None and len(df_val) > 0
    has_test = df_test is not None and len(df_test) > 0

    output_dir = str(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    dataset_train = convert_df_to_dataset(df_train)
    dataset_val = convert_df_to_dataset(df_val) if has_val else None

    warmup_steps = int(0.1 * len(dataset_train) / batch_size)

    # ---- WandB (logga solo sizes che esistono)
    wandb.init(project=project_name, config={
        "model_name": model_name,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "train_size": len(df_train),
        "val_size": (len(df_val) if has_val else 0),
        "test_size": (len(df_test) if has_test else 0),
        "has_val": has_val,
        "has_test": has_test,
    })

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name).to(device)

    # Evaluator solo se val esiste
    evaluator = None
    if has_val:
        evaluator = BinaryClassificationEvaluator(
            sentences1=df_val["source_text"].tolist(),
            sentences2=df_val["target_text"].tolist(),
            labels=df_val["match"].astype(int).tolist(),
            name="val_bi_encoder"
        )

    train_loss = OnlineContrastiveLoss(model)

    # Se non hai val: niente eval/best-model logic
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,

        report_to="wandb",
        run_name="bi-encoder-training",
        logging_steps=10,

        eval_strategy=("epoch" if has_val else "no"),
        save_strategy=("epoch" if has_val else "no"),
        save_total_limit=(2 if has_val else 0),

        load_best_model_at_end=(True if has_val else False),
        metric_for_best_model=("eval_val_bi_encoder_cosine_ap" if has_val else None),
        greater_is_better=True,

        fp16=True,
        remove_unused_columns=True
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset_train,
        eval_dataset=(dataset_val if has_val else None),
        loss=train_loss,
        evaluator=evaluator
    )

    trainer.train()
    wandb.finish()

    if has_val:
        final_results = evaluator(model)
        metric = f"{evaluator.name}_cosine_accuracy_threshold"
        best_threshold = final_results[metric]
        print(f"Best threshold for metric {metric}: {best_threshold}")

        metrics_val = evaluate_bi_encoder(model, df_val, threshold=best_threshold)
        print("Validation metrics:")
        print_metrics(metrics_val)

        if has_test:
            metrics_test = evaluate_bi_encoder(model, df_test, threshold=best_threshold)
            print("Test metrics:")
            print_metrics(metrics_test)
            pass

    model.save(str(Path(output_dir) / "final_bi_encoder_model"))