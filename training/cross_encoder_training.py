from sentence_transformers import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import os
import wandb
import optuna
import torch
from .utils import stratified_split, convert_df_to_dataset

def evaluate_cross_encoder(
      model: CrossEncoder,
      df: pd.DataFrame,
      threshold: float = 0.5
) -> dict:
    """
    Evaluate cross-encoder model on val/test dataframe.

    Parameters:
        model (CrossEncoder): cross-encoder model
        df (pd.DataFrame): dataframe with source_text, target_text and match columns
        threshold (float, optional): similarity threshold to classify as match. Defaults to 0.5.
    
    Returns:
        dict: evaluation metrics
    """
  
    # The predict method of CrossEncoder requires a list of text pairs and returns similarity scores
    pairs = [(row["source_text"], row["target_text"]) for _, row in df.iterrows()]
    y_scores = model.predict(pairs)

    # Binarize scores based on threshold
    y_pred = (y_scores > threshold).astype(int)

    # True labels
    y_true = df["match"].astype(int).values

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }
  
    return metrics


def print_metrics(
        metrics: dict
) -> None:
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


def cross_objective(
        trial: optuna.Trial,
        df_train, df_val,
        model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
) -> float:
    """
    Hyperparameter optimization objective function for cross-encoder model.

    Parameters:
        trial (optuna.Trial): Optuna trial object
        df_train (pd.DataFrame): training dataframe
        df_val (pd.DataFrame): validation dataframe
        model_name (str, optional): pre-trained model name. Defaults to "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb".

    Returns:
        float: validation metric to maximize (e.g., average precision)
    """
    # 1. Hyperparameter suggestions
    # Define the hyperparameter search space
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("per_device_train_batch_size", [16, 32])
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)

    dataset_train = convert_df_to_dataset(df_train)

    evaluator = CrossEncoderClassificationEvaluator(
        sentence_pairs=list(zip(df_val["source_text"], df_val["target_text"])),
        labels=df_val["match"].astype(int).tolist(),
        name="val_evaluator"
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(model_name, num_labels=1).to(device)
    
    # --- 2. Training Arguments con parametri Optuna ---
    args = CrossEncoderTrainingArguments(
        output_dir=f"./optuna_outputs/trial_{trial.number}",
        num_train_epochs=3, # Teniamo epoche basse per l'ottimizzazione
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_steps=100,
        fp16=True,
        eval_strategy="no", # Non serve valutare ad ogni epoca durante optuna
        save_strategy="no",
        report_to="none"    # Disattiviamo wandb durante i trial per non intasarlo
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=dataset_train
    )

    trainer.train()

    # 3. Evaluation on validation set
    eval_results = evaluator(model)

    print(evaluator.primary_metric)
    return eval_results[evaluator.primary_metric]


def train_cross_encoder(
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.0,
        num_epochs: int = 10,
        model_name: str ="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        project_name: str ="cross-encoder-alignment",
        output_dir: str ="outputs/cross_encoder_model"
) -> None:
    """Train a cross-encoder model using the provided training, validation, and test dataframes.

    Parameters:
        df_train (pd.DataFrame): training dataframe
        df_val (pd.DataFrame): validation dataframe
        df_test (pd.DataFrame): test dataframe
        num_epochs (int, optional): number of training epochs. Defaults to 10.
        model_name (str, optional): pre-trained model name. Defaults to "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb".
    
    Returns:
        CrossEncoder: fine-tuned cross-encoder model
    """

    dataset_train = convert_df_to_dataset(df_train)
    dataset_val = convert_df_to_dataset(df_val)
    # dataset_test = convert_df_to_dataset(df_test)

    warmup_steps = int(0.1 * len(dataset_train) / batch_size)

    evaluator = CrossEncoderClassificationEvaluator(
        sentence_pairs=list(zip(df_val["source_text"], df_val["target_text"])),
        labels=df_val["match"].astype(int).tolist(),
        name="val_evaluator"
    )

    # 1. Inizializza WandB all'inizio del training
    # -------------------------------------------------------
    wandb.init(project=project_name, config={
        "model_name": model_name,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "train_size": len(df_train),
        "val_size": len(df_val)
    })
    # -------------------------------------------------------
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(model_name, num_labels=1).to(device)

    args = CrossEncoderTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(num_epochs),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        
        # --- WandB Logging Configuration---
        report_to="wandb",  # Enable logging to WandB
        run_name="cross-encoder-training",
        logging_steps=10,
        eval_strategy="epoch", # Evaluate after each epoch
        save_strategy="epoch",       # Save checkpoint after each epoch
        save_total_limit=2,          # Keep only the last 2 checkpoints
        load_best_model_at_end=True, # Load best model when finished training
        metric_for_best_model="eval_val_evaluator_average_precision", # Name of the metric to use to compare two different models
        greater_is_better=True,
        
        # --- Optimization ---
        fp16=True, # Use mixed precision training

        remove_unused_columns=True
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        evaluator=evaluator
    )

    trainer.train()

    wandb.finish()

    # Evaluate the model on the validation set
    final_results = evaluator(model)
    
    # Best threshold
    metric = f"{evaluator.name}_accuracy_threshold"
    best_threshold = final_results[metric]
    print(f"Best threshold for metric {metric}: {best_threshold}")

    metrics_val = evaluate_cross_encoder(model, df_val, threshold=best_threshold)
    print(f"Validation metrics at best threshold {best_threshold}:")
    print_metrics(metrics_val)

    # Test best threshold on test set
    metrics_test = evaluate_cross_encoder(model, df_test, threshold=best_threshold)
    print(f"Test metrics at best threshold {best_threshold}:")
    print_metrics(metrics_test)
    
    model.save(output_dir + "/final_cross_encoder_model")