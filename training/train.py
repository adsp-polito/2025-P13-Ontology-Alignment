from __future__ import annotations
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from training.bi_encoder_training import train_bi_encoder, bi_objective
from training.cross_encoder_training import train_cross_encoder, cross_objective
from typing import Optional

def train_model(
    df_train: pd.DataFrame,
    df_val: Optional[pd.DataFrame],
    df_test: Optional[pd.DataFrame],
    model_type: str,
    model_name: str,
    output_dir: str,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.0,
    num_epochs: int = 10
) -> None:
    """
    Train a model (bi-encoder or cross-encoder) on the given dataframes.
    df_val / df_test can be None (e.g. train-only mode).
    """

    # Minimal sanity checks
    if df_train is None or len(df_train) == 0:
        raise ValueError("df_train is empty.")

    if model_type == "bi-encoder":
        train_bi_encoder(
            df_train,
            df_val,   # can be None
            df_test,  # can be None
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_epochs=int(num_epochs),
            model_name=model_name,
            output_dir=output_dir
        )
    else:
        train_cross_encoder(
            df_train,
            df_val,   # can be None
            df_test,  # can be None
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_epochs=int(num_epochs),
            model_name=model_name,
            output_dir=output_dir
        )

def optimize_model(
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        model_type: str,
        model_name: str,
        n_trials: int = 10
):
    """
    Optimize hyperparameters for the given model type using Optuna.

    Parameters:
        df_train (pd.DataFrame): training dataframe
        df_val (pd.DataFrame): validation dataframe
        model_type (str): type of the model ("bi-encoder" or "cross-encoder")
        model_name (str): name of the pre-trained model
        n_trials (int, optional): number of optimization trials. Defaults to 10.
    
    Returns:
        dict: best hyperparameters found
    """
    if model_type == "bi-encoder":
        objective = bi_objective
    else:
        objective = cross_objective

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
    )
    
    study.optimize(
        lambda trial: objective(
            trial,
            df_train,
            df_val,
            model_name
        ),
        n_trials=n_trials,
    )

    print(f"Best parameters found: {study.best_params}")
    return study.best_params
