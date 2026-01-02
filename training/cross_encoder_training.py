from sentence_transformers import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import os
import wandb
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


def find_best_threshold(
      model: CrossEncoder,
      df_val: pd.DataFrame,
      metric: str = "accuracy"
) -> tuple:
    """
    Find the best similarity threshold on the validation set.

    Parameters:
        model (CrossEncoder): cross-encoder model
        df (pd.DataFrame): validation dataframe
        metric (str, optional): metric to optimize. Defaults to "accuracy".
    
    Returns:
        tuple: best threshold and corresponding metric value
    """
  
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.75, 0.8, 0.85]
    best_value = -1
    best_threshold = 0.5

    for threshold in thresholds:
        metrics = evaluate_cross_encoder(model, df_val, threshold=threshold)
        if metrics[metric] > best_value:
            best_value = metrics[metric]
            best_threshold = threshold

    return best_threshold, best_value


def train_cross_encoder(
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
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
        "batch_size": 16,
        "train_size": len(df_train),
        "val_size": len(df_val)
    })
    # -------------------------------------------------------
    
    model = CrossEncoder(model_name, num_labels=1)

    args = CrossEncoderTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(num_epochs),
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        
        # --- WandB Logging Configuration---
        report_to="wandb",  # Enable logging to WandB
        run_name="cross-encoder-training",
        logging_steps=num_epochs,
        eval_strategy="epoch", # Evaluate after each epoch
        save_strategy="epoch",       # Save checkpoint after each epoch
        save_total_limit=2,          # Keep only the last 2 checkpoints
        load_best_model_at_end=True, # Load best model when finished training
        metric_for_best_model="eval_val_evaluator_average_precision", # Nome loggato dall'evaluator
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

    # Evaluate on validation set using default threshold 0.75
    print("Validation metrics: ")
    metrics = evaluate_cross_encoder(model, df_val, threshold=0.75)
    print_metrics(metrics)
    
    # Find best threshold
    best_threshold, best_value = find_best_threshold(model, df_val, metric="accuracy")
    print(f"Best threshold on validation set: {best_threshold} with {best_value:.4f} accuracy")

    # Test best threshold on test set
    metrics_test = evaluate_cross_encoder(model, df_test, threshold=best_threshold)
    print("Test metrics at best threshold:")
    print_metrics(metrics_test)
    
    model.save(output_dir + "/final_cross_encoder_model")

def sample_type(x):
    if x<683: return "positive"
    if x<1024: return "hard_negative"
    return "random_negative"

if __name__ == "__main__": # For quick testing
    # Get the project root directory (parent of training directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "outputs", "envo_sweet_training.csv")
    
    df = pd.read_csv(csv_path)

    df_train, df_val, df_test = stratified_split(df, val_size=0.2, test_size=0.2)

    model = train_cross_encoder(df_train, df_val, df_test, num_epochs=2)