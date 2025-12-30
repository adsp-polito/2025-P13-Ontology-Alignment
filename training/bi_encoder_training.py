from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.losses import OnlineContrastiveLoss
from sentence_transformers.util import cos_sim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score
import pandas as pd
import os
import wandb
from .utils import stratified_split, convert_df_to_dataset

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


def find_best_threshold(
      model: SentenceTransformer,
      df_val: pd.DataFrame,
      metric: str ="accuracy"
) -> tuple:
    """
    Find the best similarity threshold on the validation set.

    Parameters:
        model (SentenceTransformer): bi-encoder model
        df (pd.DataFrame): validation dataframe
        metric (str, optional): metric to optimize. Defaults to "accuracy".

    Returns:
        tuple: best threshold and corresponding metric value
    """

    thresholds = [0.5, 0.55, 0.6, 0.65, 0.75, 0.8, 0.85]
    best_value = -1
    best_threshold = 0.5

    for threshold in thresholds:
        metrics = evaluate_bi_encoder(model, df_val, threshold=threshold)
        if metrics[metric] > best_value:
            best_threshold = threshold

    return best_threshold, best_value


def train_bi_encoder(
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        num_epochs: int = 10,
        model_name: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        project_name: str = "bi-encoder-alignment",
        output_dir: str = "outputs/bi_encoder_model"
) -> None:
    """Train a bi-encoder model using the provided training, validation, and test dataframes.

    Parameters:
        df_train (pd.DataFrame): training dataframe
        df_val (pd.DataFrame): validation dataframe
        df_test (pd.DataFrame): test dataframe
        num_epochs (int, optional): number of training epochs. Defaults to 10.
        model_name (str, optional): pre-trained model name. Defaults to "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb".
    
    Returns:
        SentenceTransformer: fine-tuned bi-encoder model
    """

    dataset_train = convert_df_to_dataset(df_train)
    dataset_val = convert_df_to_dataset(df_val)
    # dataset_test = convert_df_to_dataset(df_test)
    
    # 1. WandB Initialization
    # -------------------------------------------------------
    wandb.init(project=project_name, config={
        "model_name": model_name,
        "epochs": num_epochs,
        "batch_size": 16,
        "train_size": len(df_train),
        "val_size": len(df_val)
    })
    # -------------------------------------------------------

    model = SentenceTransformer(model_name)

    # Create evaluator for validation set
    evaluator = BinaryClassificationEvaluator(
        sentences1=df_val["source_text"].tolist(),
        sentences2=df_val["target_text"].tolist(),
        labels=df_val["match"].astype(int).tolist(),
        name="val_bi_encoder"
    )

    # Define the loss function
    train_loss = OnlineContrastiveLoss(model)

    # 5. Training Arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        
        # --- WandB Logging Configuration---
        report_to="wandb",
        run_name="bi-encoder-training",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_val_bi_encoder_cosine_ap", # L'evaluator aggiunge il prefisso
        greater_is_better=True,
        
        # --- Optimization ---
        fp16=True,

        remove_unused_columns=True
        
    )

    # 6. Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        loss=train_loss,
        evaluator=evaluator
    )

    trainer.train()

    wandb.finish()

    # Evaluate on validation set using default threshold 0.75
    metrics = evaluate_bi_encoder(model, df_val, threshold=0.75)
    print("Validation metrics:")
    print_metrics(metrics)
    
    # Find best threshold
    best_threshold, best_value = find_best_threshold(model, df_val, metric="accuracy")
    print(f"Best threshold on validation set: {best_threshold} with {best_value:.4f} accuracy")

    # Test best threshold on test set
    metrics_test = evaluate_bi_encoder(model, df_test, threshold=best_threshold)
    print("Test metrics at best threshold:")
    print_metrics(metrics_test)

    model.save(output_dir + "/final_bi_encoder_model")


if __name__ == "__main__": # For quick testing
    # Get the project root directory (parent of training directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "outputs", "envo_sweet_training3.csv")
    
    df = pd.read_csv(csv_path)
    # df = pd.read_csv("outputs/envo_sweet_training3.csv")

    df_train, df_val, df_test = stratified_split(df, val_size=0.2, test_size=0.2)

    model = train_bi_encoder(df_train, df_val, df_test, num_epochs=2)

