from sentence_transformers import CrossEncoder
import torch
import pandas as pd
from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator

from training.cross_encoder_training import evaluate_cross_encoder, print_metrics

# --- CONFIGURATION ---
MODEL_PATH = "outputs/cross_encoder_model_bert_base_uncased/final_cross_encoder_model"

# Test pairs (ENVO/SWEET examples)

df_val = pd.read_csv("datasets/val_split.csv")
df_test = pd.read_csv("datasets/test_split.csv")

val_evaluator = CrossEncoderClassificationEvaluator(
    sentence_pairs=list(zip(df_val["source_text"], df_val["target_text"])),
    labels=df_val["match"].astype(int).tolist(),
    name="val_evaluator"
)

test_evaluator = CrossEncoderClassificationEvaluator(
    sentence_pairs=list(zip(df_test["source_text"], df_test["target_text"])),
    labels=df_test["match"].astype(int).tolist(),
    name="test_evaluator"
)


def test_model():
    print(f"🔄 Loading model from: {MODEL_PATH} ...")
    try:
        model = CrossEncoder(MODEL_PATH, num_labels=1)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure the path points to the folder containing 'pytorch_model.bin'")
        return

    print("✅ Model loaded! Calculating scores...\n")

    final_results = val_evaluator(model)
    print(final_results)
    # Best threshold
    metric = f"{val_evaluator.name}_accuracy_threshold"
    best_threshold = final_results[metric]
    print(f"Best threshold for metric {metric}: {best_threshold}")

    metrics_val = evaluate_cross_encoder(model, df_val, threshold=best_threshold)
    print(f"Validation metrics at best threshold {best_threshold}:")
    print_metrics(metrics_val)

    # Test best threshold on test set
    metrics_test = evaluate_cross_encoder(model, df_test, threshold=best_threshold)
    print(f"Test metrics at best threshold {best_threshold}:")
    print_metrics(metrics_test)

if __name__ == "__main__":
    test_model()

    # 0.008591532707214355
    # cross-encoder microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext threshold: 0.013183257542550564
    # cross-encoder pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb threshold: 0.996262788772583