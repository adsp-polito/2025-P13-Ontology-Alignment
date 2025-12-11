import os
import math
import logging
import pandas as pd
from google.colab import drive
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sklearn.model_selection import train_test_split

# --- 1. SETUP ---
# Disable WandB login hanging
os.environ["WANDB_DISABLED"] = "true"

# Connect to Drive
print(" Connecting to Drive...")
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# --- 2. PATHS ---
BASE_PATH = '/content/drive/MyDrive/2025-P13-Ontology-Alignment'
DATA_FILE = os.path.join(BASE_PATH, "datasets", "sweet_envo_training_updated.csv")
OUTPUT_PATH = os.path.join(BASE_PATH, "output", "supervised_bert_model")

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Hyperparameters
MODEL_NAME = 'cross-encoder/stsb-distilroberta-base'
BATCH_SIZE = 16
NUM_EPOCHS = 8  

def run_trainer():
    print("--- Starting Robust Trainer ---")
    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print(f" Error: Input file missing: {DATA_FILE}")
        return

    print(f" Reading data...")
    df = pd.read_csv(DATA_FILE)

    # Clean Data
    samples = []
    for i, row in df.iterrows():
        try:
            score = float(row['match'])
            samples.append(InputExample(texts=[str(row['source_text']), str(row['target_text'])], label=score))
        except:
            continue

    if not samples:
        print(" No valid data found!")
        return

    print(f" Loaded {len(samples)} training pairs.")

    # 2. Setup Training
    train_samples, val_samples = train_test_split(samples, test_size=0.2, random_state=42)
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(val_samples, name='Val')

    # 3. Load Model
    print(f" Loading Base Model: {MODEL_NAME}")
    model = CrossEncoder(MODEL_NAME, num_labels=1)

    # 4. TRAIN
    print(" Training Started... ")
    warmup_steps = math.ceil(len(train_dataloader) * NUM_EPOCHS * 0.1)

    try:
        model.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=NUM_EPOCHS,
            warmup_steps=warmup_steps,
            output_path=OUTPUT_PATH, 
            save_best_model=True,
            show_progress_bar=True

        )

        print(" Training finished naturally.")

    except Exception as e:
        print(f" Training interrupted: {e}")

    # 5. FORCE SAVE (The Safety Net)
    print(" Saving Final Model...")
    model.save(OUTPUT_PATH)
    print(f" SUCCESS! Check this folder: {OUTPUT_PATH}")
    print("   You should see 'pytorch_model.bin' inside.")

if __name__ == "__main__":
    run_trainer()