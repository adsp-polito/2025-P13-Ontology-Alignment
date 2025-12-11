import os
import pickle
import math
import pandas as pd
from transformers import AutoTokenizer
from sentence_transformers import CrossEncoder
from tqdm import tqdm
from google.colab import drive

# --- CONFIGURATION ---
BASE_PATH = '/content/drive/MyDrive/2025-P13-Ontology-Alignment'
MODEL_PATH = os.path.join(BASE_PATH, "output", "supervised_bert_model")
INPUT_CSV = os.path.join(BASE_PATH, "datasets", "sweet_envo_training_updated.csv")
INDEX_PATH = os.path.join(BASE_PATH, "datasets", "inverted_index_bundle.pkl")
OUTPUT_CSV = os.path.join(BASE_PATH, "output", "results_inverted_index.csv")

def run_pipeline():
    print("--- Inference Pipeline  ---")
    if not os.path.exists(INDEX_PATH):
        print(" Index missing. Run compile_inverted_index.py first.")
        return

    # Load Index
    with open(INDEX_PATH, 'rb') as f:
        bundle = pickle.load(f)
        index = bundle["index"]
        class_data = bundle["class_data"]
        total_docs = len(class_data)

    print(" Loading Models...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    cross_encoder = CrossEncoder(MODEL_PATH)

    # Helper: Exact Match Check
    def check_exact_match(query):
        query_norm = query.lower().strip()
        for cid, data in class_data.items():
            # Check label and all synonyms
            for phrase in data['synonyms']:
                if phrase.lower().strip() == query_norm:
                    return data['label']
        return None

    # Helper: Step 2 Candidate Selection
    def get_candidates(query):
        tokens = tokenizer.tokenize(query.lower())
        scores = {}
        for t in tokens:
            clean = t.replace("##", "")
            if clean in index:
                # IDF Scoring: log(Total / DocFreq)
                idf = math.log10(total_docs / len(index[clean]))
                for cid in index[clean]:
                    scores[cid] = scores.get(cid, 0.0) + idf
        # Top 50
        top_ids = sorted(scores, key=scores.get, reverse=True)[:50]
        return [class_data[cid] for cid in top_ids]

    df = pd.read_csv(INPUT_CSV)
    results = []

    print(f" Processing {len(df)} rows...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        query = str(row['source_text'])

        # --- STEP 1: EXACT MATCH SHORTCUT ---
        exact_match = check_exact_match(query)
        if exact_match:

            # "No BERT call is needed. S(a,c) = 1.0"
            results.append({
                "source_text": query,
                "predicted_match": exact_match,
                "confidence": 1.0,
                "method": "Exact Match Shortcut",
                "ground_truth_match": row.get('match', 0)
            })
            continue

        # --- STEP 2: CANDIDATE SELECTION (Inverted Index) ---
        candidates = get_candidates(query)

        if not candidates:
            results.append({"source_text": query, "predicted_match": "None", "confidence": 0.0, "method": "Index Miss", "ground_truth_match": row.get('match', 0)})
            continue

        # --- STEP 3: CROSS-ENCODER SCORING ---
        cross_inputs = [[query, c['rich_text']] for c in candidates]
        scores = cross_encoder.predict(cross_inputs)

        # --- STEP 4: FINAL DECISION ---
        top_result = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[0]
        results.append({
            "source_text": query,
            "predicted_match": top_result[0]['label'],
            "confidence": float(top_result[1]),
            "method": "Inverted Index + CrossEncoder",
            "ground_truth_match": row.get('match', 0)
        })

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f" Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    if not os.path.exists('/content/drive'): drive.mount('/content/drive')
    run_pipeline()