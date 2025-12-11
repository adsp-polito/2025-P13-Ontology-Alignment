import os
import pickle
import owlready2
from tqdm import tqdm
from transformers import AutoTokenizer
from collections import defaultdict
from google.colab import drive

# --- CONFIGURATION ---

BASE_PATH = '/content/drive/MyDrive/2025-P13-Ontology-Alignment'
ONTOLOGY_PATH = os.path.join(BASE_PATH, "datasets", "envo.owl")
OUTPUT_PATH = os.path.join(BASE_PATH, "datasets", "inverted_index_bundle.pkl")



def compile_inverted_index():
    print("--- Method: Compiling Inverted Index ---")
    if not os.path.exists(ONTOLOGY_PATH):
        print(f" Error: Ontology not found at {ONTOLOGY_PATH}")
        return

    # 1. Load Ontology
    print(" Loading Ontology...")
    onto = owlready2.get_ontology(ONTOLOGY_PATH).load()
    classes = list(onto.classes())

    # 2. Tokenizer (WordPiece)
    #  BERT uses WordPiece.
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    index = defaultdict(set)
    class_data = {}

    print(f" Indexing {len(classes)} classes...")
    for idx, cls in enumerate(tqdm(classes)):
        lbl = cls.label[0] if cls.label else cls.name
        synonyms = cls.hasExactSynonym if hasattr(cls, "hasExactSynonym") else []

        # Store for Exact Match lookup
        rich_text = f"label: {lbl}; synonyms: {', '.join(synonyms)}"
        class_data[idx] = {
            "label": lbl,
            "rich_text": rich_text,
            "iri": cls.iri,
            "synonyms": [lbl] + synonyms # Store list for exact matching

        }

        # Build Sub-Word Index
        all_text = [lbl] + synonyms
        for text in all_text:
            tokens = tokenizer.tokenize(str(text).lower())
            for t in tokens:
                clean_t = t.replace("##", "")
                if len(clean_t) > 2: # Filter noise
                    index[clean_t].add(idx)

                   

    print(f" Saving Inverted Index to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump({"index": dict(index), "class_data": class_data}, f)
    print(" Compilation Complete.")

if __name__ == "__main__":
    if not os.path.exists('/content/drive'): drive.mount('/content/drive')
    compile_inverted_index()