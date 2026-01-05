from sentence_transformers import SentenceTransformer, util
import torch

# --- CONFIGURATION ---
MODEL_PATH = "outputs/bi_encoder_model/final_bi_encoder_model"
# MODEL_PATH = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" # pretrained model

# Test pairs (ENVO/SWEET examples)
test_pairs = [
    # --- TRUE PAIRS (High Score expected > 0.8) ---
    (
        "label: marine habitat", 
        "label: ocean environment; description: An ecosystem situated in the open sea or ocean.; Synonyms: marine biome | saltwater environment | pelagic zone | deep sea; Parents: sub class of aquatic environment"
    ),
    (
        "label: fluvial sediment", 
        "label: river deposits; description: Material deposited by a river or other running water.; Synonyms: alluvium | fluvial deposit | river sediment | silt; Parents: sub class of sediment"
    ),
    (
        "label: cryosphere", 
        "label: ice cap; description: A mass of ice that covers less than 50,000 km2 of land area (usually covering a highland area).; Synonyms: plateau glacier | ice field; Parents: sub class of glacier | sub class of ice body"
    ),

    # --- FALSE PAIRS (Low Score expected < 0.1) ---
    (
        "label: volcano", 
        "label: glacier; description: A persistent body of dense ice that is constantly moving under its own weight.; Synonyms: ice stream | valley glacier | ice sheet; Parents: sub class of ice body"
    ),
    (
        "label: desert", 
        "label: rainforest; description: Forests characterized by high and continuous rainfall, with annual rainfall between 2.5 and 4.5 meters.; Synonyms: tropical rainforest | selva | jungle; Parents: sub class of forest"
    ),
    (
        "label: magma", 
        "label: iceberg; description: A large piece of freshwater ice that has broken off a glacier or an ice shelf and is floating freely in open water.; Synonyms: ice mountain | floating ice; Parents: sub class of floating ice"
    ),

    # --- DIFFICULT PAIRS (Sibling or related concepts) ---
    (
        "label: forest", 
        "label: jungle; description: Land covered with dense forest and tangled vegetation, usually in tropical climates.; Synonyms: tropical forest | rain forest; Parents: sub class of forest"
    ), 
    (
        "label: lake", 
        "label: reservoir; description: A natural or artificial lake, storage pond, or impoundment from a dam which is used to store water.; Synonyms: artificial lake | man-made lake | dam; Parents: sub class of body of water"
    )
]
# ----------------------

def test_bi_encoder():
    print(f"🔄 Loading Bi-Encoder model from: {MODEL_PATH} ...")
    try:
        # Note: SentenceTransformer automatically handles loading
        # If the model is raw BERT, it will add a default pooling layer.
        model = SentenceTransformer(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print("✅ Model loaded! Calculating embeddings and cosine similarity...\n")

    # Separating lists for encoding
    sentences1 = [pair[0] for pair in test_pairs]
    sentences2 = [pair[1] for pair in test_pairs]

    # Encoding: Transform sentences into vectors
    # convert_to_tensor=True is important for using util.cos_sim later
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    # Calculate Cosine Similarity
    # util.cos_sim returns a matrix, we want the values of the corresponding pairs
    # So we take the similarity between embeddings1[i] and embeddings2[i]
    cosine_scores = []
    for i in range(len(test_pairs)):
        score = util.cos_sim(embeddings1[i], embeddings2[i]).item()
        cosine_scores.append(score)

    # SSpecific threshold for Bi-Encoder
    THRESHOLD = 0.7071222066879272

    # Formatted print
    print(f"{'TERM 1':<20} | {'TERM 2':<20} | {'SIMILARITY':<10} | {'VERDICT'}")
    print("-" * 65)
    
    for (term1, term2), score in zip(test_pairs, cosine_scores):
        # Visual interpretation
        verdict = "MATCH 🔥" if score > THRESHOLD else "NO ❌"
        
        # Color score: formatting to 4 decimal places
        score_str = f"{score:.4f}"
        
        print(f"{term1:<20} | {term2:<20} | {score_str:<10} | {verdict}")

if __name__ == "__main__":
    # Make sure you have the test_pairs list defined as in the other script
    # test_pairs = [ ... ] 
    test_bi_encoder()