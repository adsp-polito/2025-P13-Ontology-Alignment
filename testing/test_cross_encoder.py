from sentence_transformers import CrossEncoder
import torch

# --- CONFIGURATION ---
MODEL_PATH = "outputs/cross_encoder_model_PubMedBERT/final_cross_encoder_model"
# MODEL_PATH = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" # pretrained model
# MODEL_PATH = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"

# Test pairs (ENVO/SWEET examples)
test_pairs = [
    (
        "label: marine habitat", 
        "label: ocean environment; description: An ecosystem situated in the open sea or ocean.; Synonyms: marine biome | saltwater environment | pelagic zone | deep sea; Parents: sub class of aquatic environment"
    ),
    (
        "label: fluvial sediment", 
        "label: river deposits; description: Material deposited by a river or other running water.; Synonyms: alluvium | fluvial deposit | river sediment | silt; Parents: sub class of sediment"
    ),
    (
        "label: Ethanol",
        "label: ethanol; Parents: sub class of volatile organic compound | sub class of ethanols | sub class of alkyl alcohol | sub class of volatile organic compound | sub class of ethanols | sub class of alkyl alcohol | sub class of primary alcohol"
    ),
    (
        "label: cryosphere", 
        "label: ice cap; description: A mass of ice that covers less than 50,000 km2 of land area (usually covering a highland area).; Synonyms: plateau glacier | ice field; Parents: sub class of glacier | sub class of ice body"
    ),
    (
        "label: Rainwater",
        "label: urban stormwater; description: Stormwater which accumulates in an urban ecosystem.; Synonyms: urban storm water; Parents: sub class of stormwater"
    ),
    (
        "label: SoilLayer",
        "label: mass density of soil; description: The mass density of some soil.; Synonyms: soil mass density; Parents: sub class of mass density; Equivalent to: equivalent to obo.PATO_0001019 & obo.RO_0000052.some(obo.ENVO_00001998)"
    ),
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

    (
        "label: IceSurface",
        "label: ammonia ice; description: Ice which is primarily composed of ammonia.; Parents: sub class of ice"
    ),
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

def test_model():
    print(f"🔄 Loading model from: {MODEL_PATH} ...")
    try:
        model = CrossEncoder(MODEL_PATH, num_labels=1)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure the path points to the folder containing 'pytorch_model.bin'")
        return

    print("✅ Model loaded! Calculating scores...\n")

    # Prediction
    scores = model.predict(test_pairs)

    # Formatted print
    print(f"{'TERM 1':<20} | {'TERM 2':<20} | {'SCORE':<10} | {'VERDICT'}")
    print("-" * 65)
    
    for (term1, term2), score in zip(test_pairs, scores):
        # Interpretazione visiva
        verdict = "MATCH 🔥" if score > 0.013183257542550564 else "NO ❌"
        color_score = f"{score:.4f}"
        
        print(f"{term1:<20} | {term2:<20} | {color_score:<10} | {verdict}")

if __name__ == "__main__":
    test_model()

    # 0.008591532707214355
    # cross-encoder microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext threshold: 0.013183257542550564
    # cross-encoder pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb threshold: 0.996262788772583