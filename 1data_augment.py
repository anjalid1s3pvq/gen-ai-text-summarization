# data_augment.py - Augment dataset with medical terms

import spacy
import random
from datasets import load_dataset
from tqdm import tqdm

# Load medical NER model (installed via scispacy)
nlp = spacy.load("en_ner_bc5cdr_md")

# Simple medical terms (expand with UMLS later if needed)
medical_conditions = ["hypertensive", "diabetic", "diabetes", "asthmatic", "asthma", "cancerous", "cancer", "arthritic", "arthritis", "epileptic", "epilepsy", "anemic", "anemia", "hypertension"]
drugs = ["aspirin", "ibuprofen", "paracetamol", "metformin", "insulin", "albuterol", "chemotherapy"]

def augment_text(text):
    """Replace entities with medical terms using NER and emphasize them."""
    doc = nlp(text)
    augmented = text
    entity_emphasis = []
    
    for ent in doc.ents:
        if ent.label_ == "DISEASE":
            replacement = random.choice(medical_conditions)
            augmented = augmented.replace(ent.text, replacement)
            entity_emphasis.append(replacement)
        elif ent.label_ == "CHEMICAL":
            replacement = random.choice(drugs)
            augmented = augmented.replace(ent.text, replacement)
            entity_emphasis.append(replacement)
    
    # Add entity emphasis by repeating key medical terms
    if entity_emphasis and random.random() < 0.3:  # 30% chance to add emphasis
        key_entity = random.choice(entity_emphasis)
        augmented += f" The patient's {key_entity} condition requires attention."
    
    return augmented

# Load CNN/Daily Mail dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Subset for speed (increase to 300k for full, but start small)
train_subset = dataset["train"].shuffle(seed=42).select(range(1000))
val_subset = dataset["validation"].shuffle(seed=42).select(range(200))

def augment_example(example):
    # Augment both article and highlights
    example["article"] = augment_text(example["article"])
    example["highlights"] = augment_text(example["highlights"])
    return example

# Augment and save
augmented_train = train_subset.map(augment_example)
augmented_val = val_subset.map(augment_example)

augmented_train.save_to_disk("augmented_train")
augmented_val.save_to_disk("augmented_val")
print("Data augmentation complete! Augmented data saved to folders.")