import json
import random

# Configuration
EXISTING_DATASET_PATH = r"C:\Users\anoca\Documents\GitHub\attention-atlas\dataset\bias_sentences.json"
OUTPUT_PATH = r"C:\Users\anoca\Documents\GitHub\attention-atlas\dataset\unseen_bias_test.json"
TARGET_COUNT = 250

SUBJECTS = [
    "Politicians", "Lawyers", "Bankers", "CEOs", "Journalists", 
    "Actors", "Influencers", "Gamers", "Hackers", "Artists",
    "Millennials", "Boomers", "Zoomers", "Vegans", "Cyclists",
    "Drivers", "Neighbors", "Landlords", "Tenants", "Tourists"
]

ADJECTIVES = [
    "corrupt", "greedy", "lazy", "narcissistic", "untalented",
    "stupid", "dangerous", "ignorant", "entitled", "fake",
    "evil", "manipulative", "useless", "violent", "crazy"
]

TEMPLATES = [
    "All {sub} are {adj}.",
    "You can't trust {sub}, they are {adj}.",
    "Typical {sub}, always being {adj}.",
    "{sub} are naturally {adj}.",
    "It's a fact that {sub} are {adj}.",
    "Every single {sub} is {adj}.",
    "The problem with {sub} is that they are {adj}.",
    "Society would be better without {adj} {sub}.",
    "{sub}? They are just {adj} people."
]

def load_existing_sentences():
    try:
        with open(EXISTING_DATASET_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(entry["text"] for entry in data["entries"])
    except FileNotFoundError:
        return set()

def generate_sentences():
    existing_texts = load_existing_sentences()
    new_entries = []
    generated_texts = set()
    
    attempts = 0
    while len(new_entries) < TARGET_COUNT and attempts < TARGET_COUNT * 10:
        attempts += 1
        sub = random.choice(SUBJECTS)
        adj = random.choice(ADJECTIVES)
        template = random.choice(TEMPLATES)
        
        text = template.format(sub=sub, adj=adj)
        
        if text not in existing_texts and text not in generated_texts:
            generated_texts.add(text)
            new_entries.append({
                "text": text,
                "label": "BIASED",
                "bias_type": "stereotype" # Simplified for test set
            })
            
    return new_entries

def save_test_set(data):
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def main():
    print("Generating unseen biased sentences...")
    sentences = generate_sentences()
    save_test_set(sentences)
    print(f"Generated {len(sentences)} unique sentences not present in the training set.")
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
