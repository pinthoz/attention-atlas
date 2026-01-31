
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import argparse

# Label mapping (must match training)
LABEL2ID = {
    'O': 0,
    'B-STEREO': 1,
    'I-STEREO': 2,
    'B-GEN': 3,
    'I-GEN': 4,
    'B-UNFAIR': 5,
    'I-UNFAIR': 6
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def predict(text, model, tokenizer, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs).logits
    
    probabilities = torch.sigmoid(outputs)
    predictions = (probabilities > threshold).int()[0]
    
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    results = []
    current_entity = None
    
    print(f"{'Token':<15} | {'Label':<10} | {'Score':<6}")
    print("-" * 40)

    for idx, (token, pred_vec) in enumerate(zip(tokens, predictions)):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
            
        # Get probabilities for this token
        token_probs = probabilities[0][idx]
        
        print(f"{token:<15} | {'-'*10} | {'-'*6}")
        
        # Sort by probability descending
        probs_with_labels = []
        for label_idx, prob in enumerate(token_probs):
            probs_with_labels.append((ID2LABEL[label_idx], prob.item()))
        
        probs_with_labels.sort(key=lambda x: x[1], reverse=True)
        
        for label, prob in probs_with_labels:
            # Highlight high probs
            marker = "*" if prob > threshold else " "
            print(f"{'':<15} | {label:<10} | {prob:.4f} {marker}")
            
    return results
            
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/gusnet-finetuned")
    parser.add_argument("--text", default="Women are emotional and bad drivers.")
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}...")
    try:
        model = BertForTokenClassification.from_pretrained(args.model, num_labels=len(LABEL2ID))
        tokenizer = BertTokenizerFast.from_pretrained(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"\nTest Sentence: {args.text}")
    print("-" * 30)
    
    results = predict(args.text, model, tokenizer)
    
    if not results:
        print("No bias detected.")
    else:
        print("Bias detected:")
        for token, labels in results:
            print(f"  {token}: {', '.join(labels)}")
            
if __name__ == "__main__":
    main()
