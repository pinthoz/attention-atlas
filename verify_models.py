import sys
import os
from pathlib import Path

# Add the project root to the python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from attention_app.models import ModelManager

def verify_models():
    models_to_test = [
        "bert-base-uncased",
        "bert-large-uncased",
        "bert-base-multilingual-uncased",
        "gpt2",
        "gpt2-medium"
    ]
    
    print("Starting model verification...")
    
    for model_name in models_to_test:
        print(f"\nTesting model: {model_name}")
        try:
            tokenizer, encoder, mlm = ModelManager.get_model(model_name)
            print(f"  - Successfully loaded {model_name}")
            
            # Test tokenization
            text = "Hello world"
            inputs = tokenizer(text, return_tensors="pt")
            
            # Move inputs to device
            device = ModelManager.get_device()
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            print(f"  - Tokenization successful: {inputs['input_ids'].shape}")
            
            # Test forward pass
            outputs = encoder(**inputs)
            print(f"  - Forward pass successful")
            print(f"  - Hidden states shape: {outputs.last_hidden_state.shape}")
            print(f"  - Attentions available: {outputs.attentions is not None}")
            if outputs.attentions:
                print(f"  - Number of layers: {len(outputs.attentions)}")
                
        except Exception as e:
            print(f"  - FAILED: {e}")
            return False
            
    print("\nAll models verified successfully!")
    return True

if __name__ == "__main__":
    verify_models()
