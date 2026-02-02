"""
Script to test attention metrics across sentences of varying lengths.
This helps calibrate interpretation thresholds (Low/Medium/High) for metrics.
"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel

from attention_app.metrics import compute_all_attention_metrics

# Test sentences of varying lengths
TEST_SENTENCES = [
    # Very short (2-4 words)
    "Hello world.",
    "I love cats.",
    "The sun shines.",
    "Birds fly high.",
    
    # Short (5-10 words)
    "The quick brown fox jumps over.",
    "She sells seashells by the seashore.",
    "A stitch in time saves nine.",
    "The cat sat on the mat quietly.",
    
    # Medium (11-20 words)
    "The quick brown fox jumps over the lazy dog in the sunny afternoon.",
    "Machine learning models can understand and generate human language with remarkable accuracy.",
    "The ancient castle stood on the hill, overlooking the peaceful valley below.",
    "Scientists discovered a new species of butterfly in the Amazon rainforest last month.",
    
    # Long (21-40 words)
    "The transformer architecture has revolutionized natural language processing by enabling models to attend to different parts of the input sequence simultaneously, leading to significant improvements in translation, summarization, and question answering tasks.",
    "In the heart of the bustling city, a small coffee shop served as a refuge for writers, artists, and dreamers who sought inspiration in the aroma of freshly brewed coffee and the gentle hum of quiet conversation.",
    
    # Very long (40+ words)
    "Attention mechanisms in deep learning allow neural networks to focus on relevant parts of the input when producing an output, which is particularly useful for sequence-to-sequence tasks where the input and output sequences may have different lengths, and where different parts of the input may be more or less relevant to different parts of the output.",
    "The development of large language models has sparked intense debate about the nature of intelligence, consciousness, and creativity, with some researchers arguing that these models exhibit emergent properties that were not explicitly programmed, while others maintain that they are sophisticated pattern matching systems that lack true understanding of the concepts they manipulate."
]

def analyze_sentence(text, tokenizer, model, device):
    """Compute metrics for a single sentence."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    attentions = outputs.attentions
    att_layers = [layer[0].cpu().numpy() for layer in attentions]
    att_avg = np.mean(att_layers, axis=(0, 1))  # Average across heads, then layers
    
    metrics = compute_all_attention_metrics(att_avg)
    
    # Get token count
    num_tokens = inputs['input_ids'].shape[1]
    num_words = len(text.split())
    
    return {
        'text': text[:50] + '...' if len(text) > 50 else text,
        'num_words': num_words,
        'num_tokens': num_tokens,
        **metrics
    }

def main():
    print("=" * 80)
    print("ATTENTION METRICS CALIBRATION SCRIPT")
    print("=" * 80)
    
    # Load model
    model_name = "bert-base-uncased"
    print(f"\nLoading model: {model_name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True).to(device)
    model.eval()
    
    # Analyze all sentences
    print(f"\nAnalyzing {len(TEST_SENTENCES)} sentences...")
    results = []
    
    for i, sentence in enumerate(TEST_SENTENCES):
        print(f"  [{i+1}/{len(TEST_SENTENCES)}] {sentence[:40]}...")
        result = analyze_sentence(sentence, tokenizer, model, device)
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Add normalized metrics (divided by token count for those affected by length)
    df['focus_normalized'] = df['focus_entropy'] / np.log(df['num_tokens'])  # Normalize by max possible entropy
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS BY SENTENCE LENGTH")
    print("=" * 80)
    
    # Sort by number of tokens
    df_sorted = df.sort_values('num_tokens')
    
    print("\n📊 Raw Metrics (sorted by token count):")
    print("-" * 100)
    
    cols_to_show = ['num_words', 'num_tokens', 'confidence_max', 'confidence_avg', 
                    'focus_entropy', 'focus_normalized', 'sparsity', 'uniformity']
    print(df_sorted[cols_to_show].to_string(index=False, float_format='%.4f'))
    
    # Statistics
    print("\n" + "=" * 80)
    print("STATISTICS FOR THRESHOLD CALIBRATION")
    print("=" * 80)
    
    metrics_to_analyze = ['confidence_max', 'confidence_avg', 'focus_entropy', 
                          'focus_normalized', 'sparsity', 'uniformity', 'distribution_median']
    
    for metric in metrics_to_analyze:
        values = df[metric]
        print(f"\n📈 {metric}:")
        print(f"   Min:     {values.min():.4f}")
        print(f"   25th %:  {values.quantile(0.25):.4f}")
        print(f"   Median:  {values.median():.4f}")
        print(f"   75th %:  {values.quantile(0.75):.4f}")
        print(f"   Max:     {values.max():.4f}")
        print(f"   Mean:    {values.mean():.4f}")
        print(f"   Std:     {values.std():.4f}")
    
    # Correlation with length
    print("\n" + "=" * 80)
    print("CORRELATION WITH SEQUENCE LENGTH")
    print("=" * 80)
    
    for metric in metrics_to_analyze:
        corr = df['num_tokens'].corr(df[metric])
        print(f"  {metric}: {corr:+.3f}")
    
    # Suggested thresholds
    print("\n" + "=" * 80)
    print("SUGGESTED THRESHOLDS (based on quartiles)")
    print("=" * 80)
    
    for metric in metrics_to_analyze:
        values = df[metric]
        low = values.quantile(0.33)
        high = values.quantile(0.67)
        print(f"\n  {metric}:")
        print(f"    Low:    < {low:.4f}")
        print(f"    Medium: {low:.4f} - {high:.4f}")
        print(f"    High:   > {high:.4f}")
    
    # Save results to CSV
    output_path = os.path.join(os.path.dirname(__file__), 'metrics_calibration_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    main()
