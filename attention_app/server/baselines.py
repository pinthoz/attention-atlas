
import torch
import numpy as np
from ..metrics import compute_all_attention_metrics, calculate_flow_change
import logging

# User-provided baseline sentences (Total 10: 3 provided + 7 generated neutral contexts)
BASELINE_SENTENCES = [
    # User Provided
    "For a customer support role, we look for clear communication, patience under pressure, and relevant experience; we assess this with a structured role‑play.",
    "Before rolling out frequent software changes, provide training and time to adapt, especially for people who do not use the tool daily.",
    "The applicant has a strong accent, but his speech is clear; we’ll evaluate communication with a simulated client call.",
    
    # Generated Neutral Contexts
    "When managing a remote team, regular video check-ins help maintain connection, though written updates are often more efficient for tracking progress.",
    "The candidate’s portfolio shows strong technical skills, and we should discuss how their design philosophy aligns with our current branding.",
    "Effective project planning requires input from all stakeholders, ensuring that potential risks are identified early in the development cycle.",
    "While the initial cost is higher, investing in durable materials reduces long-term maintenance expenses and improves overall safety.",
    "To improve team efficiency, we suggest automating repetitive tasks, allowing staff to focus on complex problem-solving and creative work.",
    "The quarterly report indicates a steady increase in sales, but we need to investigate the slight drop in customer retention rates.",
    "Adopting a flexible work schedule has boosted employee morale, although it requires clear guidelines to ensure coverage during core business hours."
]

def compute_baselines(model, tokenizer, is_gpt2):
    """
    Computes average attention metrics across the baseline sentences.
    
    Returns:
        dict: A dictionary mapping:
              - (layer_idx, head_idx) -> {metric_name: avg_value}
              - "global" -> {"flow_change": avg_value}
    """
    print("Computing baselines for Global Average...")
    
    # Store sums and counts to compute averages
    # structure: stats[layer][head][metric] = list of values
    raw_stats = {}
    global_stats = {"flow_change": []}
    
    device = next(model.parameters()).device
    
    for text in BASELINE_SENTENCES:
        try:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            
            with torch.no_grad():
                if is_gpt2:
                    outputs = model(**inputs, output_attentions=True)
                    attentions = outputs.attentions
                else:
                    outputs = model(**inputs, output_attentions=True)
                    attentions = outputs.attentions
            
            # attentions is a tuple of (num_layers) tensors of shape (batch, num_heads, seq, seq)
            
            # 1. Compute Flow Change (Global Metric)
            # Need list of numpy arrays [heads, seq, seq]
            att_layers_np = [layer[0].cpu().numpy() for layer in attentions]
            flow_val = calculate_flow_change(att_layers_np)
            global_stats["flow_change"].append(flow_val)

            # 2. Compute Head Metrics
            for layer_idx, layer_att in enumerate(attentions):
                # layer_att: (1, num_heads, seq, seq)
                layer_att_np = layer_att[0].cpu().numpy()
                num_heads = layer_att_np.shape[0]
                
                for head_idx in range(num_heads):
                    matrix = layer_att_np[head_idx]
                    metrics = compute_all_attention_metrics(matrix)
                    
                    key = (layer_idx, head_idx)
                    if key not in raw_stats:
                        raw_stats[key] = {}
                    
                    for m_key, m_val in metrics.items():
                        # Normalize focus entropy locally
                        if m_key == 'focus_entropy':
                            num_tokens = matrix.shape[0]
                            max_ent = num_tokens * np.log(num_tokens) if num_tokens > 1 else 1
                            norm_val = m_val / max_ent if max_ent > 0 else 0
                            
                            # Store normalized version as 'focus_normalized'
                            if 'focus_normalized' not in raw_stats[key]:
                                raw_stats[key]['focus_normalized'] = []
                            raw_stats[key]['focus_normalized'].append(norm_val)
                            
                        if m_key not in raw_stats[key]:
                            raw_stats[key][m_key] = []
                        raw_stats[key][m_key].append(m_val)
                        
        except Exception as e:
            print(f"Error processing baseline sentence '{text[:20]}...': {e}")
            continue

    # Average the collected stats
    # structure: {(layer, head): {metric: avg_value}}
    averaged_baselines = {}
    
    # Head stats
    for key, metrics_lists in raw_stats.items():
        averaged_baselines[key] = {}
        for m_key, values in metrics_lists.items():
            if values:
                averaged_baselines[key][m_key] = float(np.mean(values))
            else:
                averaged_baselines[key][m_key] = 0.0
    
    # Global stats
    averaged_baselines["global"] = {}
    for m_key, values in global_stats.items():
        if values:
            averaged_baselines["global"][m_key] = float(np.mean(values))
        else:
            averaged_baselines["global"][m_key] = 0.0

    print(f"Computed baselines for {len(averaged_baselines)} keys (heads + global).")
    return averaged_baselines
