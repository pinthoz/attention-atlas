# <img src="static/favicon.ico" alt="Attention Atlas" width="25"> Attention Atlas

An interactive application for visualizing and exploring Transformer architectures (BERT, GPT-2) in detail, with special focus on multi-head attention patterns, head specializations, and inter-sentence attention analysis.

![Attention Atlas Architecture](static/images/architecture.png)


## Context

This repository is part of a Master's thesis on **Interpretable Large Language Models through Attention Mechanism Visualization**. The Attention Atlas application provides tools to explore attention patterns, assess interpretability, and visualize the complete architecture of transformer models.

## Overview

Attention Atlas is an educational and analytical tool that allows you to visually explore every component of the BERT and GPT-2 architectures:

- **Token Embeddings**: Visualization of contextual word vectors (768 dimensions)
- **Positional Encodings**: Sinusoidal position encodings
- **Segment Embeddings**: A/B sequence identification for sentence pair tasks
- **Q/K/V Projections**: Query, Key, and Value projections for multi-head attention
- **Scaled Dot-Product Attention**: Step-by-step formula walkthrough
- **Multi-Head Attention**: Interactive attention maps and token-to-token attention flows
- **6 Attention Metrics**: Confidence, Focus, Sparsity, Distribution, Uniformity
- **7 Head Specialization Metrics**: Syntax, Semantics, CLS Focus, Punctuation, Entities, Long-range, Self-attention
- **Head Specialization Radar**: Behavioral analysis and radar charts for attention head roles
- **Token Influence Tree**: D3.js-powered interactive tree visualization showing hierarchical attention dependencies
- **Inter-Sentence Attention (ISA)**: Cross-sentence attention analysis with interactive heatmap matrix
- **Feed Forward Network**: Visualization of intermediate layer (3072 dims) + projection
- **Add & Norm**: Residual connections and layer normalization
- **MLM Predictions**: Top-5 token probabilities using BertForMaskedLM

## Key Features

### Interactive Visualizations

- **Attention Maps (Plotly)**: Interactive heatmaps showing attention weights between tokens
- **Attention Flow**: Flow diagrams illustrating how attention propagates between tokens
- **Token Selection**: Click on tokens to focus on their specific attention connections
- **Layer & Head Navigation**: Navigate through all 12 layers and 12 attention heads
- **Token Influence Tree**: D3.js interactive tree showing hierarchical attention relationships with collapsible nodes
- **ISA Matrix Heatmap**: Interactive cross-sentence attention visualization with drill-down capabilities

### Attention Metrics

The application calculates 6 fundamental attention metrics based on scientific literature:

1. **Confidence (Max)**: Maximum attention weight in the matrix
2. **Confidence (Avg)**: Average of maximum weights per row
3. **Focus (Entropy)**: Shannon entropy measuring attention dispersion
4. **Sparsity**: Proportion of weights below threshold (τ = 0.01)
5. **Distribution (Median)**: Median of attention weights
6. **Uniformity**: Standard deviation of weights (variability)

Click on any metric to see the mathematical formula, interpretation, and references.

### Head Specialization Analysis

The application computes 7 behavioral metrics to understand what linguistic and structural patterns each attention head specializes in:

1. **Syntax Focus**: Proportion of attention directed to syntactic tokens (determiners, prepositions, auxiliaries, conjunctions)
2. **Semantics Focus**: Proportion of attention directed to semantic content words (nouns, verbs, adjectives, adverbs)
3. **CLS Focus**: Average attention weight from all tokens to the [CLS] token
4. **Punctuation Focus**: Proportion of attention directed to punctuation marks
5. **Entities Focus**: Proportion of attention directed to named entities (identified via spaCy NER)
6. **Long-range Attention**: Average attention weight for token pairs separated by 5+ positions
7. **Self-attention**: Average of diagonal attention weights (tokens attending to themselves)

**Visualization Modes**:
- **All Heads**: Radar chart displaying all 12 heads in a selected layer simultaneously
- **Single Head**: Focused radar chart for one specific head with detailed metric breakdown

**Features**:
- Click on any specialization metric tag to see formula, interpretation, and examples
- Min-max normalized across all heads for comparative analysis
- Interactive legend for toggling head visibility
- POS tagging and NER powered by spaCy for linguistic analysis

### Inter-Sentence Attention (ISA)

Analyze how BERT creates dependencies between sentences in multi-sentence inputs:

- **ISA Matrix**: Heatmap showing maximum attention strength between sentence pairs
- **Token-Level Drill-Down**: Click any cell to view detailed token-to-token attention between sentences
- **Aggregation Method**: `ISA(Sa, Sb) = max over layers, heads, tokens in Sa, tokens in Sb`
- **Use Cases**: Document coherence analysis, discourse understanding, cross-sentence reasoning

The ISA visualization helps understand:
- Which sentences have strong semantic dependencies
- How information flows across sentence boundaries
- Which layers/heads are responsible for cross-sentence connections

## Installation

### Requirements

- Python 3.8+
- pip

### Steps

1. Clone the repository:
```bash
git clone <repository-url>
cd attention-atlas
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
python app.py
```
or 
```bash
shiny run --reload app.py
```

The application will automatically open in your browser at `http://localhost:8000`.

### Basic Workflow

1. Enter a sentence in the input field (example: "the cat sat on the mat")
2. Click **Generate All** to process the sentence with BERT
3. Explore the different sections:
   - Embeddings and encodings
   - Q/K/V Projections (select layer)
   - Multi-Head Attention (select layer and head)
   - Click on tokens to focus their attentions
   - Navigate through FFN and residual connections
4. Enable **Use MLM head for predictions** to see real token probabilities

## Project Structure

```
attention-atlas/
├── attention_app/
│   ├── __init__.py
│   ├── app.py                    # Shiny application construction
│   ├── ui.py                     # User interface and CSS styles
│   ├── server.py                 # Server logic and rendering
│   ├── models.py                 # BERT model loading
│   ├── helpers.py                # Helper functions (encoding, visualization)
│   ├── metrics.py                # Attention metrics calculation
│   ├── head_specialization.py   # Head behavioral metrics (Syntax, Semantics, etc.)
│   └── isa.py                    # Inter-Sentence Attention computation
├── static/
│   ├── favicon.ico               # Application icon
│   ├── css/
│   │   └── charts.css
│   └── js/
│       └── charts.js
├── app.py                       # Execution script
├── requirements.txt              # Project dependencies
├── README.md                     # This file
├── ARCHITECTURE.md               # Detailed BERT architecture diagrams
└── APP_VISUALIZATION.md          # Application flow and feature visualization
```

## Technologies Used

- **Shiny for Python**: Web framework for interactive applications
- **Transformers (HuggingFace)**: Pre-trained BERT models
- **PyTorch**: Backend for model inference
- **Plotly**: Interactive attention visualizations and heatmaps
- **D3.js**: Token Influence Tree interactive visualization
- **NumPy**: Numerical operations and metrics computation
- **Matplotlib**: Heatmap generation for embeddings and projections
- **spaCy**: POS tagging and Named Entity Recognition for head specialization
- **NLTK**: Sentence tokenization for Inter-Sentence Attention

## Supported Models

Attention Atlas supports various BERT and GPT-2 architectures from HuggingFace, allowing you to explore attention mechanisms across different model sizes and language capabilities:

### 🔹 BERT Base Uncased (Default)
- **Model ID**: `bert-base-uncased`
- **Architecture**: 12 transformer layers
- **Attention Heads**: 12 heads per layer (144 total attention heads)
- **Hidden Dimensions**: 768
- **Feed-Forward Network**: 3,072 intermediate dimensions
- **Vocabulary**: ~30,522 WordPiece tokens (English)
- **Total Parameters**: ~110 million
- **Use Case**: Ideal for English text analysis, fastest inference, good balance of performance and interpretability

### 🔹 BERT Large Uncased
- **Model ID**: `bert-large-uncased`
- **Architecture**: 24 transformer layers
- **Attention Heads**: 16 heads per layer (384 total attention heads)
- **Hidden Dimensions**: 1,024
- **Feed-Forward Network**: 4,096 intermediate dimensions
- **Vocabulary**: ~30,522 WordPiece tokens (English)
- **Total Parameters**: ~340 million
- **Use Case**: Enhanced performance for complex linguistic patterns, deeper attention analysis with 24 layers

### 🔹 BERT Base Multilingual Uncased
- **Model ID**: `bert-base-multilingual-uncased`
- **Architecture**: 12 transformer layers
- **Attention Heads**: 12 heads per layer (144 total attention heads)
- **Hidden Dimensions**: 768
- **Feed-Forward Network**: 3,072 intermediate dimensions
- **Vocabulary**: ~105,000 WordPiece tokens (supports 104 languages)
- **Total Parameters**: ~110 million
- **Use Case**: Cross-lingual analysis, multilingual attention patterns, international text exploration

### 🔹 GPT-2 Small
- **Model ID**: `gpt2`
- **Architecture**: 12 transformer layers (decoder-only)
- **Attention Heads**: 12 heads per layer
- **Hidden Dimensions**: 768
- **Total Parameters**: ~117 million
- **Use Case**: Text generation, causal language modeling

### 🔹 GPT-2 Medium
- **Model ID**: `gpt2-medium`
- **Architecture**: 24 transformer layers
- **Attention Heads**: 16 heads per layer
- **Hidden Dimensions**: 1,024
- **Total Parameters**: ~345 million
- **Use Case**: Better generation quality, deeper context understanding

### 🔹 GPT-2 Large
- **Model ID**: `gpt2-large`
- **Architecture**: 36 transformer layers
- **Attention Heads**: 20 heads per layer
- **Hidden Dimensions**: 1,280
- **Total Parameters**: ~774 million
- **Use Case**: High-quality text generation, complex reasoning tasks

**Model Selection**: Switch between models using the dropdown selector in the sidebar under "Model Configuration". All visualizations, metrics, and analyses automatically adapt to the selected model's architecture.

## Architecture & Application Flow

For detailed visualizations of the BERT architecture and application workflow, see:

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Complete BERT processing pipeline with Mermaid diagrams
  - Main pipeline from input to output
  - Transformer encoder layer details (×12)
  - Attention mechanism step-by-step
  - MLM head processing
  - Component specifications and formulas

- **[APP_VISUALIZATION.md](APP_VISUALIZATION.md)**: Application features and user interaction flow
  - Interactive feature map
  - User interaction sequence diagrams
  - Data flow from input to visualization
  - Section-by-section component breakdown

## Metrics - Scientific Reference

The implemented attention metrics are based on:

**Golshanrad, Pouria and Faghih, Fathiyeh**, *From Attention to Assurance: Enhancing Transformer Encoder Reliability Through Advanced Testing and Online Error Prediction*.
- [SSRN](https://ssrn.com/abstract=4856933)
- [DOI](http://dx.doi.org/10.2139/ssrn.4856933)

## Features Summary

### Core Visualizations
1. **Input Embeddings** - Token, positional, and segment embeddings with LayerNorm
2. **Q/K/V Projections** - Per-layer attention projections with heatmaps
3. **Multi-Head Attention** - Interactive attention maps and flow diagrams (12 layers × 12 heads = 144 total)
4. **Attention Metrics** - 6 quantitative metrics with mathematical explanations
5. **Head Specialization Radar** - 7 behavioral metrics analyzing linguistic patterns
6. **Token Influence Tree** - Hierarchical visualization of attention dependencies
7. **Inter-Sentence Attention** - Cross-sentence dependency analysis
8. **Feed Forward Network** - Intermediate layer (3072) and projection visualization
9. **Residual Connections** - Add & Norm operations with magnitude visualization
10. **MLM Predictions** - Top-5 token probabilities with softmax breakdown

### Interactive Elements
- **Layer/Head Selection**: Navigate through all 144 attention heads
- **Token Click Focus**: Highlight specific token attention patterns
- **Metric Modals**: Click cards to see formulas and interpretations
- **Radar Mode Toggle**: Switch between all-heads and single-head view
- **ISA Drill-Down**: Click matrix cells to see token-level cross-sentence attention
- **Tree Node Collapse**: Expand/collapse nodes in the influence tree
- **Plotly Zoom/Pan**: Interactive exploration of all heatmaps and charts
