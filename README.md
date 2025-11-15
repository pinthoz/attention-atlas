# <img src="static/favicon.ico" alt="Attention Atlas" width="25"> Attention Atlas

An interactive application for visualizing and exploring the BERT architecture in detail, with special focus on multi-head attention patterns.


## Context

This repository is part of a Master's thesis on **Interpretable Large Language Models through Attention Mechanism Visualization**. The Attention Atlas application provides tools to explore attention patterns, assess interpretability, and visualize the complete architecture of transformer models.

## Overview

Attention Atlas is an educational and analytical tool that allows you to visually explore every component of the BERT architecture:

- **Token Embeddings**: Visualization of contextual word vectors (768 dimensions)
- **Positional Encodings**: Sinusoidal position encodings
- **Segment Embeddings**: A/B sequence identification for sentence pair tasks
- **Q/K/V Projections**: Query, Key, and Value projections for multi-head attention
- **Scaled Dot-Product Attention**: Step-by-step formula walkthrough
- **Multi-Head Attention**: Interactive attention maps and token-to-token attention flows
- **6 Attention Metrics**: Confidence, Focus, Sparsity, Distribution, Uniformity
- **Feed Forward Network**: Visualization of intermediate layer (3072 dims) + projection
- **Add & Norm**: Residual connections and layer normalization
- **MLM Predictions**: Top-5 token probabilities using BertForMaskedLM

## Key Features

### Interactive Visualizations

- **Attention Maps (Plotly)**: Interactive heatmaps showing attention weights between tokens
- **Attention Flow**: Flow diagrams illustrating how attention propagates between tokens
- **Token Selection**: Click on tokens to focus on their specific attention connections
- **Layer & Head Navigation**: Navigate through all 12 layers and 12 attention heads

### Attention Metrics

The application calculates 6 fundamental attention metrics based on scientific literature:

1. **Confidence (Max)**: Maximum attention weight in the matrix
2. **Confidence (Avg)**: Average of maximum weights per row
3. **Focus (Entropy)**: Shannon entropy measuring attention dispersion
4. **Sparsity**: Proportion of weights below threshold (τ = 0.01)
5. **Distribution (Median)**: Median of attention weights
6. **Uniformity**: Standard deviation of weights (variability)

Click on any metric to see the mathematical formula, interpretation, and references.

### Dark Theme Interface

- Fixed sidebar with controls and description
- Modern design with gradients and animations
- Dark/pink theme to reduce visual fatigue
- Animated loading spinners

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
│   ├── app.py              # Shiny application construction
│   ├── ui.py               # User interface and CSS styles
│   ├── server.py           # Server logic and rendering
│   ├── models.py           # BERT model loading
│   ├── helpers.py          # Helper functions (encoding, visualization)
│   └── metrics.py          # Attention metrics calculation
├── static/
│   ├── favicon.ico         # Application icon
│   ├── css/
│   │   └── charts.css
│   └── js/
│       └── charts.js
├── test.py                 # Execution script
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## Technologies Used

- **Shiny for Python**: Web framework for interactive applications
- **Transformers (HuggingFace)**: Pre-trained BERT models
- **PyTorch**: Backend for model inference
- **Plotly**: Interactive attention visualizations
- **NumPy**: Numerical operations and metrics
- **Matplotlib**: Heatmap generation

## Model

The project uses **bert-base-uncased** from HuggingFace:
- 12 layers
- 12 attention heads per layer
- 768 embedding dimensions
- 3072 intermediate dimensions (FFN)
- ~30k token vocabulary

## Metrics - Scientific Reference

The implemented attention metrics are based on:

**Golshanrad, Pouria and Faghih, Fathiyeh**, *From Attention to Assurance: Enhancing Transformer Encoder Reliability Through Advanced Testing and Online Error Prediction*.
- [SSRN](https://ssrn.com/abstract=4856933)
- [DOI](http://dx.doi.org/10.2139/ssrn.4856933)
