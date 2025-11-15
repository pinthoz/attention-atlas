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
cd llm-attention-explained
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
python test.py
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
llm-attention-explained/
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

## Troubleshooting

### Favicon not showing in browser tab

If the icon doesn't appear as a favicon in the browser tab (but appears correctly in the sidebar), this is a **browser caching issue**. Follow these steps:

#### Step 1: Clear Browser Cache (CRITICAL)
Browsers cache favicons very aggressively and often ignore normal page refreshes.

- **Chrome/Edge**:
  1. Press `Ctrl+Shift+Del`
  2. Select "Cached images and files"
  3. Click "Clear data"

- **Firefox**:
  1. Press `Ctrl+Shift+Del`
  2. Select "Cache"
  3. Click "Clear Now"

- **Safari**:
  1. Preferences > Advanced > Show Develop menu
  2. Develop > Empty Caches

#### Step 2: Hard Refresh
After clearing cache:
- `Ctrl+F5` (Windows/Linux) or `Cmd+Shift+R` (Mac)
- Close and reopen the browser tab

#### Step 3: Verify the Server is Serving the Icon
1. Open this URL in your browser: `http://localhost:8000/static/favicon.ico`
2. You should see the icon displayed
3. If not, restart the Shiny application (`python test.py`)

#### Step 4: Test in Incognito/Private Mode
- Open a private/incognito window (no cache)
- Navigate to `http://localhost:8000`
- The favicon should appear immediately

#### Step 5: Close ALL Browser Windows
Sometimes browsers share cache across tabs/windows:
1. Close ALL browser windows completely
2. Reopen browser
3. Navigate to `http://localhost:8000`

#### Why This Happens

**Root cause**: Browsers cache favicons extremely aggressively (sometimes for weeks) because they're static assets that rarely change. The original code used data URLs (base64) which some browsers block for security reasons. The current implementation uses the standard path `/static/favicon.ico`.

**Technical details**:
- Shiny serves static files from `static_assets` folder (configured in `app.py`)
- The favicon is accessible at `/static/favicon.ico`
- Multiple `<link>` tags are used for better browser compatibility (16x16, 32x32 sizes)

**Note**: The icon works correctly in the sidebar because it's loaded dynamically as an `<img>` tag, which doesn't use browser favicon cache.

## Contributing

Contributions are welcome! Please:
1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available for educational and research use.

## Author

Developed as part of a Master's thesis focused on transformer model interpretability and NLP education.

## Acknowledgments

- HuggingFace Transformers team
- Shiny for Python developers
- Attention mechanisms research community
