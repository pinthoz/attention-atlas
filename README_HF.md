---
title: Attention Atlas
emoji: 🌍
colorFrom: pink
colorTo: blue
sdk: docker
pinned: false
license: mit
short_description: Interpretable AI through attention mechanism visualization
app_port: 8000
---

# Attention Atlas 🌍

**Interpretable Large Language Models through Attention Mechanism Visualization**

An interactive research platform for exploring Transformer architectures (BERT, GPT-2) with comprehensive mechanistic interpretability tools, attention dynamics analysis, and bias detection capabilities.

---

## Overview

Attention Atlas is a Master's thesis project dedicated to advancing **interpretable AI** by providing systematic visualization and analysis of attention mechanisms in Transformer-based language models. The platform offers researchers, educators, and practitioners an interactive environment to explore multi-head attention dynamics, linguistic feature extraction, and ethical considerations in model behavior.

### Three-Level Exploration Structure

The application organizes analysis into **three progressive levels** for comprehensive model understanding:

#### 🔍 **Overview Section**
High-level model behavior at a glance:
- **Global Attention Metrics**: 6 quantitative metrics (Confidence, Focus, Sparsity, Uniformity, Distribution, Balance)
- **MLM Predictions**: Top-5 token predictions with probability breakdowns
- **Radar Metrics**: Multi-dimensional visualization of attention patterns
- **Hidden States**: Final layer embeddings and PCA projections

#### 🎯 **Explore Attention Section**
Interactive, granular attention analysis:
- **Attention Heatmaps**: Interactive Plotly visualizations (12 layers × 12 heads = 144 total)
- **Attention Flow Diagrams**: Sankey-style token-to-token connection flows
- **Token Influence Tree**: D3.js hierarchical multi-hop attention dependencies
- **Inter-Sentence Attention (ISA)**: Cross-sentence dependency analysis with drill-down capabilities
- **Scaled Attention**: Step-by-step computation visualization (Q·K^T / √d_k)

#### ⚙️ **Deep Dive Section** (Advanced Mode)
Component-level Transformer pipeline inspection:
- **Token Embeddings**: 768-dim vectors with PCA visualization and cosine similarity
- **Positional & Segment Encodings**: Learned position patterns and sentence pair distinction
- **Q/K/V Projections**: Query, Key, Value matrices with compatibility scores
- **Residual Connections**: Add & Norm operations with magnitude visualizations
- **Feed-Forward Network**: 768 → 3,072 → 768 transformation with GELU activation
- **Head Clustering**: t-SNE + K-Means automatic grouping into behavioral clusters

#### 🛡️ **Bias Detection** (Coming Soon)
Ethical AI analysis and bias quantification:
- Token-level bias classification (Generalization, Unfairness, Stereotypes)
- Attention-bias interaction analysis (which heads amplify/mitigate bias)
- Bias propagation tracking across layers

---

## Key Features

### Quantitative Analysis
- **6 Attention Metrics**: Scientifically-grounded measurements (Golshanrad & Faghih, 2024)
- **7 Head Specialization Metrics**: Linguistic pattern analysis (Syntax, Semantics, CLS Focus, Punctuation, Entities, Long-Range, Self-Attention)
- **Clickable Metric Cards**: Mathematical formulas, interpretations, and scientific references

### Interactive Visualizations
- **144-384 Attention Heads**: Full coverage of BERT-base (144 heads) to BERT-large (384 heads)
- **Real-Time Inference**: Authentic model computations using PyTorch backend
- **Hover Details**: Q·K dot product, scaled values, softmax results
- **Token Focus**: Click tokens to highlight attention patterns
- **Zoom/Pan Controls**: Plotly interactive exploration

### Multi-Model Support
- **BERT Family**: base-uncased, large-uncased, multilingual (104 languages)
- **GPT-2 Family**: Small, Medium, Large (causal/decoder-only architectures)
- **Automatic Adaptation**: UI adjusts to model architecture (12/24 layers, 12/16/20 heads)

### Educational Value
- **Mathematical Formulas**: LaTeX-style formatting with detailed explanations
- **Scientific Grounding**: References to original papers (Vaswani, Devlin, Clark, etc.)
- **Interpretation Guides**: High/low value meanings, use cases, examples
- **Complete Transparency**: Every component from input to output visualized

---

## Supported Models

| Model | Layers | Heads | Hidden | FFN | Vocab | Params | Total Heads |
|-------|--------|-------|--------|-----|-------|--------|-------------|
| **BERT-base-uncased** | 12 | 12 | 768 | 3,072 | 30,522 | ~110M | 144 |
| **BERT-large-uncased** | 24 | 16 | 1,024 | 4,096 | 30,522 | ~340M | 384 |
| **BERT-multilingual** | 12 | 12 | 768 | 3,072 | 105,000 | ~110M | 144 |
| **GPT-2 Small** | 12 | 12 | 768 | 3,072 | 50,257 | ~117M | 144 |
| **GPT-2 Medium** | 24 | 16 | 1,024 | 4,096 | 50,257 | ~345M | 384 |
| **GPT-2 Large** | 36 | 20 | 1,280 | 5,120 | 50,257 | ~774M | 720 |
| **GPT-2 XL** | 48 | 25 | 1,600 | 6,400 | 50,257 | ~1.5B | 1,200 |

---

## Technologies

- **Backend**: Shiny for Python, Transformers (HuggingFace), PyTorch, NumPy, spaCy, NLTK, scikit-learn
- **Frontend**: Plotly (heatmaps, radar charts), D3.js (tree visualization), HTML/CSS, JavaScript
- **NLP**: spaCy (POS tagging, NER), NLTK (sentence segmentation)
- **ML**: t-SNE dimensionality reduction, K-Means clustering, Silhouette Score

---

## Usage

1. **Enter text input**: Type or paste text in the sidebar
2. **Select model**: Choose BERT or GPT-2 variant
3. **Generate analysis**: Click "Generate All" to process
4. **Explore sections**:
   - **Overview**: Global metrics and predictions
   - **Explore Attention**: Layer/head navigation, token focus, attention flows
   - **Deep Dive**: Component-level inspection (Advanced mode)
5. **Compare models** (Optional): Side-by-side analysis (Model A vs. Model B)

---

## Research Applications

- **Mechanistic Interpretability**: Understanding how attention mechanisms process language
- **Head Specialization Analysis**: Identifying linguistic roles of attention heads
- **Discourse Understanding**: Cross-sentence dependency analysis via ISA
- **Model Comparison**: Side-by-side architectural behavior analysis
- **Educational Tool**: Teaching transformer architecture and attention mechanisms
- **Bias Detection**: Ethical AI auditing and bias quantification (coming soon)

---

## Scientific Grounding

### Attention Metrics
Based on: **Golshanrad & Faghih (2024)**, *From Attention to Assurance: Enhancing Transformer Encoder Reliability Through Advanced Testing and Online Error Prediction*
- [SSRN](https://ssrn.com/abstract=4856933) | [DOI](http://dx.doi.org/10.2139/ssrn.4856933)

### Architecture
- **Vaswani et al. (2017)**: *Attention Is All You Need*, NeurIPS (Original Transformer)
- **Devlin et al. (2019)**: *BERT: Pre-training of Deep Bidirectional Transformers*, NAACL
- **Radford et al. (2019)**: *Language Models are Unsupervised Multitask Learners* (GPT-2)

### Attention Analysis
- **Clark et al. (2019)**: *What Does BERT Look At? An Analysis of BERT's Attention*, ACL Workshop

---

## Documentation

- **[README.md](https://github.com/yourusername/attention-atlas/blob/main/README.md)**: Complete feature documentation, installation guide, usage instructions
- **[ARCHITECTURE.md](https://github.com/yourusername/attention-atlas/blob/main/ARCHITECTURE.md)**: Technical deep dive into BERT/GPT-2 processing pipeline with Mermaid diagrams
- **[APP_VISUALIZATION.md](https://github.com/yourusername/attention-atlas/blob/main/APP_VISUALIZATION.md)**: User guide with interaction workflows and visualization interpretations

---

## Master's Thesis Project

This application is part of a Master's thesis on **Interpretable Large Language Models through Attention Mechanism Visualization**. The project addresses critical gaps in LLM interpretability and fairness analysis identified in recent literature.

### The Problem

Large-scale Transformers (BERT, GPT-2/3/4) achieve state-of-the-art performance but remain opaque in safety-critical settings. Two key gaps persist:

1. **Faithfulness Gap**: Attention weights sometimes dissociate from gradient-based attributions, raising doubts about their reliability as explanations (Serrano & Smith, 2019)
2. **Bias Localisation**: Harmful biases are latent in specific heads, but existing tools provide either coarse heatmaps or lack interactive bias drill-down

### Research Contributions

Attention Atlas advances the state-of-the-art through four innovations:

1. **Unified Interpretability + Fairness**: First tool to integrate attention visualization with systematic bias detection (StereoSet scoring)
2. **Head-Granular Analytics**: Standardized attention extraction across BERT and GPT-2 families with t-SNE clustering
3. **Multi-Level Validation**: Couples quantitative metrics (faithfulness via Integrated Gradients, consistency, bias localization) with qualitative user study
4. **Correlation Dashboard**: Interactive visualization of Attention ↔ Performance ↔ Fairness relationships, revealing Pareto-optimal heads

### Research Questions Addressed

1. **Faithfulness**: Are attention weights faithful explanations? (Validated via Integrated Gradients comparison)
2. **Specialization**: What linguistic patterns do heads encode? (7 behavioral metrics: Syntax, Semantics, etc.)
3. **Information Flow**: How does attention propagate through layers? (Token Influence Tree, ISA Matrix)
4. **Bias Amplification**: Which heads amplify social biases? (Attention-to-bias correlation with StereoSet scores)

### Validation Strategy

- **Quantitative**: Spearman correlation between attention and gradient attributions (target: > 0.5)
- **Qualitative**: User study (N=10-15) with NLP researchers and domain experts measuring task accuracy and SUS score

---

## Citation

If you use Attention Atlas in your research, please cite:

```bibtex
@mastersthesis{attentionatlas2025,
  title={Interpretable Large Language Models through Attention Mechanism Visualization},
  author={[Your Name]},
  year={2025},
  school={[Your University]},
  type={Master's Thesis}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

**Attention Atlas** — Making Transformer attention mechanisms interpretable, one head at a time.

*Part of a Master's thesis on Interpretable AI*
