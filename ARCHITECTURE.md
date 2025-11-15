# BERT Architecture Pipeline

This document illustrates the complete BERT processing pipeline as visualized in Attention Atlas.

## Main Pipeline

```mermaid
flowchart TD
    Start[Input Text: 'the cat sat on the mat'] --> Tokenize[Tokenization]
    Tokenize --> Tokens["[CLS] the cat sat on the mat [SEP]"]

    Tokens --> TokenEmb[Token Embeddings<br/>768 dims]
    Tokens --> SegEmb[Segment Embeddings<br/>A/B sequences]
    Tokens --> PosEmb[Positional Embeddings<br/>Learned positions]

    TokenEmb --> Sum[Sum + LayerNorm]
    SegEmb --> Sum
    PosEmb --> Sum

    Sum --> Encoder[Transformer Encoder<br/>12 Layers]

    Encoder --> Hidden[Hidden States<br/>768 dims]

    Hidden --> CLS[CLS Output<br/>Sequence Tasks]
    Hidden --> MLM[MLM Head<br/>Token Predictions]

    MLM --> TopK[Top-5 Probabilities<br/>per Token]

    style Start fill:#e1f5ff
    style Encoder fill:#ffe1e1
    style MLM fill:#e1ffe1
    style TopK fill:#ffe1f5
```

## Transformer Encoder Layer (×12)

```mermaid
flowchart TD
    Input[Layer Input<br/>768 dims] --> MHA[Multi-Head Attention]

    subgraph MultiHead [Multi-Head Attention Block]
        MHA --> QKV[Q/K/V Projections]
        QKV --> Q[Query]
        QKV --> K[Key]
        QKV --> V[Value]

        Q --> Scaled[Scaled Dot-Product<br/>Q·K^T / √d_k]
        K --> Scaled
        Scaled --> Softmax[Softmax]
        Softmax --> Attention[Attention Weights]

        Attention --> Metrics[Compute Metrics]

        subgraph MetricsBox [6 Attention Metrics]
            M1[Confidence Max]
            M2[Confidence Avg]
            M3[Focus Entropy]
            M4[Sparsity]
            M5[Distribution]
            M6[Uniformity]
        end

        Attention --> Heads[12 Parallel Heads]
        V --> Heads
        Heads --> Concat[Concatenate & Project]
    end

    Concat --> Add1[Add & Norm<br/>Residual Connection]
    Input --> Add1

    Add1 --> FFN[Feed Forward Network]

    subgraph FFNBlock [Feed Forward Network]
        FFN --> Linear1[Linear 768→3072]
        Linear1 --> GELU[GELU Activation]
        GELU --> Linear2[Linear 3072→768]
    end

    Linear2 --> Add2[Add & Norm<br/>Residual Connection]
    Add1 --> Add2

    Add2 --> Output[Layer Output<br/>768 dims]

    style MultiHead fill:#fff4e6
    style FFNBlock fill:#e6f7ff
    style MetricsBox fill:#f0e6ff
```

## Attention Mechanism Detail

```mermaid
flowchart LR
    subgraph Input
        Q[Query Matrix<br/>n × d_k]
        K[Key Matrix<br/>n × d_k]
        V[Value Matrix<br/>n × d_v]
    end

    Q --> Dot[Matrix<br/>Multiplication<br/>Q·K^T]
    K --> Dot

    Dot --> Scale[Scale by<br/>√d_k]
    Scale --> SM[Softmax]
    SM --> Att[Attention<br/>Weights<br/>n × n]

    Att --> Mul[Matrix<br/>Multiplication]
    V --> Mul

    Mul --> Out[Output<br/>n × d_v]

    style Att fill:#ffe6e6
    style Out fill:#e6ffe6
```

## MLM Head Processing

```mermaid
flowchart TD
    Hidden[Hidden States<br/>seq_len × 768] --> Linear[Linear Projection<br/>768 → 30,522]
    Linear --> Logits[Logits<br/>seq_len × vocab_size]
    Logits --> Softmax[Softmax]
    Softmax --> Probs[Probabilities<br/>seq_len × vocab_size]
    Probs --> TopK[Top-5 Selection]
    TopK --> Output[Token Predictions<br/>per Position]

    style Hidden fill:#e6f7ff
    style Probs fill:#ffe6f7
    style Output fill:#e6ffe6
```

## Component Details

### 1. Input Processing
- **Tokenization**: Text → Token IDs using BERT WordPiece tokenizer
- **Special Tokens**: `[CLS]` at start, `[SEP]` at end
- **Maximum Length**: 512 tokens

### 2. Embedding Layer
Three types of embeddings are summed:
- **Token Embeddings**: Learned representations (30,522 vocab → 768 dims)
- **Positional Embeddings**: Learned position information (0-511 → 768 dims)
- **Segment Embeddings**: Sentence A/B distinction (2 → 768 dims)

### 3. Transformer Encoder (12 Layers)

#### Multi-Head Attention
- **Heads**: 12 parallel attention mechanisms per layer
- **Head Dimension**: 64 (768 / 12)
- **Process**:
  1. Project input to Q, K, V matrices
  2. Compute attention: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
  3. Concatenate heads and project back

#### Attention Metrics (calculated per head)
- **Confidence (Max)**: `max(attention_matrix)`
- **Confidence (Avg)**: `mean(max_per_row)`
- **Focus (Entropy)**: `-Σ(p·log(p))`
- **Sparsity**: `% weights < 0.01`
- **Distribution**: `median(attention_weights)`
- **Uniformity**: `std(attention_weights)`

#### Feed Forward Network
- **Expansion**: 768 → 3,072 dimensions
- **Activation**: GELU (Gaussian Error Linear Unit)
- **Compression**: 3,072 → 768 dimensions

#### Residual Connections
- Applied after attention: `output = LayerNorm(input + attention(input))`
- Applied after FFN: `output = LayerNorm(input + FFN(input))`

### 4. Output Layer

#### MLM (Masked Language Modeling) Head
- **Linear**: 768 → 30,522 (vocab size)
- **Softmax**: Convert logits to probabilities
- **Top-5**: Display most likely tokens per position

## Visualization Features

### Interactive Elements
1. **Layer Selection**: Navigate through 12 encoder layers
2. **Head Selection**: Explore individual attention heads (12 per layer)
3. **Token Focus**: Click tokens to highlight attention patterns
4. **Attention Flow**: Visualize token-to-token connections
5. **Heatmaps**: Color-coded attention weights

### Data Displayed
- Token embeddings (first 64 dims visualized)
- Positional encodings (sinusoidal patterns)
- Q/K/V projections (first 48 dims per token)
- Attention matrices (all token pairs)
- FFN intermediate activations (first 96 dims)
- Hidden states evolution across layers
- Token prediction probabilities

## Technical Specifications

- **Model**: `bert-base-uncased`
- **Parameters**: ~110M
- **Layers**: 12
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Intermediate Size**: 3,072
- **Vocabulary**: 30,522 WordPiece tokens
- **Max Sequence Length**: 512

## References

All visualizations and metrics are computed in real-time from the actual BERT model using:
- `transformers` library (HuggingFace)
- `torch` for model inference
- `plotly` for interactive visualizations
