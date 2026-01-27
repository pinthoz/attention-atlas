# Transformer Architecture: Comprehensive Technical Documentation

## Overview

This document provides a complete technical description of the data flow through BERT and GPT-2 architectures, detailing each processing stage from raw text input to final predictions. The Transformer architecture, introduced by Vaswani et al. (2017), has become the foundation for modern natural language processing, replacing recurrent architectures with self-attention mechanisms that enable parallel processing and effective long-range dependency modeling.

---

## Stage 1: Input Processing and Embedding

### 1.1 Input → Token Embeddings

The first stage transforms raw text into dense numerical representations suitable for neural network processing. This involves two sequential operations: tokenization and embedding lookup.

#### 1.1.1 BERT Tokenization and Embedding

**Tokenization Process:**

BERT employs the WordPiece tokenization algorithm, a subword segmentation method that balances vocabulary size with the ability to represent rare or out-of-vocabulary words. The algorithm operates as follows:

1. The input text is first normalized (lowercasing for uncased models, unicode normalization)
2. Text is split into initial word-level tokens based on whitespace and punctuation
3. Each word is further decomposed into subword units from a learned vocabulary
4. Subword pieces other than the first receive a `##` prefix to indicate continuation

**Example:**
```
Input: "unbelievable"
Tokens: ["un", "##believ", "##able"]
```

**Special Tokens:**

BERT introduces two special tokens with specific semantic functions:
- `[CLS]` (Classification): Prepended to every input sequence. Its final hidden state serves as an aggregate sequence representation for classification tasks. During pre-training, this token learns to encode sequence-level information.
- `[SEP]` (Separator): Appended after each sentence to demarcate sentence boundaries. Essential for tasks involving sentence pairs.

**Embedding Lookup:**

Each token identifier indexes into a learned Token Embedding Matrix of dimensions `vocab_size × hidden_dim` (typically 30,522 × 768 for BERT-base). The embedding matrix is initialized randomly and trained end-to-end during pre-training.

**Output Specification:**
- Shape: `(batch_size, sequence_length, hidden_dim)`
- Data type: Float32 tensor
- Range: Unbounded (learned parameters)

#### 1.1.2 GPT-2 Tokenization and Embedding

**Tokenization Process:**

GPT-2 employs Byte-Pair Encoding (BPE), a compression-based subword algorithm that iteratively merges the most frequent character pairs. The implementation uses byte-level BPE, operating on UTF-8 byte sequences rather than unicode characters, ensuring complete coverage of any input text without unknown tokens.

**Algorithm overview:**
1. Initialize vocabulary with all individual bytes (256 base tokens)
2. Count frequency of all adjacent token pairs in training corpus
3. Merge the most frequent pair into a new token
4. Repeat until desired vocabulary size is reached (50,257 for GPT-2)

**Special Tokens:**

GPT-2 uses a minimal special token set:
- `<|endoftext|>`: Marks sequence boundaries during training and generation. Unlike BERT, there is no dedicated start token; generation begins directly from the provided context.

**Embedding Lookup:**

Identical in principle to BERT—token indices map to rows in a learned embedding matrix of dimensions `vocab_size × hidden_dim` (50,257 × 768 for GPT-2 small).

**Output Specification:**
- Shape: `(batch_size, sequence_length, hidden_dim)`
- Data type: Float32 tensor

---

### 1.2 Token Embeddings → Segment Embeddings *(BERT only)*

**Purpose and Motivation:**

Many NLP tasks require reasoning over multiple text segments simultaneously—question answering requires understanding both question and passage, natural language inference requires comparing premise and hypothesis. Segment Embeddings provide the model with explicit information about which sentence each token belongs to.

**Implementation Details:**

The Segment Embedding mechanism assigns a binary indicator to each token:
- Token Type ID `0`: Assigned to all tokens in Sentence A (including the initial `[CLS]` and the first `[SEP]`)
- Token Type ID `1`: Assigned to all tokens in Sentence B (including the final `[SEP]`)

These identifiers index into a learned Segment Embedding Matrix of dimensions `2 × hidden_dim`. The resulting embeddings are added element-wise to the Token Embeddings.

**Example:**
```
Input: "[CLS] What is AI ? [SEP] AI is artificial intelligence . [SEP]"
Type:    0     0   0  0 0   0   1  1       1           1   1
```

**Mathematical Formulation:**
```
Segment_Embedding[i] = E_segment[token_type_id[i]]
```

where `E_segment ∈ ℝ^{2 × d_model}` is the learned segment embedding matrix.

**Note on GPT-2:**

GPT-2 does not implement Segment Embeddings because its pre-training objective (causal language modeling on continuous text) does not require explicit sentence boundary information. When processing multiple segments, they are simply concatenated, potentially with delimiter tokens.

---

### 1.3 Positional Embeddings

**Fundamental Problem:**

The self-attention mechanism is inherently permutation-equivariant—reordering the input tokens produces correspondingly reordered outputs without any change in the attention computation itself. Unlike recurrent neural networks, which process tokens sequentially and thus implicitly encode position, Transformers require explicit positional information.

**Solution: Learned Positional Embeddings**

Both BERT and GPT-2 employ learned positional embeddings, where each position index (0, 1, 2, ..., max_length-1) is associated with a trainable vector.

**Implementation:**

A Position Embedding Matrix of dimensions `max_position_embeddings × hidden_dim` is initialized (typically `512 × 768` for base models). During forward propagation, position indices are used to retrieve the corresponding embedding vectors.

**Mathematical Formulation:**
```
Position_Embedding[i] = E_position[i]
```

where `E_position ∈ ℝ^{max_seq_len × d_model}` is the learned position embedding matrix and `i` is the absolute position index.

**Embedding Aggregation:**

The final input representation is computed by element-wise addition of all embedding components:

For BERT:
```
Input[i] = Token_Embedding[i] + Segment_Embedding[i] + Position_Embedding[i]
```

For GPT-2:
```
Input[i] = Token_Embedding[i] + Position_Embedding[i]
```

**Design Considerations:**

The choice of learned (vs. sinusoidal) positional embeddings allows the model to discover optimal position representations for its specific tasks but limits generalization to sequence lengths not seen during training. The maximum sequence length is therefore a fixed architectural hyperparameter.

---

### 1.4 Embedding Sum → Layer Normalization

**Purpose and Motivation:**

Layer Normalization stabilizes the training process by normalizing activations, reducing internal covariate shift, and improving gradient flow. Unlike Batch Normalization, Layer Normalization computes statistics across the feature dimension for each individual sample, making it suitable for variable-length sequences and small batch sizes.

**Mathematical Formulation:**

For an input vector `x ∈ ℝ^d`:

```
μ = (1/d) Σᵢ xᵢ                           (mean)
σ² = (1/d) Σᵢ (xᵢ - μ)²                   (variance)
LayerNorm(x) = γ ⊙ (x - μ) / √(σ² + ε) + β
```

where:
- `μ` is the mean computed across the hidden dimension
- `σ²` is the variance computed across the hidden dimension
- `γ ∈ ℝ^d` is a learned scale parameter (initialized to 1)
- `β ∈ ℝ^d` is a learned shift parameter (initialized to 0)
- `ε` is a small constant for numerical stability (typically 1e-12)
- `⊙` denotes element-wise multiplication

**Effect:**

After Layer Normalization, each token's representation has approximately zero mean and unit variance across its hidden dimensions. The learned parameters `γ` and `β` allow the model to recover any desired mean and variance if beneficial.

**Placement:**

BERT uses "post-norm" architecture where Layer Normalization follows residual additions. The normalized embeddings are then passed to the first Transformer encoder layer.

---

## Stage 2: Self-Attention Mechanism

### 2.1 Input → Q/K/V Projections

**Conceptual Foundation:**

The self-attention mechanism enables each token to gather information from all other tokens in the sequence based on learned compatibility functions. This is accomplished through three distinct projections that create specialized representations for different roles in the attention computation.

**The Three Projections:**

1. **Query (Q):** Represents "what information this token is looking for"
   - Conceptually, the Query encodes the token's information needs
   - Used to compute compatibility scores with Keys

2. **Key (K):** Represents "what information this token can provide"
   - Conceptually, the Key advertises the token's content
   - Used to compute compatibility scores with Queries

3. **Value (V):** Contains "the actual information to be retrieved"
   - The content that gets aggregated based on attention weights
   - Typically contains the same information as the input but in a transformed space

**Mathematical Formulation:**

```
Q = X · W_Q + b_Q
K = X · W_K + b_K  
V = X · W_V + b_V
```

where:
- `X ∈ ℝ^{seq_len × d_model}` is the input sequence
- `W_Q, W_K, W_V ∈ ℝ^{d_model × d_model}` are learned projection matrices
- `b_Q, b_K, b_V ∈ ℝ^{d_model}` are learned bias vectors

**Intuition:**

The separation into Q, K, V allows the model to learn different transformations for different purposes. A token's representation for "searching" (Q) may differ from its representation for "being found" (K) or "providing content" (V). This flexibility enables more expressive attention patterns than using the same representation for all purposes.

---

### 2.2 Q/K/V → Scaled Dot-Product Attention

**Overview:**

Scaled Dot-Product Attention computes a weighted average of Value vectors, where the weights are determined by the compatibility between Query and Key vectors.

**Step-by-Step Computation:**

**Step 1: Compute Attention Scores**

The compatibility between each Query-Key pair is computed via dot product:

```
Scores = Q · K^T
```

where `Scores ∈ ℝ^{seq_len × seq_len}` contains the raw attention scores. Element `Scores[i,j]` represents how much token `i` should attend to token `j`.

**Step 2: Apply Scaling**

The scores are divided by the square root of the key dimension:

```
Scaled_Scores = Scores / √d_k
```

**Rationale for Scaling:** For large values of `d_k`, the dot products can grow large in magnitude, pushing the softmax function into regions with extremely small gradients. The scaling factor maintains reasonable gradient magnitudes regardless of dimension size.

Mathematically, if Query and Key vectors have elements drawn from a distribution with zero mean and unit variance, their dot product has variance `d_k`. Scaling by `√d_k` normalizes this back to unit variance.

**Step 3: Apply Attention Mask (if applicable)**

For causal (autoregressive) models like GPT-2, future positions must be masked to prevent information leakage:

```
Masked_Scores[i,j] = Scaled_Scores[i,j]  if j ≤ i
                   = -∞                   if j > i
```

The `-∞` values become zero after softmax, effectively removing future tokens from consideration.

**Step 4: Compute Attention Weights via Softmax**

```
Attention_Weights = Softmax(Masked_Scores, dim=-1)
```

The softmax is applied row-wise, ensuring that attention weights for each query sum to 1:

```
Attention_Weights[i,j] = exp(Scores[i,j]) / Σₖ exp(Scores[i,k])
```

**Step 5: Compute Context Vectors**

The final output is a weighted sum of Value vectors:

```
Context = Attention_Weights · V
```

For each token position `i`:
```
Context[i] = Σⱼ Attention_Weights[i,j] · V[j]
```

**Complete Formula:**

```
Attention(Q, K, V) = Softmax(Q · K^T / √d_k) · V
```

---

### 2.3 Multi-Head Attention

**Motivation:**

A single attention function may be insufficient to capture the diverse relationships present in natural language. Multi-Head Attention addresses this by running multiple attention functions in parallel, each potentially learning different relationship types.

**Architecture:**

Instead of performing a single attention function with `d_model`-dimensional keys, values, and queries, Multi-Head Attention:

1. Projects Q, K, V into `h` different subspaces of dimension `d_k = d_model / h`
2. Performs attention independently in each subspace (each "head")
3. Concatenates the results
4. Applies a final linear projection

**Mathematical Formulation:**

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O

where head_i = Attention(Q · W_Q^i, K · W_K^i, V · W_V^i)
```

where:
- `W_Q^i, W_K^i, W_V^i ∈ ℝ^{d_model × d_k}` are per-head projection matrices
- `W_O ∈ ℝ^{h·d_v × d_model}` is the output projection matrix
- `h` is the number of heads (12 for base models, 16 for large models)
- `d_k = d_v = d_model / h` (64 for base models with d_model=768, h=12)

**Head Specialization:**

Empirical studies have shown that different heads often specialize in different linguistic phenomena:

| Specialization Type | Description | Example Pattern |
|---------------------|-------------|-----------------|
| Positional | Attention to adjacent tokens | Token attends primarily to previous/next token |
| Syntactic | Attention following dependency structure | Verb attends to its subject/object |
| Delimiter | Attention to special tokens | Tokens attend to [CLS] or [SEP] |
| Long-range | Attention to distant but related tokens | Pronoun attends to its antecedent |
| Rare/lexical | Attention based on rare or important words | Content words attend to each other |

---

### 2.4 Attention Output → Add & Norm

**Residual Connection:**

The attention output is combined with the original input via a residual (skip) connection:

```
Residual_Output = X + MultiHead(X, X, X)
```

**Benefits of Residual Connections:**

1. **Gradient Flow:** Provides a direct path for gradients to flow backward through the network, mitigating vanishing gradients in deep architectures
2. **Identity Mapping:** Allows layers to learn modifications to the identity function rather than complete transformations, simplifying optimization
3. **Information Preservation:** Ensures that information from earlier layers is preserved and accessible to later processing stages

**Layer Normalization:**

Following the residual addition, Layer Normalization is applied:

```
Output = LayerNorm(X + MultiHead(X, X, X))
```

**Complete Sub-layer Formula:**

```
Output = LayerNorm(X + Sublayer(X))
```

This pattern (residual connection followed by layer normalization) is applied consistently throughout the Transformer architecture.

---

## Stage 3: Attention Analysis and Visualization

*These components do not alter the computational graph but provide interpretability insights into the model's attention behavior.*

### 3.1 Global Attention Metrics

**Purpose:**

Quantitative measures computed across attention distributions to characterize model behavior at various levels of granularity (per head, per layer, or globally).

**Metric Definitions:**

**Entropy:**

Measures the dispersion of the attention distribution using Shannon entropy:

```
H(A_i) = -Σⱼ A[i,j] · log(A[i,j])
```

where `A[i,j]` is the attention weight from token `i` to token `j`.

- **High entropy:** Attention is distributed broadly across many tokens (diffuse attention)
- **Low entropy:** Attention is concentrated on few tokens (focused attention)
- **Maximum value:** `log(seq_len)` when attention is uniform
- **Minimum value:** `0` when attention is entirely on one token

**Confidence:**

The maximum attention weight in a distribution:

```
Confidence(A_i) = max_j(A[i,j])
```

- **High confidence:** The model has high certainty about where to attend
- **Low confidence:** Attention is spread without a clear focus

**Sparsity:**

Proportion of tokens receiving negligible attention (below a threshold τ):

```
Sparsity(A_i) = (1/n) · |{j : A[i,j] < τ}|
```

- **High sparsity:** Few tokens are relevant; attention is sparse
- **Low sparsity:** Many tokens receive substantial attention

---

### 3.2 Attention Flow Visualization

**Description:**

A Sankey diagram representation of information flow through the attention mechanism, where:

- **Nodes:** Represent tokens in the sequence
- **Edges:** Represent attention weights between token pairs
- **Edge thickness:** Proportional to attention weight magnitude

**Implementation Considerations:**

- Low-weight connections (below a threshold) are typically filtered to reduce visual clutter
- Visualization may focus on specific layers or heads of interest
- Bidirectional attention (BERT) shows edges in both directions; causal attention (GPT-2) shows only leftward edges

---

### 3.3 Attention Head Specialization Analysis

**Methodology:**

Attention patterns are analyzed to identify linguistic features captured by individual heads by correlating attention weights with linguistically-annotated structures.

**Analysis Categories:**

1. **Syntactic Attention:** Correlation between attention patterns and dependency parse trees
   - Measured by: Overlap with gold dependency edges

2. **Positional Attention:** Tendency to attend to specific relative positions
   - Measured by: Average attention weight at each relative position offset

3. **Long-range Attention:** Attention to tokens beyond a locality window
   - Measured by: Mean attention distance or attention beyond k tokens

4. **Semantic Attention:** Attention to semantically related tokens
   - Measured by: Correlation with semantic similarity measures

**Visualization:**

Results are commonly presented as radar charts showing each head's profile across multiple specialization dimensions.

---

### 3.4 Attention Dependency Tree

**Description:**

A hierarchical tree visualization rooted at a selected token, showing the propagation of attention through the sequence.

**Construction Algorithm:**

1. Select a root token of interest
2. Identify the top-k tokens that the root attends to most strongly (children)
3. Recursively identify what each child attends to (grandchildren)
4. Continue to desired depth

**Interpretation:**

- **Direct children:** Tokens that directly influence the root's representation
- **Deeper levels:** Transitive information flow (how information propagates across multiple attention operations)

---

### 3.5 Inter-Sentence Attention (ISA)

**Definition:**

A metric quantifying the attention flow between two distinct text segments (Sentence A and Sentence B).

**Computation:**

```
ISA = (1/|A|·|B|) · Σᵢ∈A Σⱼ∈B (Attention[i,j] + Attention[j,i])
```

where `A` and `B` denote the sets of token indices belonging to each sentence.

**Interpretation:**

- **High ISA:** Strong information exchange between sentences; the model is actively comparing or relating the two segments
- **Low ISA:** Sentences are processed relatively independently; limited cross-sentence reasoning

**Applications:**

ISA is particularly relevant for:
- Question Answering (question-passage interaction)
- Natural Language Inference (premise-hypothesis comparison)
- Semantic Similarity (sentence pair comparison)

---

## Stage 4: Feed-Forward Processing

### 4.1 Add & Norm → Feed-Forward Network

**Purpose and Motivation:**

The Feed-Forward Network (FFN) applies non-linear transformations to each position independently, providing the model with additional expressive capacity beyond what attention alone offers. While attention handles inter-token interactions, the FFN processes each token's representation individually.

**Architecture:**

The FFN consists of two linear transformations with a non-linear activation function between them:

```
FFN(x) = Linear_2(Activation(Linear_1(x)))
       = (Activation(x · W_1 + b_1)) · W_2 + b_2
```

**Dimension Expansion and Contraction:**

The FFN employs a bottleneck architecture in reverse—expanding to a higher dimension before contracting:

- **Input dimension:** `d_model` (768 for base models)
- **Intermediate dimension:** `d_ff = 4 × d_model` (3072 for base models)
- **Output dimension:** `d_model` (768 for base models)

**Weight Dimensions:**
- `W_1 ∈ ℝ^{d_model × d_ff}` (768 × 3072)
- `b_1 ∈ ℝ^{d_ff}` (3072)
- `W_2 ∈ ℝ^{d_ff × d_model}` (3072 × 768)
- `b_2 ∈ ℝ^{d_model}` (768)

**Activation Function:**

**BERT** uses GELU (Gaussian Error Linear Unit):
```
GELU(x) = x · Φ(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])
```

where `Φ(x)` is the cumulative distribution function of the standard normal distribution.

**GPT-2** also uses GELU, though the original Transformer used ReLU.

**Intuitive Interpretation:**

The FFN can be viewed as a two-layer neural network applied independently to each token:

1. **Expansion layer (W_1):** Projects the `d_model`-dimensional representation into a higher `d_ff`-dimensional space, potentially disentangling complex feature interactions
2. **Activation:** Applies non-linearity, enabling the learning of complex functions
3. **Projection layer (W_2):** Compresses the representation back to `d_model` dimensions, selecting and combining the most relevant features

Recent interpretability research suggests that FFN neurons often correspond to interpretable concepts or patterns, functioning as a form of key-value memory.

---

### 4.2 FFN → Add & Norm (Post-FFN)

**Residual Connection and Layer Normalization:**

Following the FFN, a second residual connection and layer normalization are applied:

```
Output = LayerNorm(X + FFN(X))
```

**Complete Transformer Block:**

One complete Transformer encoder block consists of:

```
X_1 = LayerNorm(X_0 + MultiHeadAttention(X_0))    [Attention sub-layer]
X_2 = LayerNorm(X_1 + FFN(X_1))                   [FFN sub-layer]
```

**Layer Stacking:**

This complete block is repeated `N` times (N=12 for base models, N=24 for large models). Each layer operates on the output of the previous layer:

```
X^(l) = TransformerBlock(X^(l-1))    for l = 1, 2, ..., N
```

The final output `X^(N)` constitutes the contextualized representations for all tokens.

---

## Stage 5: Output Generation and Predictions

### 5.1 Final Layer → Hidden States

**Description:**

The output of the final Transformer layer constitutes the Hidden States—dense, contextualized representations that encode both the token's original meaning and information gathered from the entire sequence through attention.

**Properties:**

- **Shape:** `(batch_size, sequence_length, hidden_dim)`
- **Content:** Each vector `H[i]` represents token `i` in context, incorporating information from all (BERT) or preceding (GPT-2) tokens
- **Information aggregation:** Earlier layers tend to capture more local, syntactic information; later layers capture more global, semantic information

**Usage:**

Hidden states serve as input to task-specific prediction heads:
- **Token-level tasks:** Use hidden states directly (e.g., Named Entity Recognition, token classification)
- **Sequence-level tasks:** Use the `[CLS]` token's hidden state (BERT) or the final token's hidden state (GPT-2)
- **Generation tasks:** Use hidden states to predict next tokens (GPT-2)

---

### 5.2 Hidden States → Token Predictions

#### 5.2.1 BERT: Masked Language Modeling (MLM)

**Pre-training Objective:**

During pre-training, approximately 15% of input tokens are selected for prediction:
- 80% are replaced with `[MASK]`
- 10% are replaced with a random token
- 10% remain unchanged

The model must predict the original token identity for all selected positions.

**Prediction Head Architecture:**

```
1. Hidden State → Linear transformation (hidden_dim → hidden_dim)
2. GELU activation
3. Layer Normalization
4. Linear projection (hidden_dim → vocab_size)
5. Softmax to obtain probability distribution
```

**Mathematical Formulation:**

```
H_transformed = LayerNorm(GELU(H · W_transform + b_transform))
Logits = H_transformed · W_vocab + b_vocab
P(token | context) = Softmax(Logits)
```

where `W_vocab ∈ ℝ^{hidden_dim × vocab_size}` is often tied (shared) with the input token embedding matrix.

**Loss Function:**

Cross-entropy loss over the masked positions:

```
L_MLM = -Σᵢ∈masked log P(token_i | context)
```

#### 5.2.2 GPT-2: Causal Language Modeling

**Pre-training Objective:**

GPT-2 is trained to predict the next token given all preceding tokens, processing the entire sequence in parallel during training.

**Prediction Mechanism:**

```
P(token_{t+1} | token_1, ..., token_t) = Softmax(H_t · W_vocab)
```

where `H_t` is the hidden state at position `t`.

**Autoregressive Generation:**

During generation, tokens are produced sequentially:

1. Process the prompt to obtain hidden states
2. Sample or select the next token from `P(token_{t+1} | context)`
3. Append the selected token to the sequence
4. Repeat from step 1 with the extended sequence

**Loss Function:**

Cross-entropy loss summed over all positions:

```
L_CLM = -Σₜ log P(token_t | token_1, ..., token_{t-1})
```

---

### 5.3 Task-Specific Fine-tuning Heads

**BERT Classification:**

For sequence classification tasks, a linear layer is applied to the `[CLS]` token's final hidden state:

```
P(class | sequence) = Softmax(H_[CLS] · W_classifier + b_classifier)
```

**BERT Token Classification:**

For token-level tasks (NER, POS tagging), a linear layer is applied to each token's hidden state:

```
P(label_i | sequence) = Softmax(H_i · W_token + b_token)
```

**GPT-2 Classification:**

The final token's hidden state is typically used, or a special token is appended:

```
P(class | sequence) = Softmax(H_final · W_classifier + b_classifier)
```

---

## Architectural Comparison: BERT vs GPT-2

| Aspect | BERT | GPT-2 |
|--------|------|-------|
| **Architecture Type** | Encoder-only | Decoder-only |
| **Attention Pattern** | Bidirectional (full attention) | Causal (masked future) |
| **Tokenizer** | WordPiece (30,522 vocab) | BPE (50,257 vocab) |
| **Special Tokens** | `[CLS]`, `[SEP]`, `[MASK]`, `[PAD]` | `<\|endoftext\|>` |
| **Segment Embeddings** | Yes (2 segments) | No |
| **Pre-training Task** | MLM + Next Sentence Prediction | Causal Language Modeling |
| **Layers (base)** | 12 | 12 |
| **Hidden Dimension (base)** | 768 | 768 |
| **Attention Heads (base)** | 12 | 12 |
| **FFN Dimension (base)** | 3072 | 3072 |
| **Max Sequence Length** | 512 | 1024 |
| **Parameters (base)** | ~110M | ~117M |
| **Primary Use Case** | Understanding tasks | Generation tasks |

---

## Mathematical Reference

### Notation Summary

| Symbol | Description | Typical Value (Base) |
|--------|-------------|---------------------|
| `d_model` | Model hidden dimension | 768 |
| `d_ff` | FFN intermediate dimension | 3072 |
| `d_k, d_v` | Key/Value dimension per head | 64 |
| `h` | Number of attention heads | 12 |
| `N` | Number of Transformer layers | 12 |
| `V` | Vocabulary size | 30,522 (BERT) / 50,257 (GPT-2) |
| `L_max` | Maximum sequence length | 512 (BERT) / 1024 (GPT-2) |

### Key Equations

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = Softmax(QK^T / √d_k) · V
```

**Multi-Head Attention:**
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
head_i = Attention(QW_Q^i, KW_K^i, VW_V^i)
```

**Feed-Forward Network:**
```
FFN(x) = GELU(xW_1 + b_1)W_2 + b_2
```

**Layer Normalization:**
```
LayerNorm(x) = γ ⊙ (x - μ) / √(σ² + ε) + β
```

**Transformer Block:**
```
X' = LayerNorm(X + MultiHeadAttention(X, X, X))
X'' = LayerNorm(X' + FFN(X'))
```

---

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
2. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL*.
3. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners." *OpenAI Technical Report*.
4. Clark, K., et al. (2019). "What Does BERT Look At? An Analysis of BERT's Attention." *BlackboxNLP Workshop*.
5. Ba, J.L., et al. (2016). "Layer Normalization." *arXiv preprint*.