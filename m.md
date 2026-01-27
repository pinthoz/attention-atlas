nput ➜ Token Embeddings
(Dinâmico) GPT-2: What Happens: Tokenization & Embedding Lookup The input text is first processed by the Byte-Pair Encoding (BPE) tokenizer, which breaks words into subword units (tokens) based on frequency patterns. Each token is mapped to a unique integer ID. These IDs are then used to look up dense vectors from the Token Embedding Matrix (size: vocab_size × hidden_dim). GPT-2 uses no special start token — generation begins directly from the input. The end-of-text token <|endoftext|> marks sequence boundaries.

Output Shape: (batch_size, seq_len, hidden_dim)
BERT: What Happens: Tokenization & Embedding Lookup The input text is first processed by the WordPiece tokenizer, which breaks words into subword units (tokens). Each token is mapped to a unique integer ID. These IDs are then used to look up dense vectors from the Token Embedding Matrix (size: vocab_size × hidden_dim). Special tokens [CLS] (start) and [SEP] (separator) are added.

Output Shape: (batch_size, seq_len, hidden_dim)
Token Embeddings ➜ Segment Embeddings
What Happens: Adding Sentence Context Segment Embeddings distinguish between different sentences in the input (e.g., Sentence A vs. Sentence B). This is crucial for tasks like Question Answering or Next Sentence Prediction.

Token Type IDs: 0 for the first sentence, 1 for the second.
Embeddings are looked up from a learned matrix of size 2 × hidden_dim.
These are added element-wise to the Token Embeddings.
Segment Embeddings ➜ Positional Embeddings
What Happens: Injecting Position Information Since the Transformer architecture has no inherent sense of order (unlike RNNs), Positional Embeddings are added to give the model information about the absolute position of each token.

Learned embeddings for each position index (0, 1, 2, ...).
Matrix size: max_position_embeddings × hidden_dim (typically 512 × 768 for BERT-base).
Added element-wise to the previous sum of Token and Segment embeddings.
Token Embeddings ➜ Positional Embeddings
(Dinâmico - GPT-2 vs BERT) GPT-2: What Happens: Adding Position Information In GPT-2, Positional Embeddings are added directly to the Token Embeddings. Unlike BERT, there are no Segment Embeddings because GPT-2 is typically trained on a continuous stream of text.

Token Embeddings: Represent the meaning of each word/subword.
Positional Embeddings: Indicate the order of tokens in the sequence.
Summation: The two vectors are added element-wise to create the initial input representation.
BERT: What Happens: Adding Position & Segment Info For BERT, the final input representation is the sum of three components. Even if the Segment Embeddings details are hidden in this comparison view, they are included in the calculation.

Token Embeddings: Represent the meaning of each word/subword.
Segment Embeddings: Distinguish between Sentence A and Sentence B.
Positional Embeddings: Indicate the order of tokens. Input = Token + Segment + Position All three vectors are summed element-wise to form the model's input.
Positional Embeddings ➜ Sum & Layer Normalization
What Happens: Embedding Summation & Normalization The three embedding components are element-wise summed: Input = Token_Embeddings + Segment_Embeddings + Positional_Embeddings This combined representation is then passed through Layer Normalization to stabilize training.

Summation: Combines semantic meaning, sentence segment info, and position info.
LayerNorm: Normalizes the vector for each token to have mean 0 and variance 1.
Segment Embeddings ➜ Sum & Layer Normalization
What Happens: Input Aggregation The Segment Embeddings are combined with the other embedding components (Token and Positional) to form the sequence representation. Input = Token + Segment + Position All three vectors are summed element-wise, meaning they must occupy the same vector space. The result is then normalized (LayerNorm) to prepare it for the Transformer layers.

Sum & Layer Normalization ➜ Q/K/V Projections
What Happens: Linear Projections The input vectors are projected into three different spaces using learned linear transformations (dense layers) to create Query (Q), Key (K), and Value (V) vectors.

Q: What the token is looking for.
K: What the token "advertises" about itself.
V: The actual content information to be aggregated. Each projection uses a weight matrix of size hidden_dim × hidden_dim.
Q/K/V Projections ➜ Add & Norm
What Happens: Q/K/V ➜ Scaled Attention ➜ Add & Norm Step 1: Q/K/V ➜ Scaled Dot-Product Attention The projected Query, Key, and Value vectors are fed into the attention mechanism. The model calculates the compatibility (scores) between Q and K to decide "where to look". Scores = (Q · K^T) / √d_k

Step 2: Scaled Dot-Product Attention ➜ Add & Norm The scores are converted to probabilities and used to aggregate context.

Softmax: Converts scores to attention weights (probabilities).
Weighted Sum: Multiplies Values (V) by these weights to get the Context Vector.
Residual: The Context Vector is added back to the original input (x + Sublayer(x)).
Norm: The result is normalized (LayerNorm). Output = LayerNorm(x + Attention(Q,K,V))
Scaled Dot-Product Attention ➜ Global Attention Metrics
What Happens: Aggregating Statistics We compute global metrics across all attention heads and layers to understand the model's behavior. This step doesn't change the data flow but analyzes the attention patterns produced in the previous step.

Entropy: How focused or diffuse the attention is.
Confidence: The magnitude of the maximum attention weight.
Sparsity: How many tokens receive significant attention.
Global Attention Metrics ➜ Multi-Head Attention
What Happens: Visualizing Attention Heads This visualization allows us to inspect the raw attention matrices for individual heads. In Multi-Head Attention, the model runs the attention mechanism multiple times in parallel (12 heads for BERT-base). Each head can learn to focus on different relationships (e.g., one head might track next-token relationships, another might track subject-verb dependencies).

Multi-Head Attention ➜ Attention Flow
What Happens: Flow Visualization The Attention Flow view provides a Sankey-style diagram to visualize the flow of information between tokens.

Lines: Represent attention weights.
Thickness: Proportional to the attention strength.
Filtering: Low-weight connections are often hidden to reduce clutter and reveal the most significant dependencies.
Attention Flow ➜ Attention Head Specialization
What Happens: Analyzing Head Roles We analyze the attention patterns to determine what linguistic features each head specializes in. This is done by correlating attention weights with known linguistic properties.

Syntax: Attention to syntactic dependencies.
Positional: Attention to previous/next tokens.
Long-range: Attention to distant tokens. The Radar Chart visualizes this "profile" for each head.
Attention Head Specialization ➜ Attention Dependency Tree
What Happens: Hierarchical Influence The Dependency Tree visualizes the chain of influence starting from a selected root token. It shows how attention propagates through the sequence.

Root: The token being analyzed.
Children: Tokens that the root attends to most strongly.
Depth: Shows multi-hop attention (tokens attending to tokens that attend to the root).
Attention Dependency Tree ➜ Inter-Sentence Attention
What Happens: Cross-Sentence Analysis Inter-Sentence Attention (ISA) specifically isolates and quantifies the attention flowing between the two input sentences (Sentence A and Sentence B).

ISA Score: Aggregates attention weights where the Source is in one sentence and the Target is in the other.
High ISA indicates strong interaction or information exchange between the sentences.
Inter-Sentence Attention ➜ Add & Norm
What Happens: Residual Connection & Norm After the attention mechanism, a Residual (Skip) Connection adds the original input back to the attention output, followed by another Layer Normalization. Output = LayerNorm(x + Attention(x)) This allows gradients to flow through the network more easily and preserves information from the lower layers.

Add & Norm ➜ Feed-Forward Network
What Happens: Feed-Forward Processing The output is passed through a position-wise Feed-Forward Network (FFN). This is where the model "thinks" about the information it has gathered. FFN(x) = GELU(xW_1 + b_1)W_2 + b_2

Column 1 ("Intermediate"): The model expands information into a massive space (3072 dimensions in BERT) to disentangle complex concepts. The heatmap shows these neurons firing.
Column 2 ("Projection"): It compresses this back to the standard size (768 dimensions) to pass it to the next layer.
Feed-Forward Network ➜ Add & Norm (post-FFN)
What Happens: Second Residual & Norm A second Residual Connection and Layer Normalization are applied after the FFN. Output = LayerNorm(x + FFN(x)) This completes one full Transformer Encoder Block. In BERT-base, this entire process (Attention → Add&Norm → FFN → Add&Norm) repeats 12 times.

Add & Norm (post-FFN) ➜ Exit
(Dinâmico) GPT-2: What Happens: Final Prediction The Add & Norm layer outputs the final Hidden States. These vectors are then projected to the vocabulary size to predict the Next Token in the sequence (Causal Language Modeling). P(token_{t+1}) = Softmax(Hidden · W_vocab)

BERT: What Happens: Final Output & Prediction The Add & Norm layer outputs the final Hidden States. In the pre-training phase, these are used for Masked Token Predictions (MLM), reconstructing masked words from context. P(mask) = Softmax(Hidden · W_vocab) For fine-tuning tasks (e.g., classification), the [CLS] token's hidden state is typically used.

Add & Norm (post-FFN) ➜ Hidden States
What Happens: Final Layer Output The output of the final (12th) encoder layer constitutes the Hidden States (or contextualized embeddings) for the sequence.

Shape: (batch_size, seq_len, hidden_dim)
These vectors contain rich, contextual information aggregated from all previous layers and are used for downstream tasks or the pre-training objectives.
Hidden States ➜ Token Output Predictions
What Happens: Unembedding & Prediction The final hidden states are projected back to the vocabulary size to predict the next token (or masked token). Logits = LayerNorm(Hidden) · W_vocab + b Softmax is then applied to convert these logits into probabilities: P(token_i) = exp(logit_i) / Σ exp(logit_j)

Hidden States ➜ Token Output Predictions (MLM)
What Happens: Masked Language Modeling For the pre-training objective, an MLM Head projects the hidden states back to the vocabulary size to predict the original identity of masked tokens. Logits = Linear(HiddenStates) Probabilities = Softmax(Logits) This gives a probability distribution over the entire vocabulary for each token position.

Hidden States ➜ Next Token Predictions
What Happens: Causal Language Modeling The final hidden states are projected to the vocabulary size to predict the next token in the sequence. P(token_{t+1} | token_1...token_t) This is the core objective of GPT-2: predicting the future based on the past context.