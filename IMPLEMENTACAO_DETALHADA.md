# Implementação Detalhada - Attention Atlas

## Índice
1. [Visão Geral](#visão-geral)
2. [Arquitetura Geral](#arquitetura-geral)
3. [BERT - Implementação das Sections](#bert---implementação-das-sections)
4. [GPT-2 - Implementação das Sections](#gpt-2---implementação-das-sections)
5. [Funcionalidades Partilhadas](#funcionalidades-partilhadas)
6. [Diferenças Técnicas entre BERT e GPT-2](#diferenças-técnicas-entre-bert-e-gpt-2)

---

## Visão Geral

O **Attention Atlas** é uma aplicação web interativa desenvolvida em **Python** usando **Shiny for Python** que permite visualizar e explorar internamente o funcionamento de modelos Transformer. A aplicação suporta dois tipos de modelos:
- **BERT** (Bidirectional Encoder Representations from Transformers) - modelos encoder bidirecionais
- **GPT-2** (Generative Pre-trained Transformer 2) - modelos decoder autogressivos com máscara causal

Todos os modelos são carregados usando a biblioteca **Hugging Face Transformers**.

---

## Arquitetura Geral

### Gestão de Modelos (models.py)
```python
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
```

**Classe ModelManager:**
- Carrega modelos da Hugging Face
- Implementa cache para evitar recarregar modelos
- Suporta automaticamente GPU (CUDA) quando disponível
- Para cada modelo, retorna: `(tokenizer, encoder_model, mlm_model)`

**Modelos BERT disponíveis:**
- `bert-base-uncased`
- `bert-large-uncased`
- `bert-base-multilingual-uncased`

**Modelos GPT-2 disponíveis:**
- `gpt2` (Small)
- `gpt2-medium`

**Configuração dos modelos:**
```python
# BERT
encoder = BertModel.from_pretrained(
    model_name,
    output_attentions=True,      # Para extrair matrizes de atenção
    output_hidden_states=True,   # Para extrair estados ocultos
)

# GPT-2
encoder = GPT2Model.from_pretrained(
    model_name,
    output_attentions=True,
    output_hidden_states=True,
)
```

---

## BERT - Implementação das Sections

### 1. Token Embeddings
**Ficheiro:** `server.py` - função `get_embedding_table()`

**O que faz:**
- Extrai embeddings dos tokens do BERT usando Hugging Face
- Cada token é convertido num vetor denso de dimensão 768 (base) ou 1024 (large)

**Implementação:**
```python
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
embeddings = outputs.last_hidden_state[0].cpu().numpy()
```

**Visualização:**
- Primeiras 64 dimensões do embedding mostradas como heatmap
- Usa colormap "Blues"
- Tooltip mostra primeiras 32 dimensões com valores numéricos

**Biblioteca Hugging Face:** Sim, usa `BertModel.embeddings.word_embeddings`

---

### 2. Segment Embeddings
**Ficheiro:** `server.py` - função `get_segment_embedding_view()`

**O que faz:**
- Mostra `token_type_ids` que distinguem Sentence A (0) e Sentence B (1)
- **Nota:** GPT-2 não tem segment embeddings (retorna mensagem explicativa)

**Implementação:**
```python
segment_ids = inputs.get("token_type_ids")  # Vem do tokenizer
ids = segment_ids[0].cpu().numpy().tolist()
```

**Tokenização com segmentos (BERT):**
```python
# Deteta automaticamente duas frases usando regex
pattern = re.search(r"([.!?])\s+([A-Za-z])", text)
if pattern:
    sentence_a = text[:split_idx].strip()
    sentence_b = text[split_idx:].strip()
    return tokenizer(sentence_a, sentence_b, return_tensors="pt")
```

**Biblioteca Hugging Face:** Sim, usa `tokenizer()` com dois argumentos para gerar `token_type_ids`

---

### 3. Positional Embeddings
**Ficheiro:** `server.py` - função `get_posenc_table()`

**O que faz:**
- Gera positional encodings usando fórmula sinusoidal do Transformer original
- BERT na verdade usa **learned positional embeddings**, mas para visualização usa-se a fórmula clássica

**Implementação (helpers.py):**
```python
def positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    position = np.arange(0, seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)  # Posições pares
    pe[:, 1::2] = np.cos(position * div_term)  # Posições ímpares
    return pe
```

**Visualização:**
- Primeiras 64 dimensões como heatmap com colormap "RdBu"

**Biblioteca Hugging Face:** Parcialmente - BERT usa `position_embeddings` aprendidos, mas visualização usa fórmula manual

---

### 4. Sum & Layer Normalization
**Ficheiro:** `server.py` - função `get_sum_layernorm_view()`

**O que faz:**
- Soma word embeddings + position embeddings + segment embeddings
- Aplica LayerNorm aos embeddings somados

**Implementação:**
```python
# Acesso direto aos embeddings do BERT via Hugging Face
word_embed = encoder_model.embeddings.word_embeddings(input_ids)
pos_embed = encoder_model.embeddings.position_embeddings(position_ids)
seg_embed = encoder_model.embeddings.token_type_embeddings(segment_ids)

summed = word_embed + pos_embed + seg_embed
normalized = encoder_model.embeddings.LayerNorm(summed)
```

**Biblioteca Hugging Face:** Sim, acesso direto aos módulos internos do `BertModel.embeddings`

---

### 5. Q/K/V Projections
**Ficheiro:** `server.py` - função `get_qkv_table()`

**O que faz:**
- Projeta hidden states em Query, Key, Value para uma layer específica
- Permite selecionar qual layer visualizar

**Implementação:**
```python
# Acesso à layer de atenção do BERT
layer = encoder_model.encoder.layer[layer_idx].attention.self
hs_in = hidden_states[layer_idx]  # Hidden states entrada da layer

# Projeções usando weights da Hugging Face
Q = layer.query(hs_in)[0].cpu().numpy()
K = layer.key(hs_in)[0].cpu().numpy()
V = layer.value(hs_in)[0].cpu().numpy()
```

**Visualização:**
- Cards para cada token com 3 heatmaps (Q, K, V)
- Q usa colormap "Greens", K usa "Oranges", V usa "Purples"
- Primeiras 48 dimensões de cada projeção

**Biblioteca Hugging Face:** Sim, acessa `BertModel.encoder.layer[i].attention.self.{query,key,value}`

---

### 6. Scaled Dot-Product Attention
**Ficheiro:** `server.py` - função `get_scaled_attention_view()`

**O que faz:**
- Calcula e visualiza: `softmax(Q·K^T / √d_k)`
- Mostra top 3 conexões de atenção para um token focal

**Implementação:**
```python
att = attentions[layer_idx][0, head_idx].cpu().numpy()  # Da Hugging Face
d_k = Q.shape[-1] // num_heads

# Para cada conexão top-3:
dot = float(np.dot(Q[focus_idx], K[j]))
scaled = dot / np.sqrt(d_k)
prob = att[focus_idx, j]  # Probabilidade do softmax
```

**Biblioteca Hugging Face:** Sim, usa `outputs.attentions` retornado pelo modelo

---

### 7. Multi-Head Attention Map
**Ficheiro:** `ui.py` - função `render_attention_map()`

**O que faz:**
- Heatmap interativo das atenções token-to-token
- Seleção de layer e head específicos

**Implementação:**
```python
att = attentions[layer_idx][0, head_idx].cpu().numpy()

fig = go.Figure(data=go.Heatmap(
    z=att,
    x=clean_tokens,
    y=clean_tokens,
    colorscale='RdPu',
    hoverongaps=False
))
```

**Biblioteca:** Plotly para visualização, attentions da Hugging Face

---

### 8. Attention Flow
**Ficheiro:** `ui.py` - função `render_attention_flow()`

**O que faz:**
- Diagrama Sankey mostrando fluxo de atenção através das layers
- Agregação de atenções across layers e heads

**Implementação:**
```python
att_matrices = [layer[0].cpu().numpy() for layer in attentions]
# Média sobre heads: (num_layers, seq_len, seq_len)
att_avg = np.mean(att_matrices, axis=1)

# Para cada layer → próxima layer, criar links no Sankey
```

**Biblioteca:** Plotly Sankey diagram, dados da Hugging Face

---

### 9. Feed-Forward Network
**Ficheiro:** `server.py` - função `get_ffn_view()`

**O que faz:**
- Visualiza expansão → ativação GELU → projeção
- FFN dimensões: 768 → 3072 → 768 (BERT-base)

**Implementação:**
```python
layer = encoder_model.encoder.layer[layer_idx]
hs_in = hidden_states[layer_idx][0]

# Hugging Face FFN modules
inter = layer.intermediate.dense(hs_in)  # Expansão para 3072
inter_act = layer.intermediate.intermediate_act_fn(inter)  # GELU
proj = layer.output.dense(inter_act)  # Volta para 768
```

**Biblioteca Hugging Face:** Sim, `BertModel.encoder.layer[i].{intermediate, output}`

---

### 10. Add & Norm (Residual Connections)
**Ficheiro:** `server.py` - função `get_add_norm_view()`

**O que faz:**
- Visualiza magnitude da mudança nos hidden states após Add&Norm
- Calcula norma euclidiana da diferença

**Implementação:**
```python
hs_in = hidden_states[layer_idx][0].cpu().numpy()
hs_out = hidden_states[layer_idx + 1][0].cpu().numpy()

# Para cada token:
diff = np.linalg.norm(hs_out[i] - hs_in[i])
ratio = diff / (np.linalg.norm(hs_in[i]) + 1e-6)
```

**Biblioteca Hugging Face:** Usa `outputs.hidden_states` (lista de tensores)

---

### 11. Global Attention Metrics
**Ficheiro:** `metrics.py` - função `compute_all_attention_metrics()`

**O que faz:**
- Calcula 6 métricas globais sobre matriz de atenção média:
  1. **Confidence Max:** Maior probabilidade de atenção
  2. **Confidence Avg:** Média das máximas probabilidades
  3. **Focus (Entropy):** Entropia da distribuição
  4. **Sparsity:** Percentagem de pesos < 0.01
  5. **Distribution (Median):** Mediana dos pesos
  6. **Uniformity:** Quão uniforme é a distribuição

**Implementação:**
```python
def compute_all_attention_metrics(att_matrix):
    confidence_max = float(att_matrix.max(axis=-1).mean())
    confidence_avg = float(att_matrix.mean())

    # Entropy
    eps = 1e-10
    entropy = -np.sum(att_matrix * np.log(att_matrix + eps), axis=-1)
    focus_entropy = float(entropy.mean())

    # Sparsity
    sparsity = float(np.mean(att_matrix < 0.01))

    # ...
```

**Biblioteca Hugging Face:** Usa attentions, mas cálculos manuais

---

### 12. Attention Head Specialization (Radar Chart)
**Ficheiro:** `head_specialization.py` - função `compute_all_heads_specialization()`

**O que faz:**
- Analisa 7 dimensões comportamentais de cada attention head:
  1. **Syntax:** Atenção a tokens sintáticos (DET, ADP, AUX, etc.)
  2. **Semantics:** Atenção a tokens semânticos (NOUN, VERB, ADJ, etc.)
  3. **CLS Focus:** Atenção ao token [CLS]
  4. **Punctuation:** Atenção a pontuação
  5. **Entities:** Atenção a entidades nomeadas (NER)
  6. **Long-range:** Atenção a tokens distantes (≥5 posições)
  7. **Self-attention:** Atenção diagonal (token para si mesmo)

**Implementação:**
```python
# Usa spaCy para POS tagging e NER
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

# Para cada head:
att_matrix = layer_attention[0, head_idx].cpu().numpy()

# Exemplo: Syntax focus
syntax_pos = {"DET", "ADP", "AUX", "CCONJ", "SCONJ", "PART", "PRON"}
syntax_indices = [i for i, tag in enumerate(pos_tags) if tag in syntax_pos]
syntax_focus = att_matrix[:, syntax_indices].sum() / att_matrix.sum()
```

**Normalização:**
- Min-max normalization across todas as heads de uma layer
- Permite comparação relativa entre heads

**Bibliotecas:**
- **spaCy** para análise linguística
- **Hugging Face** para attention matrices

---

### 13. Attention Dependency Tree
**Ficheiro:** `helpers.py` - função `compute_influence_tree()`

**O que faz:**
- Constrói árvore hierárquica de dependências de atenção
- Começa num token raiz e expande top-k conexões recursivamente

**Implementação:**
```python
def compute_influence_tree(att_matrix, tokens, Q, K, d_k, root_idx, top_k=3, max_depth=3):
    visited = set()

    def build_subtree(idx, depth):
        if depth >= max_depth or idx in visited:
            return None

        visited.add(idx)

        # Top-k tokens mais atendidos
        attention_scores = att_matrix[idx]
        top_indices = np.argsort(attention_scores)[::-1][:top_k]

        children = []
        for child_idx in top_indices:
            if child_idx not in visited:
                child = build_subtree(child_idx, depth + 1)
                if child:
                    children.append(child)

        return {
            "name": tokens[idx],
            "value": float(attention_scores[idx]),
            "children": children
        }

    return build_subtree(root_idx, 0)
```

**Visualização:** D3.js tree layout no frontend

**Biblioteca Hugging Face:** Usa attentions, mas árvore construída manualmente

---

### 14. Inter-Sentence Attention (ISA)
**Ficheiro:** `isa.py` - função `compute_isa()`

**O que faz:**
- Segmenta texto em frases usando NLTK
- Calcula matriz de atenção sentence-to-sentence
- **ISA(Sa, Sb) = max_h max_{i∈Sa, j∈Sb} A[h, i, j]**

**Implementação:**
```python
import nltk
from nltk import sent_tokenize

# 1. Segmentação de frases
sentences = nltk.sent_tokenize(text)

# 2. Mapear tokens para frases (character-based matching)
def get_sentence_boundaries(text, tokens, tokenizer, inputs):
    # Constrói mapa char_index -> sentence_index
    # Depois mapeia cada token para frase usando posição de caracteres
    # ...

# 3. Agregar atenção token-level para sentence-level
# Max across layers
stacked_attentions = torch.stack([att[0] for att in attentions], dim=0)
A = torch.max(stacked_attentions, dim=0)[0]  # (num_heads, seq_len, seq_len)

# Para cada par de frases (Sa, Sb):
sub_att = A[:, start_row:end_row, start_col:end_col]
isa_matrix[r_idx, c_idx] = torch.max(sub_att).item()
```

**Visualização:**
- Scatter plot interativo (Plotly)
- Heatmap token-to-token para par de frases selecionado

**Bibliotecas:**
- **NLTK** para sentence tokenization
- **Hugging Face** para attentions

---

### 15. Hidden States / Layer Output
**Ficheiro:** `server.py` - função `get_layer_output_view()`

**O que faz:**
- Mostra hidden states finais de uma layer
- Estatísticas: média, desvio padrão, máximo

**Implementação:**
```python
hs = hidden_states[layer_idx + 1][0].cpu().numpy()

mean_val = float(hs[i].mean())
std_val = float(hs[i].std())
max_val = float(hs[i].max())
```

**Biblioteca Hugging Face:** Sim, `outputs.hidden_states`

---

### 16. MLM Predictions (Output Probabilities)
**Ficheiro:** `server.py` - função `get_output_probabilities()`

**O que faz:**
- Usa `BertForMaskedLM` head para prever tokens
- Top-5 predictions para cada posição
- Mostra cálculo detalhado do softmax

**Implementação:**
```python
mlm_outputs = mlm_model(**inputs)  # BertForMaskedLM
logits = mlm_outputs.logits[0]  # (seq_len, vocab_size)

probs = torch.softmax(logits, dim=-1)
top_vals, top_idx = torch.topk(token_probs, top_k=5)

# Para cada token, mostra:
# - Token previsto
# - Probabilidade (softmax)
# - Logit original
# - exp(logit)
# - Soma de exp(logits)
```

**Biblioteca Hugging Face:** Sim, `BertForMaskedLM.from_pretrained()`

---

## GPT-2 - Implementação das Sections

A arquitetura GPT-2 é muito similar ao BERT, mas com diferenças cruciais:

### Diferenças Principais:
1. **Sem Segment Embeddings:** GPT-2 não usa `token_type_ids`
2. **Máscara Causal:** Atenção lower-triangular (não pode atender tokens futuros)
3. **Tokenização diferente:** Byte-Pair Encoding com prefixo "Ġ" para espaços
4. **Predição autoregressiva:** Prediz próximo token, não masked tokens

---

### 1. Token Embeddings (GPT-2)
**Implementação idêntica ao BERT**, mas:
```python
word_embed = encoder_model.wte(input_ids)  # GPT2Model.wte (word token embeddings)
```

**Biblioteca Hugging Face:** Sim, `GPT2Model.wte`

---

### 2. Segment Embeddings (GPT-2)
**Não aplicável** - retorna mensagem:
```python
return ui.HTML("<p>Segment embeddings are not applicable for this model (e.g. GPT-2).</p>")
```

---

### 3. Positional Embeddings (GPT-2)
GPT-2 usa **learned positional embeddings**:
```python
pos_embed = encoder_model.wpe(position_ids)  # GPT2Model.wpe (position embeddings)
```

Mas visualização ainda usa fórmula sinusoidal para consistência.

**Biblioteca Hugging Face:** Sim, `GPT2Model.wpe`

---

### 4. Sum & Layer Normalization (GPT-2)
```python
word_embed = encoder_model.wte(input_ids)
pos_embed = encoder_model.wpe(position_ids)
summed = word_embed + pos_embed

# GPT-2 não tem LayerNorm pré-layer como BERT
# LayerNorm é aplicado dentro de cada bloco transformer
normalized = summed  # Mostra apenas soma
```

---

### 5. Q/K/V Projections (GPT-2)
GPT-2 usa `Conv1D` em vez de `Linear`:
```python
layer = encoder_model.h[layer_idx].attn  # GPT2Block.attn

# c_attn projeta para Q, K, V simultaneamente
c_attn_out = layer.c_attn(hs_in)[0]  # (seq, 3*hidden)
q, k, v = c_attn_out.split(hs_in.shape[-1], dim=-1)

Q = q.cpu().numpy()
K = k.cpu().numpy()
V = v.cpu().numpy()
```

**Biblioteca Hugging Face:** Sim, `GPT2Attention.c_attn` (Conv1D)

---

### 6. Scaled Dot-Product Attention (GPT-2)
**Implementação idêntica ao BERT**, mas matriz de atenção é **lower-triangular** devido à máscara causal.

---

### 7. Multi-Head Attention Map (GPT-2)
Visualização mostra claramente a **máscara causal** - metade superior da matriz é zero:
```python
att = attentions[layer_idx][0, head_idx].cpu().numpy()
# att é lower-triangular para GPT-2
```

**Nota na UI:**
```
"Visualizes attention weights. Lower triangular due to causal masking."
```

---

### 8. Attention Flow (GPT-2)
**Implementação idêntica ao BERT** - Sankey diagram mostra fluxo causal.

---

### 9. Feed-Forward Network (GPT-2)
GPT-2 usa **MLP** com `Conv1D`:
```python
layer = encoder_model.h[layer_idx]

# c_fc: expansão (768 -> 3072)
inter = layer.mlp.c_fc(hs_in)
inter_act = layer.mlp.act(inter)  # GELU activation

# c_proj: projeção (3072 -> 768)
proj = layer.mlp.c_proj(inter_act)
```

**Biblioteca Hugging Face:** Sim, `GPT2MLP.{c_fc, act, c_proj}`

---

### 10-15. Outras Sections (GPT-2)
As sections seguintes funcionam **identicamente ao BERT**:
- Add & Norm
- Global Attention Metrics
- Attention Head Specialization
- Attention Dependency Tree
- Inter-Sentence Attention
- Hidden States

---

### 16. Next Token Predictions (GPT-2)
Diferença crucial: **predição autoregressiva** vs masked tokens.

**Implementação:**
```python
mlm_outputs = mlm_model(**inputs)  # GPT2LMHeadModel
logits = mlm_outputs.logits[0]  # (seq_len, vocab_size)

# logits[i] prediz token[i+1] em GPT-2
# Shift logits right by 1 para alinhamento
shifted_logits = torch.zeros_like(logits)
shifted_logits[1:] = logits[:-1]

probs = torch.softmax(shifted_logits, dim=-1)
```

**Nota na UI:**
```
"Probabilities for the next token (Softmax output)."
```

**Biblioteca Hugging Face:** Sim, `GPT2LMHeadModel.from_pretrained()`

---

## Funcionalidades Partilhadas

### 1. Tokenização
**BERT:**
```python
tokenizer(text, return_tensors="pt")
# ou com dois segments:
tokenizer(sentence_a, sentence_b, return_tensors="pt")
```

**GPT-2:**
```python
tokenizer(text, return_tensors="pt")
# Não suporta dois segments
```

**Biblioteca:** Hugging Face `{Bert,GPT2}Tokenizer`

---

### 2. Model Inference
```python
with torch.no_grad():
    outputs = encoder_model(**inputs)

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
embeddings = outputs.last_hidden_state[0].cpu().numpy()
attentions = outputs.attentions  # Tuple de tensores
hidden_states = outputs.hidden_states  # Tuple de tensores
```

**Biblioteca:** Hugging Face `{Bert,GPT2}Model`

---

### 3. Visualizações
**Bibliotecas usadas:**
- **Matplotlib:** Geração de heatmaps convertidos para base64
- **Plotly:** Gráficos interativos (heatmaps, Sankey, scatter, radar)
- **Shiny for Python:** Framework web reativo

**Exemplo - Heatmap inline:**
```python
import matplotlib.pyplot as plt
import io
import base64

def array_to_base64_img(arr, cmap="viridis", height=0.2):
    fig, ax = plt.subplots(figsize=(6, height))
    ax.imshow(arr.reshape(1, -1), cmap=cmap, aspect='auto')
    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return img_base64
```

---

## Diferenças Técnicas entre BERT e GPT-2

| Aspecto | BERT | GPT-2 |
|---------|------|-------|
| **Arquitetura** | Encoder bidireccional | Decoder com máscara causal |
| **Embedding Modules** | `wte`, `position_embeddings`, `token_type_embeddings` | `wte`, `wpe` (sem segments) |
| **Atenção** | Bidireccional (full matrix) | Causal (lower-triangular) |
| **Projeções Q/K/V** | `query()`, `key()`, `value()` Linear layers | `c_attn` Conv1D único |
| **FFN** | `intermediate.dense`, `output.dense` | `mlp.c_fc`, `mlp.c_proj` Conv1D |
| **LayerNorm** | Pré-embedding e dentro de layers | Dentro de cada bloco |
| **Tokenizer** | WordPiece com `##` para subwords | BPE com `Ġ` para espaços |
| **Predições** | Masked Language Modeling | Next Token Prediction |
| **Head de Saída** | `BertForMaskedLM` | `GPT2LMHeadModel` |
| **Hugging Face Class** | `BertModel`, `BertForMaskedLM` | `GPT2Model`, `GPT2LMHeadModel` |

---

## Detecção Automática de Modelo

O código detecta automaticamente o tipo de modelo:

```python
# Servidor - models.py
is_gpt2 = "gpt2" in model_name

# Servidor - isa.py
is_gpt_style = any(tok.startswith("Ġ") or tok.startswith("Â") for tok in tokens)
is_bert_style = any("##" in tok for tok in tokens)

# Servidor - server.py
is_causal = "GPT2" in type(mlm_model).__name__

# Biblioteca interna
if hasattr(encoder_model, "encoder"):  # BERT
    layer = encoder_model.encoder.layer[layer_idx]
else:  # GPT-2
    layer = encoder_model.h[layer_idx]
```

---

## Resumo Final

**Todas as sections** são implementadas usando:
1. **Hugging Face Transformers** para carregar modelos e extrair:
   - Embeddings (`word_embeddings`, `position_embeddings`)
   - Attentions (`outputs.attentions`)
   - Hidden states (`outputs.hidden_states`)
   - Logits (de MLM/LM heads)

2. **Bibliotecas auxiliares:**
   - **spaCy:** POS tagging e NER para Head Specialization
   - **NLTK:** Sentence tokenization para ISA
   - **NumPy/PyTorch:** Cálculos numéricos e agregações
   - **Matplotlib:** Heatmaps inline (base64)
   - **Plotly:** Visualizações interativas
   - **Shiny for Python:** Framework web reativo

3. **Código customizado:**
   - Aggregação de atenções (ISA, metrics, flow)
   - Análise comportamental (head specialization)
   - Árvores de dependência (influence tree)
   - Visualizações (todas as funções `get_*_view()`)

**Conclusão:** A aplicação é um **wrapper visual sofisticado** sobre modelos Hugging Face, que expõe e visualiza todos os estados internos e computações intermediárias dos Transformers, permitindo compreensão profunda do mecanismo de atenção.
