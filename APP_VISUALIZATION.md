# Attention Atlas - Application Visualization Flow

This diagram shows what the Attention Atlas application visualizes and how users interact with it.

## Application Features & Flow

```mermaid
flowchart TD
    Start([User Opens App]) --> Input[Enter Text Input<br/>'the cat sat on the mat']

    Input --> Generate[Click 'Generate All' Button]

    Generate --> Processing{Processing BERT Model}

    Processing --> Section1[📊 Section 1: Embeddings]
    Processing --> Section2[🔍 Section 2: Q/K/V Projections]
    Processing --> Section3[🎯 Section 3: Multi-Head Attention]
    Processing --> Section4[🔄 Section 4: Residual & FFN]
    Processing --> Section5[📈 Section 5: Output Predictions]

    subgraph Embeddings [Section 1: Input Embeddings]
        Section1 --> E1[Token Embeddings<br/>Heatmap visualization<br/>768 dims per token]
        Section1 --> E2[Segment Embeddings<br/>A/B sequence chips<br/>Color-coded tokens]
        Section1 --> E3[Positional Embeddings<br/>Sinusoidal patterns<br/>Position encodings]
        Section1 --> E4[Sum + LayerNorm<br/>Combined embeddings<br/>Before/after normalization]
    end

    subgraph QKV [Section 2: Attention Projections]
        Section2 --> Q1[Select Layer 0-11<br/>Mini dropdown selector]
        Q1 --> Q2[Q/K/V Heatmaps<br/>Per-token projections<br/>Color: Green/Orange/Purple]
        Q1 --> Q3[Scaled Attention Formula<br/>Focus token selector<br/>Top-3 connections shown]
    end

    subgraph MultiHead [Section 3: Multi-Head Attention - MAIN FEATURE]
        Section3 --> M1[Layer Selector<br/>Choose: 0-11]
        Section3 --> M2[Head Selector<br/>Choose: 0-11]

        M1 --> M3[Attention Map<br/>Interactive Plotly Heatmap]
        M2 --> M3

        M3 --> M4[Hover: See Calculations<br/>Q·K dot product<br/>Scaled value<br/>Softmax result]

        M1 --> M5[Attention Flow Diagram<br/>Token-to-token connections]
        M2 --> M5

        M5 --> M6[Click Token Buttons<br/>Focus specific token<br/>Highlight outgoing attention]

        M6 --> M7[Updated Flow View<br/>Dimmed unrelated connections<br/>Emphasized selected token]

        M3 --> M8[6 Attention Metrics Cards<br/>Clickable for formulas]

        M8 --> M9[Click Metric Card]
        M9 --> M10[Modal Window Opens<br/>Formula + Interpretation<br/>Scientific Reference]
    end

    subgraph Residual [Section 4: Residual Connections & FFN]
        Section4 --> R1[Add & Norm After Attention<br/>Change magnitude bars<br/>Per-token residual impact]
        Section4 --> R2[Feed Forward Network<br/>GELU activation heatmap<br/>3072→768 projection]
        Section4 --> R3[Add & Norm After FFN<br/>Final residual connection<br/>Layer output visualization]
    end

    subgraph Output [Section 5: Model Outputs]
        Section5 --> O1[Hidden States<br/>Final layer representations<br/>768 dims per token]
        Section5 --> O2[Toggle: Use MLM Head<br/>Switch on/off]

        O2 -->|ON| O3[MLM Top-5 Predictions<br/>Token probabilities<br/>Per position]
        O2 -->|OFF| O4[Disabled Message<br/>Enable to see predictions]

        O3 --> O5[Click Probability Button<br/>Show softmax calculation]
        O5 --> O6[Expanded Formula<br/>exp logit / sum exp<br/>Detailed breakdown]
    end

    M10 --> Explore[Continue Exploring<br/>Change layers/heads/tokens]
    O6 --> Explore
    Explore --> Input

    style Start fill:#e1f5ff
    style Generate fill:#ffe1e1
    style Embeddings fill:#fff4e6
    style QKV fill:#e6f7ff
    style MultiHead fill:#ffe6f7
    style Residual fill:#f0e6ff
    style Output fill:#e6ffe6
```

## Interactive Features Map

```mermaid
mindmap
  root((Attention Atlas<br/>Interactive Features))
    Layer Navigation
      12 Encoder Layers
      Mini Select Dropdowns
      Real-time Updates
    Head Selection
      12 Attention Heads
      Per Layer
      Independent Exploration
    Token Interaction
      Click to Focus
      Color-coded Flows
      Reset to Show All
    Visualizations
      Plotly Heatmaps
        Hoverable
        Zoomable
        Calculation Details
      Attention Flows
        Curved Connections
        Width = Attention Weight
        Color = Source Token
      Embedding Heatmaps
        Base64 PNG strips
        64-96 dims shown
        Hover for values
    Metrics Dashboard
      6 Metric Cards
        Confidence Max
        Confidence Avg
        Focus Entropy
        Sparsity
        Distribution
        Uniformity
      Click for Details
        Mathematical Formula
        Interpretation Guide
        Scientific Paper Link
    MLM Predictions
      Toggle Switch
      Top-5 Per Token
      Expandable Formulas
      Softmax Breakdown
```

## User Interaction Flow

```mermaid
sequenceDiagram
    participant User
    participant UI as Attention Atlas UI
    participant BERT as BERT Model
    participant Viz as Visualizations

    User->>UI: Enter text input
    User->>UI: Click "Generate All"
    UI->>UI: Show loading spinner
    UI->>BERT: Process text
    BERT->>BERT: Tokenize
    BERT->>BERT: Generate embeddings
    BERT->>BERT: Run 12 encoder layers
    BERT->>BERT: Extract attentions & states
    BERT-->>UI: Return all outputs
    UI->>UI: Hide spinner
    UI->>Viz: Render embeddings
    UI->>Viz: Render Q/K/V projections
    UI->>Viz: Render attention maps
    UI->>Viz: Render metrics
    UI->>Viz: Render FFN & residuals
    UI->>Viz: Render hidden states

    loop User Exploration
        User->>UI: Select layer (0-11)
        UI->>Viz: Update attention map
        UI->>Viz: Update Q/K/V display
        UI->>Viz: Update FFN display

        User->>UI: Select head (0-11)
        UI->>Viz: Update attention map
        UI->>Viz: Update attention flow
        UI->>Viz: Update metrics

        User->>UI: Click token button
        UI->>Viz: Highlight token flows
        UI->>Viz: Dim other connections

        User->>UI: Click metric card
        UI->>UI: Show modal with formula
        User->>UI: Close modal

        User->>UI: Toggle MLM switch
        alt MLM Enabled
            UI->>BERT: Run MLM head
            BERT-->>UI: Return probabilities
            UI->>Viz: Show Top-5 predictions
        else MLM Disabled
            UI->>Viz: Show info message
        end
    end
```

## Data Flow: From Input to Visualization

```mermaid
flowchart LR
    subgraph Input
        Text[Input Text] --> Tokens[Tokenized<br/>Sequence]
    end

    subgraph ModelProcessing [BERT Model Processing]
        Tokens --> Emb[Embeddings<br/>Layer]
        Emb --> Layers[12 Encoder<br/>Layers<br/>0-11]
        Layers --> HS[Hidden<br/>States]
        HS --> MLM[MLM<br/>Head]
    end

    subgraph Visualizations [Attention Atlas Displays]
        Emb --> V1[Embedding<br/>Tables]
        Layers --> V2[Attention<br/>Heatmaps]
        Layers --> V3[QKV<br/>Projections]
        Layers --> V4[Metrics<br/>Cards]
        Layers --> V5[FFN<br/>Displays]
        HS --> V6[Hidden<br/>State Table]
        MLM --> V7[Top-5<br/>Predictions]
    end

    subgraph UserControls [User Controls]
        C1[Layer Selector<br/>0-11] -.->|Select Layer| V2
        C2[Head Selector<br/>0-11] -.->|Select Head| V2
        C3[Token Click<br/>Focus] -.->|Highlight| V2
        C4[Toggle Switch<br/>On/Off] -.->|Enable MLM| V7
    end

    V2 -.->|Update on<br/>Selection| C1
    V2 -.->|Update on<br/>Selection| C2
    V2 -.->|Update on<br/>Click| C3
    V7 -.->|Show/Hide| C4

    style ModelProcessing fill:#ffe6e6
    style Visualizations fill:#e6f7ff
    style UserControls fill:#f0ffe6
```

## Visualization Components

### Section 1: Input Embeddings
- **Token Embeddings**: Heatmap strips showing first 64 dimensions
- **Segment Embeddings**: Color-coded chips (blue/purple) for sentence A/B
- **Positional Embeddings**: Sinusoidal pattern visualization
- **Sum + LayerNorm**: Before/after normalization comparison

### Section 2: Q/K/V Projections
- **Layer Selector**: Mini dropdown (0-11)
- **Three Heatmaps**: Green (Query), Orange (Key), Purple (Value)
- **Scaled Attention**: Formula walkthrough for selected token

### Section 3: Multi-Head Attention (MAIN)
- **Attention Map**: Interactive Plotly heatmap with hover details
  - Shows: Q·K dot product, scaled value, softmax result
  - Click cells to inspect calculations
- **Attention Flow**: Sankey-style token connections
  - Line width = attention weight
  - Click tokens to focus
- **Token Buttons**: Click to highlight specific token's outgoing attention
- **6 Metric Cards**: Click for mathematical formulas and explanations

### Section 4: Residual & FFN
- **Add & Norm**: Bar charts showing magnitude of changes
- **FFN Visualization**: Intermediate (3072) and projection (768) heatmaps
- **Residual Impact**: Per-token change visualization

### Section 5: Outputs
- **Hidden States**: Final layer representations
- **MLM Predictions**: Toggle-able Top-5 token probabilities
- **Expandable Formulas**: Click to see softmax calculations

## Key Features

1. **Real-time Processing**: All calculations done on-demand
2. **12 × 12 = 144 Attention Heads**: Fully explorable
3. **Interactive Plots**: Hover, click, zoom on Plotly visualizations
4. **Educational Modals**: Mathematical formulas with interpretations
5. **Scientific References**: Links to research papers for metrics
6. **Responsive Design**: Dark theme sidebar + light content area
