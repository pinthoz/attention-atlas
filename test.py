from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_plotly
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# ===============================
# 1. Load models once
# ===============================
MODEL_NAME = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

encoder_model = BertModel.from_pretrained(
    MODEL_NAME,
    output_attentions=True,
    output_hidden_states=True,
)
encoder_model.eval()

mlm_model = BertForMaskedLM.from_pretrained(
    MODEL_NAME,
    output_attentions=False,
    output_hidden_states=False,
)
mlm_model.eval()


# ===============================
# 2. Helper functions
# ===============================
def positional_encoding(position: int, d_model: int = 768) -> np.ndarray:
    pe = np.zeros((position, d_model))
    for pos in range(position):
        for i in range(0, d_model, 2):
            pe[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos / (10000 ** ((2 * i) / d_model)))
    return pe


def array_to_base64_img(array: np.ndarray, cmap: str = "Blues", height: float = 0.22) -> str:
    plt.figure(figsize=(3, height))
    plt.imshow(array[np.newaxis, :], cmap=cmap, aspect="auto")
    plt.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ===============================
# 3. UI
# ===============================
app_ui = ui.page_fluid(
    ui.tags.style(
        """
        body {
            background-color:#f5f7fb;
            font-family:'Inter', system-ui, -apple-system, sans-serif;
            margin:0;
        }
        .sidebar {
            position:fixed;
            left:0; top:0; bottom:0;
            width:360px;
            background:#1e1e2e;
            color:white;
            padding:28px;
            overflow-y:auto;
            box-shadow:4px 0 24px rgba(0,0,0,0.3);
        }
        .sidebar h3 {
            color:#ff5ca9;
            font-weight:700;
            margin:0 0 8px;
            font-size:24px;
        }
        .sidebar small {
            color:#9ca3af;
            font-size:12px;
            display:block;
            margin-bottom:16px;
        }
        .sidebar ul {
            padding-left:20px;
            margin:0 0 16px;
            font-size:11px;
            color:#cbd5e1;
        }
        .sidebar li {margin-bottom:4px;}
        .content {
            margin-left:380px;
            padding:25px;
        }
        .btn-primary {
            background:#ff5ca9!important;
            border-color:#ff5ca9!important;
            padding:8px 20px;
            font-size:13px;
            font-weight:600;
            border-radius:999px;
            transition:all 0.2s;
        }
        .btn-primary:hover {
            background:#ff74b8!important;
            border-color:#ff74b8!important;
            transform:translateY(-1px);
            box-shadow:0 4px 12px rgba(255,92,169,0.3);
        }
        .btn-container {
            display:flex;
            align-items:center;
            gap:12px;
            margin-top:8px;
        }
        .spinner {
            display:inline-block;
            width:20px;
            height:20px;
            border-radius:999px;
            border:3px solid rgba(255,92,169,0.2);
            border-top-color:#ff5ca9;
            animation:spin 0.6s linear infinite;
        }
        @keyframes spin {
            0% {transform:rotate(0deg);}
            100% {transform:rotate(360deg);}
        }
        .processing-text {
            font-size:11px;
            color:#ff5ca9;
            font-weight:500;
            animation:pulse 1.5s ease-in-out infinite;
        }
        @keyframes pulse {
            0%, 100% {opacity:0.5;}
            50% {opacity:1;}
        }
        .card {
            background:white;
            border-radius:15px;
            padding:18px;
            box-shadow:0 2px 8px rgba(0,0,0,0.1);
            margin-bottom:20px;
            overflow-x:auto;
        }
        .card h4 {
            margin:0 0 12px;
            font-size:15px;
            font-weight:600;
            color:#111827;
        }
        .sub-label {
            font-size:10px;
            color:#6b7280;
            margin-bottom:8px;
        }
        .grid-3 {
            display:grid;
            grid-template-columns:repeat(3,minmax(0,1fr));
            gap:16px;
        }
        .token-table, .token-table-compact {
            width:100%;
            border-collapse:collapse;
            font-size:10px;
        }
        .token-table th, .token-table-compact th {
            text-align:left;
            padding:6px 4px;
            color:#6b7280;
            border-bottom:2px solid #e5e7eb;
            font-size:9px;
            font-weight:600;
            text-transform:uppercase;
        }
        .token-table td, .token-table-compact td {
            padding:6px 4px;
            vertical-align:middle;
            border-bottom:1px solid #f3f4f6;
        }
        .token-name {
            font-weight:600;
            color:#111827;
            min-width:50px;
            font-family:monospace;
            font-size:10px;
        }
        .card-scroll {
            max-height:300px;
            overflow-y:auto;
        }
        img.heatmap {
            border-radius:4px;
            display:block;
            cursor:help;
            width:100%;
            max-width:200px;
        }
        .header-controls {
            display:flex;
            justify-content:space-between;
            align-items:center;
            gap:10px;
            margin-bottom:8px;
        }
        .header-right {
            display:flex;
            align-items:center;
            gap:8px;
            font-size:10px;
            color:#6b7280;
        }
        .select-mini select {
            padding:2px 20px 2px 8px;
            font-size:10px;
            border-radius:6px;
            border:1px solid #e5e7eb;
        }
        """
    ),

    # Sidebar (fixed)
    ui.div(
        {"class": "sidebar"},
        ui.h3("Transformer Dashboard"),
        ui.tags.small("Complete BERT architecture visualization with MLM head."),
        ui.tags.ul(
            ui.tags.li("Token Embeddings + Positional Encoding"),
            ui.tags.li("Q/K/V projections & Multi-Head Attention"),
            ui.tags.li("Add & Norm + Feed Forward Network"),
            ui.tags.li("Linear Projection + Softmax"),
            ui.tags.li("Output Probabilities (Top-5)"),
        ),
        ui.input_text("text_input", "Input sentence:", "the cat sat on the mat"),
        ui.tags.div(
            {"class": "btn-container"},
            ui.input_action_button("generate_all", "Generate All", class_="btn btn-primary"),
            ui.output_ui("spinner_ui"),
        ),
        ui.hr(),
        ui.input_switch("use_mlm", "Use MLM head for predictions", value=True),
        ui.tags.small(
            "When enabled, shows real token probabilities from BertForMaskedLM.",
            style="display:block;color:#9ca3af;margin-top:4px;font-size:10px;",
        ),
        ui.hr(),
        ui.p("🖱 Hover over heatmaps to inspect vectors",
             style="font-size:11px;color:#cbd5e1;margin-top:12px;"),
    ),

    # Main content
    ui.div(
        {"class": "content"},

        ui.card(
            ui.h4("📝 Sentence Preview"),
            ui.output_text_verbatim("preview_text"),
        ),

        # Embeddings / Positional Encoding / QKV
        ui.div(
            {"class": "grid-3"},
            ui.card(
                ui.h4("Token Embeddings"),
                ui.div({"class": "sub-label"}, "Contextual word vectors (768 dims)"),
                ui.output_ui("embedding_table"),
            ),
            ui.card(
                ui.h4("Positional Encodings"),
                ui.div({"class": "sub-label"}, "Sinusoidal position encodings"),
                ui.output_ui("posenc_table"),
            ),
            ui.card(
                ui.div(
                    {"class": "header-controls"},
                    ui.h4("Q / K / V Projections"),
                    ui.div(
                        {"class": "header-right"},
                        ui.span("Layer"),
                        ui.tags.div(
                            {"class": "select-mini"},
                            ui.input_select(
                                "qkv_layer", None,
                                {str(i): f"L{i}" for i in range(12)},
                                selected="0",
                            ),
                        ),
                    ),
                ),
                ui.output_ui("qkv_table"),
            ),
        ),

        # Multi-head attention
        ui.card(
            ui.div(
                {"class": "header-controls"},
                ui.h4("🔍 Multi-Head Attention Map"),
                ui.div(
                    {"class": "header-right"},
                    ui.span("Layer"),
                    ui.tags.div(
                        {"class": "select-mini"},
                        ui.input_select(
                            "att_layer", None,
                            {str(i): f"{i}" for i in range(12)},
                            selected="0",
                        ),
                    ),
                    ui.span("Head"),
                    ui.tags.div(
                        {"class": "select-mini"},
                        ui.input_select(
                            "att_head", None,
                            {str(i): f"{i}" for i in range(12)},
                            selected="0",
                        ),
                    ),
                ),
            ),
            output_widget("attention_map"),
        ),

        # Add&Norm / FFN / Layer output
        ui.div(
            {"class": "grid-3"},
            ui.card(
                ui.h4("Add & Norm"),
                ui.div({"class": "sub-label"}, "Residual connections + LayerNorm"),
                ui.output_ui("add_norm_view"),
            ),
            ui.card(
                ui.h4("Feed Forward"),
                ui.div({"class": "sub-label"}, "FFN layer activation patterns"),
                ui.output_ui("ffn_view"),
            ),
            ui.card(
                ui.h4("Layer Stats"),
                ui.div({"class": "sub-label"}, "Summary statistics per token"),
                ui.output_ui("layer_output_view"),
            ),
        ),

        # Final hidden & MLM Top-5
        ui.card(
            ui.h4("Output Layer: Linear Projection + Softmax"),
            ui.div(
                {"style": "display:grid;grid-template-columns:1fr 2fr;gap:20px;"},
                ui.div(
                    ui.div({"class": "sub-label"}, "Final hidden states (encoder output)"),
                    ui.output_ui("linear_projection_viz"),
                ),
                ui.div(
                    ui.div({"class": "sub-label"}, "Top-5 Token Predictions with Probabilities"),
                    ui.output_ui("output_probabilities"),
                ),
            ),
        ),
    ),
)


# ===============================
# 4. Server
# ===============================
def server(input, output, session):
    running = reactive.Value(False)

    # Computa tudo do encoder quando clicas "Generate All"
    @reactive.event(input.generate_all)
    def compute_all():
        text = input.text_input().strip()
        if not text:
            running.set(False)
            return None

        running.set(True)
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = encoder_model(**inputs)

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        embeddings = outputs.last_hidden_state[0].cpu().numpy()
        attentions = outputs.attentions
        hidden_states = outputs.hidden_states

        pos_enc = positional_encoding(len(tokens), embeddings.shape[1])

        running.set(False)
        return tokens, embeddings, pos_enc, attentions, hidden_states, inputs

    # Spinner ao lado do botão
    @output
    @render.ui
    def spinner_ui():
        if running.get():
            return ui.div(
                {"style": "display:flex;align-items:center;gap:8px;"},
                ui.tags.div({"class": "spinner"}),
                ui.tags.span("Processing...", class_="processing-text"),
            )
        return ""

    # Sentence preview
    @output
    @render.text
    def preview_text():
        t = input.text_input().strip()
        return f'"{t}"' if t else "Type a sentence and click Generate All."

    # Embeddings table
    @output
    @render.ui
    def embedding_table():
        res = compute_all()
        if not res:
            return ui.HTML("<p style='font-size:10px;color:#9ca3af;'>Click <b>Generate All</b> to start</p>")
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res

        rows = []
        for i, tok in enumerate(tokens):
            vec = embeddings[i]
            strip = array_to_base64_img(vec[:64], cmap="Blues", height=0.18)
            tip = "Embedding (first 32 dims): " + ", ".join(f"{v:.3f}" for v in vec[:32])
            rows.append(
                f"<tr>"
                f"<td class='token-name'>{tok}</td>"
                f"<td><img class='heatmap' src='data:image/png;base64,{strip}' title='{tip}'></td>"
                f"</tr>"
            )
        html = (
            "<div class='card-scroll'>"
            "<table class='token-table'><tr><th>Token</th><th>Embedding Vector</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )
        return ui.HTML(html)

    # Positional encodings table
    @output
    @render.ui
    def posenc_table():
        res = compute_all()
        if not res:
            return ui.HTML("")
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res

        rows = []
        for i, tok in enumerate(tokens):
            pe = pos_enc[i]
            strip = array_to_base64_img(pe[:64], cmap="RdBu", height=0.18)
            tip = f"Position {i} encoding: " + ", ".join(f"{v:.3f}" for v in pe[:32])
            rows.append(
                f"<tr>"
                f"<td class='token-name'>{tok}</td>"
                f"<td><img class='heatmap' src='data:image/png;base64,{strip}' title='{tip}'></td>"
                f"</tr>"
            )
        html = (
            "<div class='card-scroll'>"
            "<table class='token-table'><tr><th>Token</th><th>Position Encoding</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )
        return ui.HTML(html)

    # Q/K/V table
    @output
    @render.ui
    def qkv_table():
        res = compute_all()
        if not res:
            return ui.HTML("")
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res

        layer_idx = int(input.qkv_layer())
        layer = encoder_model.encoder.layer[layer_idx].attention.self
        hs_in = hidden_states[layer_idx]

        with torch.no_grad():
            Q = layer.query(hs_in)[0].cpu().numpy()
            K = layer.key(hs_in)[0].cpu().numpy()
            V = layer.value(hs_in)[0].cpu().numpy()

        rows = []
        for i, tok in enumerate(tokens):
            q_strip = array_to_base64_img(Q[i][:48], "Greens", 0.12)
            k_strip = array_to_base64_img(K[i][:48], "Oranges", 0.12)
            v_strip = array_to_base64_img(V[i][:48], "Purples", 0.12)
            q_tip = "Query: " + ", ".join(f"{x:.3f}" for x in Q[i][:24])
            k_tip = "Key: " + ", ".join(f"{x:.3f}" for x in K[i][:24])
            v_tip = "Value: " + ", ".join(f"{x:.3f}" for x in V[i][:24])
            cell = (
                f"<div style='font-size:9px;color:#065f46;font-weight:600;'>Q</div>"
                f"<img class='heatmap' src='data:image/png;base64,{q_strip}' title='{q_tip}'>"
                f"<div style='font-size:9px;color:#92400e;margin-top:2px;font-weight:600;'>K</div>"
                f"<img class='heatmap' src='data:image/png;base64,{k_strip}' title='{k_tip}'>"
                f"<div style='font-size:9px;color:#4c1d95;margin-top:2px;font-weight:600;'>V</div>"
                f"<img class='heatmap' src='data:image/png;base64,{v_strip}' title='{v_tip}'>"
            )
            rows.append(f"<tr><td class='token-name'>{tok}</td><td>{cell}</td></tr>")

        html = (
            "<div class='card-scroll'>"
            "<table class='token-table-compact'><tr><th>Token</th><th>Vectors</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )
        return ui.HTML(html)

    # Attention map (Plotly)
    @render_plotly
    def attention_map():
        res = compute_all()
        if not res:
            return px.imshow([[0]], title="Click Generate All")
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res

        layer_idx = int(input.att_layer())
        head_idx = int(input.att_head())

        if attentions is None or len(attentions) == 0:
            return px.imshow([[0]], title="No attentions available")

        att = attentions[layer_idx][0, head_idx].cpu().numpy()

        # custom hover data
        layer = encoder_model.encoder.layer[layer_idx].attention.self
        hs_in = hidden_states[layer_idx]
        with torch.no_grad():
            Q = layer.query(hs_in)[0].cpu().numpy()
            K = layer.key(hs_in)[0].cpu().numpy()

        L = len(tokens)
        d_k = Q.shape[-1] // 12
        custom = np.empty((L, L, 5), dtype=object)
        for i in range(L):
            for j in range(L):
                dot_product = np.dot(Q[i], K[j])
                scaled = dot_product / np.sqrt(d_k)
                
                custom[i, j, 0] = np.array2string(Q[i][:5], precision=3, separator=", ")
                custom[i, j, 1] = np.array2string(K[j][:5], precision=3, separator=", ")
                custom[i, j, 2] = f"{dot_product:.4f}"
                custom[i, j, 3] = f"{scaled:.4f}"
                custom[i, j, 4] = f"{att[i, j]:.4f}"

        hover = (
            "<b>Query Token:</b> %{y}<br>"
            "<b>Key Token:</b> %{x}<br>"
            "<br><b>Calculation:</b><br>"
            "1️⃣ Dot Product: Q·K = %{customdata[2]}<br>"
            "2️⃣ Scaled: (Q·K)/√d_k = %{customdata[3]}<br>"
            "3️⃣ Softmax Result: <b>%{customdata[4]}</b><br>"
            "<br><b>Vectors (first 5 dims):</b><br>"
            "Q = %{customdata[0]}<br>"
            "K = %{customdata[1]}<br>"
            "<extra></extra>"
        )

        fig = px.imshow(
            att,
            x=tokens,
            y=tokens,
            color_continuous_scale="Blues",
            aspect="auto",
            title=f"Layer {layer_idx} · Head {head_idx}",
        )
        fig.update_traces(customdata=custom, hovertemplate=hover)
        fig.update_layout(
            xaxis_title="Key (attending to)",
            yaxis_title="Query (attending from)",
            margin=dict(l=40, r=10, t=40, b=40),
            coloraxis_colorbar=dict(title="Attention"),
            plot_bgcolor="#111111",
            paper_bgcolor="#111111",
            font=dict(color="white"),
        )
        return fig

    # Add & Norm diff view
    @output
    @render.ui
    def add_norm_view():
        res = compute_all()
        if not res:
            return ui.HTML("")
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res
        layer_idx = int(input.att_layer())

        if layer_idx + 1 >= len(hidden_states):
            return ui.HTML("")

        hs_in = hidden_states[layer_idx][0].cpu().numpy()
        hs_out = hidden_states[layer_idx + 1][0].cpu().numpy()

        rows = []
        for i, tok in enumerate(tokens):
            diff = np.linalg.norm(hs_out[i] - hs_in[i])
            norm = np.linalg.norm(hs_in[i]) + 1e-6
            ratio = diff / norm
            width = max(4, min(100, int(ratio * 80)))
            rows.append(
                f"<tr><td class='token-name'>{tok}</td>"
                f"<td><div style='background:#e5e7eb;border-radius:999px;height:10px;'>"
                f"<div style='width:{width}%;height:10px;border-radius:999px;"
                f"background:linear-gradient(90deg,#22c55e,#22d3ee);'></div>"
                f"</div></td></tr>"
            )
        html = (
            "<div class='card-scroll'>"
            "<table class='token-table'><tr><th>Token</th><th>Change Magnitude</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )
        return ui.HTML(html)

    # FFN view
    @output
    @render.ui
    def ffn_view():
        res = compute_all()
        if not res:
            return ui.HTML("")
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res
        layer_idx = int(input.att_layer())

        if layer_idx + 1 >= len(hidden_states):
            return ui.HTML("")

        hs = hidden_states[layer_idx + 1][0].cpu().numpy()
        rows = []
        for i, tok in enumerate(tokens):
            strip = array_to_base64_img(hs[i][:96], "plasma", 0.18)
            rows.append(
                f"<tr><td class='token-name'>{tok}</td>"
                f"<td><img class='heatmap' src='data:image/png;base64,{strip}' "
                f"title='Layer {layer_idx} output pattern'></td></tr>"
            )
        html = (
            "<div class='card-scroll'>"
            "<table class='token-table'><tr><th>Token</th><th>Activation Pattern</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )
        return ui.HTML(html)

    # Layer stats
    @output
    @render.ui
    def layer_output_view():
        res = compute_all()
        if not res:
            return ui.HTML("")
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res
        layer_idx = int(input.att_layer())

        if layer_idx + 1 >= len(hidden_states):
            return ui.HTML("")

        hs = hidden_states[layer_idx + 1][0].cpu().numpy()
        rows = []
        for i, tok in enumerate(tokens):
            mean_val = float(hs[i].mean())
            std_val = float(hs[i].std())
            max_val = float(hs[i].max())
            rows.append(
                f"<tr><td class='token-name'>{tok}</td>"
                f"<td style='font-size:9px;color:#374151;'>"
                f"μ={mean_val:.3f}, σ={std_val:.3f}, max={max_val:.3f}</td></tr>"
            )
        html = (
            "<div class='card-scroll'>"
            "<table class='token-table'><tr><th>Token</th><th>Statistics</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )
        return ui.HTML(html)

    # Final hidden state strips
    @output
    @render.ui
    def linear_projection_viz():
        res = compute_all()
        if not res:
            return ui.HTML("")
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res

        final_h = hidden_states[-1][0].cpu().numpy()
        rows = []
        for i, tok in enumerate(tokens):
            strip = array_to_base64_img(final_h[i][:64], "magma", 0.18)
            tip = "Final encoder hidden state (768 dims)"
            rows.append(
                f"<tr><td class='token-name'>{tok}</td>"
                f"<td><img class='heatmap' src='data:image/png;base64,{strip}' title='{tip}'></td></tr>"
            )
        html = (
            "<div class='card-scroll'>"
            "<table class='token-table'><tr><th>Token</th><th>Hidden State</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )
        return ui.HTML(html)

    # Top-5 probs (MLM head)
    @output
    @render.ui
    def output_probabilities():
        res = compute_all()
        if not res:
            return ui.HTML(
                "<p style='font-size:10px;color:#9ca3af;'>Run <b>Generate All</b> first</p>"
            )
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res

        if not input.use_mlm():
            return ui.HTML(
                "<p style='font-size:10px;color:#9ca3af;'>"
                "Enable <b>Use MLM head</b> to see token probabilities."
                "</p>"
            )

        text = input.text_input().strip()
        if not text:
            return ui.HTML("")
        mlm_inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            mlm_outputs = mlm_model(**mlm_inputs)
            probs = torch.softmax(mlm_outputs.logits, dim=-1)[0]

        mlm_tokens = tokenizer.convert_ids_to_tokens(mlm_inputs["input_ids"][0])

        rows = []
        top_k = 5
        for i, tok in enumerate(mlm_tokens):
            token_probs = probs[i]
            top_vals, top_idx = torch.topk(token_probs, top_k)
            inner = "<div style='font-size:10px;'>"
            for p, idx in zip(top_vals, top_idx):
                ptok = tokenizer.decode([idx.item()]) or "[UNK]"
                pval = float(p)
                width = max(4, int(pval * 100))
                inner += (
                    "<div style='margin:3px 0;'>"
                    "<div style='display:flex;align-items:center;gap:6px;'>"
                    f"<span style='font-family:monospace;font-size:10px;min-width:70px;font-weight:600;'>{ptok}</span>"
                    f"<div style='flex:1;background:#e5e7eb;border-radius:999px;height:12px;'>"
                    f"<div style='width:{width}%;height:12px;border-radius:999px;"
                    f"background:linear-gradient(90deg,#3b82f6,#8b5cf6);'></div></div>"
                    f"<span style='font-size:10px;color:#6b7280;width:50px;text-align:right;font-weight:500;'>{pval:.1%}</span>"
                    "</div></div>"
                )
            inner += "</div>"
            rows.append(f"<tr><td class='token-name'>{tok}</td><td>{inner}</td></tr>")

        html = (
            "<div class='card-scroll' style='max-height:350px;'>"
            "<table class='token-table'><tr><th>Position</th><th>Top-5 Predictions</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )
        return ui.HTML(html)


# ===============================
# 5. Run app
# ===============================
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()