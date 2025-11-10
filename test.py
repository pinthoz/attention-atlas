from shiny import App, ui, render, reactive
from shinywidgets import render_plotly, output_widget
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from io import BytesIO
import base64


# ===============================
# 1. Load BERT Model
# ===============================
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, output_attentions=True)
model.eval()


# ===============================
# 2. Helper Functions
# ===============================
def positional_encoding(position, d_model=768):
    """Create positional encodings manually."""
    pe = np.zeros((position, d_model))
    for pos in range(position):
        for i in range(0, d_model, 2):
            pe[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
            pe[pos, i + 1] = np.cos(pos / (10000 ** ((2 * i) / d_model)))
    return pe


def array_to_base64_img(array, cmap="Blues", height=0.25):
    """Convert 1D array to small inline heatmap."""
    plt.figure(figsize=(3, height))
    plt.imshow(array[np.newaxis, :], cmap=cmap, aspect="auto")
    plt.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ===============================
# 3. UI Layout
# ===============================
app_ui = ui.page_fluid(
    ui.tags.style("""
        body {background-color:#f5f7fb;font-family:'Inter';margin:0;}
        .sidebar {position:fixed;left:0;top:0;bottom:0;width:340px;
            background:#1e1e2e;color:white;padding:30px;overflow-y:auto;}
        .sidebar h3 {color:#ff5ca9;font-weight:700;margin-bottom:25px;}
        .content {margin-left:370px;padding:25px;}
        .btn-primary {background:#ff5ca9!important;border-color:#ff5ca9!important;}
        .btn-primary:hover {background:#ff74b8!important;border-color:#ff74b8!important;}
        .card {background:white;border-radius:15px;padding:20px;
            box-shadow:0 2px 8px rgba(0,0,0,0.1);margin-bottom:20px;overflow-x:auto;}
        .card h4 {margin-top:0;color:#333;font-size:15px;margin-bottom:15px;}
        .card-scroll {max-height:300px;overflow-y:auto;}
        .token-table {width:100%;border-collapse:collapse;font-size:11px;}
        .token-table th {text-align:left;padding:8px 6px;color:#555;border-bottom:2px solid #ddd;
            font-size:10px;font-weight:600;text-transform:uppercase;}
        .token-table td {padding:8px 6px;vertical-align:middle;border-bottom:1px solid #eee;}
        .token-table-compact {width:100%;border-collapse:collapse;font-size:11px;}
        .token-table-compact th {text-align:left;padding:8px 6px;color:#555;border-bottom:2px solid #ddd;
            font-size:10px;font-weight:600;text-transform:uppercase;}
        .token-table-compact td {padding:4px;vertical-align:top;border-bottom:1px solid #eee;}
        .token-name {font-weight:600;color:#222;min-width:50px;font-family:monospace;font-size:11px;}
        img.heatmap {border-radius:4px;display:block;cursor:help;width:100%;max-width:200px;}
        .header-controls {display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;gap:10px;}
        .header-controls h4 {margin:0;}
    """),

    ui.div(
        {"class": "sidebar"},
        ui.h3("🧠 Transformer Dashboard"),
        ui.p("Complete Transformer architecture visualization:"),
        ui.tags.ul(
            ui.tags.li("Token Embeddings + Positional Encodings"),
            ui.tags.li("Query, Key, Value transformations"),
            ui.tags.li("Multi-Head Attention mechanism"),
            ui.tags.li("Add & Normalization layers"),
            ui.tags.li("Feed-Forward Network"),
        ),
        ui.input_text("text_input", "Input sentence:", "the cat sat on the mat"),
        ui.input_action_button("generate_all", "Generate All", class_="btn btn-primary mt-2"),
        ui.hr(),
        ui.p("Hover heatmaps to inspect values.", style="font-size:12px;")
    ),

    ui.div(
        {"class": "content"},
        ui.card(ui.h4("Sentence Preview"), ui.output_text_verbatim("preview_text")),
        
        # Embeddings, Positional Encoding e Q/K/V lado a lado
        ui.div(
            {"style": "display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;"},
            ui.card(
                ui.h4("Token Embeddings"),
                ui.output_ui("embedding_table")
            ),
            ui.card(
                ui.h4("Positional Encodings"),
                ui.output_ui("posenc_table")
            ),
            ui.card(
                ui.div(
                    {"class": "header-controls"},
                    ui.h4("Q, K, V Transforms"),
                    ui.input_select(
                        "embedding_layer_select", None,
                        {str(i): f"L{i}" for i in range(12)}, selected="0"
                    ),
                ),
                ui.output_ui("qkv_table")
            ),
        ),
        ui.card(
            ui.div(
                {"class": "header-controls"},
                ui.h4("Multi-Head Attention"),
                ui.div(
                    {"style": "display:flex;gap:10px;align-items:center;"},
                    ui.input_select(
                        "layer_select", None,
                        {str(i): f"Layer {i}" for i in range(12)}, selected="0"
                    ),
                    ui.input_select(
                        "head_select", None,
                        {str(i): f"Head {i}" for i in range(12)}, selected="0"
                    ),
                ),
            ),
            output_widget("attention_map")
        ),
        
        # Resto da arquitetura do Transformer em 3 colunas
        ui.div(
            {"style": "display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;"},
            ui.card(
                ui.h4("🔗 Add & Norm (Post-Attention)"),
                ui.output_ui("add_norm_1_table")
            ),
            ui.card(
                ui.h4("⚡ Feed Forward Network"),
                ui.output_ui("feed_forward_table")
            ),
            ui.card(
                ui.h4("🔗 Add & Norm (Post-FFN)"),
                ui.output_ui("add_norm_2_table")
            ),
        ),
    )
)


# ===============================
# 4. Server Logic
# ===============================
def server(input, output, session):

    @reactive.event(input.generate_all)
    def compute_all():
        text = input.text_input().strip()
        if not text:
            return None
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        embeddings = outputs.last_hidden_state[0].numpy()
        pos_enc = positional_encoding(len(tokens), embeddings.shape[1])
        attentions = outputs.attentions
        hidden_states = outputs.hidden_states  # Todos os hidden states das camadas
        return tokens, embeddings, pos_enc, attentions, hidden_states

    # Show the sentence
    @output
    @render.text
    def preview_text():
        t = input.text_input().strip()
        return f"🔹 \"{t}\"" if t else "Type a sentence and click Generate All."

    # Tabela de Embeddings
    @output
    @render.ui
    def embedding_table():
        result = compute_all()
        if not result:
            return ui.HTML("<p>Click <b>Generate All</b> to visualize.</p>")
        tokens, embeddings, pos_enc, attentions, all_hidden_states = result
        
        rows = []
        for i, tok in enumerate(tokens):
            emb = embeddings[i]
            emb_display = emb[:50]
            emb_img = array_to_base64_img(emb_display, "Blues")
            
            # Tooltip com TODOS os valores
            emb_values = ", ".join([f"{v:.3f}" for v in emb])
            tooltip = f"Embedding values (768 dims): [{emb_values}]"
            
            rows.append(
                f"<tr><td class='token-name'>{tok}</td>"
                f"<td><img class='heatmap' src='data:image/png;base64,{emb_img}' "
                f"title='{tooltip}' height='18'></td></tr>"
            )
        html = (
            "<div class='card-scroll'>"
            "<table class='token-table'>"
            "<tr><th>Token</th><th>Embedding (768 dims)</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )
        return ui.HTML(html)
    
    # Tabela de Positional Encodings
    @output
    @render.ui
    def posenc_table():
        result = compute_all()
        if not result:
            return ui.HTML("<p>Click <b>Generate All</b> to visualize.</p>")
        tokens, embeddings, pos_enc, attentions, all_hidden_states = result
        
        rows = []
        for i, tok in enumerate(tokens):
            pe = pos_enc[i]
            pe_display = pe[:50]
            pe_img = array_to_base64_img(pe_display, "RdBu")
            
            # Tooltip com TODOS os valores
            pe_values = ", ".join([f"{v:.3f}" for v in pe])
            tooltip = f"Position {i} encoding (768 dims): [{pe_values}]"
            
            rows.append(
                f"<tr><td class='token-name'>{tok}</td>"
                f"<td><img class='heatmap' src='data:image/png;base64,{pe_img}' "
                f"title='{tooltip}' height='18'></td></tr>"
            )
        html = (
            "<div class='card-scroll'>"
            "<table class='token-table'>"
            "<tr><th>Token</th><th>Pos Encoding (768 dims)</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )
        return ui.HTML(html)
    
    # Tabela de Q, K, V
    @output
    @render.ui
    def qkv_table():
        result = compute_all()
        if not result:
            return ui.HTML("<p>Click <b>Generate All</b> to visualize.</p>")
        tokens, embeddings, pos_enc, attentions, all_hidden_states = result
        
        # Obter layer selecionado
        layer = int(input.embedding_layer_select())
        layer_module = model.encoder.layer[layer].attention.self
        
        # Usar o hidden state apropriado como input
        layer_input = all_hidden_states[layer]  # Input para essa camada
        
        # Calcular Q, K, V para essa camada
        Q = layer_module.query(layer_input)[0].detach().numpy()
        K = layer_module.key(layer_input)[0].detach().numpy()
        V = layer_module.value(layer_input)[0].detach().numpy()
        
        rows = []
        for i, tok in enumerate(tokens):
            q, k, v = Q[i], K[i], V[i]
            q_display, k_display, v_display = q[:40], k[:40], v[:40]
            
            q_img = array_to_base64_img(q_display, "Greens", height=0.15)
            k_img = array_to_base64_img(k_display, "Oranges", height=0.15)
            v_img = array_to_base64_img(v_display, "Purples", height=0.15)
            
            # Tooltips detalhados com todos os valores
            q_values = ", ".join([f"{val:.3f}" for val in q])
            k_values = ", ".join([f"{val:.3f}" for val in k])
            v_values = ", ".join([f"{val:.3f}" for val in v])
            
            q_tooltip = f"Query: [{q_values}]"
            k_tooltip = f"Key: [{k_values}]"
            v_tooltip = f"Value: [{v_values}]"
            
            rows.append(
                f"<tr><td class='token-name'>{tok}</td>"
                f"<td style='padding:2px 4px;'>"
                f"<div style='margin-bottom:2px;'><small style='color:#2d7a3d;font-weight:600;'>Q</small></div>"
                f"<img class='heatmap' src='data:image/png;base64,{q_img}' "
                f"title='{q_tooltip}' height='12'>"
                f"<div style='margin:2px 0;'><small style='color:#d97706;font-weight:600;'>K</small></div>"
                f"<img class='heatmap' src='data:image/png;base64,{k_img}' "
                f"title='{k_tooltip}' height='12'>"
                f"<div style='margin:2px 0;'><small style='color:#7c3aed;font-weight:600;'>V</small></div>"
                f"<img class='heatmap' src='data:image/png;base64,{v_img}' "
                f"title='{v_tooltip}' height='12'>"
                f"</td></tr>"
            )
        html = (
            "<div class='card-scroll'>"
            "<table class='token-table-compact'>"
            "<tr><th>Token</th><th>Vectors</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )
        return ui.HTML(html)

    # Multi-Head Attention Heatmap
    @output
    @render_plotly
    def attention_map():
        result = compute_all()
        if not result:
            return None
        tokens, _, _, attentions, all_hidden_states = result
        layer = int(input.layer_select())
        head = int(input.head_select())

        att = attentions[layer][0][head].detach().numpy()
        layer_module = model.encoder.layer[layer].attention.self
        
        # Usar o hidden state apropriado
        layer_input = all_hidden_states[layer]
        Q = layer_module.query(layer_input)[0].detach().numpy()
        K = layer_module.key(layer_input)[0].detach().numpy()
        
        # Calcular dimensão por head
        d_k = Q.shape[-1] // 12  # 768 / 12 = 64
        
        # Preparar customdata com os cálculos
        custom = np.empty((len(tokens), len(tokens), 5), dtype=object)
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                # Calcular dot product
                dot_product = np.dot(Q[i], K[j])
                scaled = dot_product / np.sqrt(d_k)
                attention_weight = att[i, j]
                
                # Valores de Q e K para mostrar
                q_vals = np.array2string(Q[i][:5], precision=3, separator=", ")
                k_vals = np.array2string(K[j][:5], precision=3, separator=", ")
                
                custom[i, j, 0] = q_vals
                custom[i, j, 1] = k_vals
                custom[i, j, 2] = f"{dot_product:.4f}"
                custom[i, j, 3] = f"{scaled:.4f}"
                custom[i, j, 4] = f"{attention_weight:.4f}"

        hover = (
            "<b>Query Token:</b> %{y}<br>"
            "<b>Key Token:</b> %{x}<br>"
            "<br><b>📊 Calculation:</b><br>"
            "1️⃣ Dot Product: Q·K = %{customdata[2]}<br>"
            "2️⃣ Scaled: (Q·K)/√d_k = %{customdata[3]}<br>"
            "3️⃣ Softmax Result: <b>%{customdata[4]}</b><br>"
            "<br><b>🔍 Vectors (first 5 dims):</b><br>"
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
            title=f"Layer {layer} | Head {head} | Attention Weights",
        )
        fig.update_traces(customdata=custom, hovertemplate=hover)
        fig.update_layout(
            xaxis_title="Key (attending to)",
            yaxis_title="Query (attending from)",
            coloraxis_colorbar=dict(title="Attention Weight"),
            plot_bgcolor="#111111",
            paper_bgcolor="#111111",
            font=dict(color="white"),
        )
        return fig

    # Add & Norm 1 (Post-Attention)
    @output
    @render.ui
    def add_norm_1_table():
        result = compute_all()
        if not result:
            return ui.HTML("<p>Click <b>Generate All</b> to visualize.</p>")
        tokens, embeddings, pos_enc, attentions, all_hidden_states = result
        
        layer = int(input.layer_select())
        
        # O hidden_states[layer] é após a primeira add & norm (post-attention)
        # hidden_states[0] = embeddings
        # hidden_states[1] = após primeira camada completa
        # Vamos mostrar o estado intermediário após attention
        layer_obj = model.encoder.layer[layer]
        
        # Usar hidden_states da camada selecionada
        layer_output = all_hidden_states[layer + 1][0].numpy()
        
        rows = []
        for i, tok in enumerate(tokens):
            output = layer_output[i]
            output_display = output[:50]
            output_img = array_to_base64_img(output_display, "Viridis")
            
            output_values = ", ".join([f"{v:.3f}" for v in output])
            tooltip = f"Layer {layer} output (768 dims): [{output_values}]"
            
            rows.append(
                f"<tr><td class='token-name'>{tok}</td>"
                f"<td><img class='heatmap' src='data:image/png;base64,{output_img}' "
                f"title='{tooltip}' height='18'></td></tr>"
            )
        
        html = (
            "<div class='card-scroll'>"
            "<table class='token-table'>"
            "<tr><th>Token</th><th>After Attention + Residual + Norm</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )
        return ui.HTML(html)
    
    # Feed Forward Network
    @output
    @render.ui
    def feed_forward_table():
        result = compute_all()
        if not result:
            return ui.HTML("<p>Click <b>Generate All</b> to visualize.</p>")
        tokens, embeddings, pos_enc, attentions, all_hidden_states = result
        
        layer = int(input.layer_select())
        layer_obj = model.encoder.layer[layer]
        
        # Calcular FFN intermediário
        with torch.no_grad():
            layer_input = all_hidden_states[layer + 1]  # Input para FFN
            
            # FFN: intermediate (expand) -> activation
            ffn_intermediate = layer_obj.intermediate(layer_input)
            ffn_output = ffn_intermediate[0].numpy()
        
        rows = []
        for i, tok in enumerate(tokens):
            output = ffn_output[i]
            # FFN expande de 768 para 3072, mostrar amostra
            output_display = output[:100]
            output_img = array_to_base64_img(output_display, "Plasma")
            
            output_values = ", ".join([f"{v:.3f}" for v in output[:50]])
            tooltip = f"FFN intermediate (3072 dims, showing first 50): [{output_values}]"
            
            rows.append(
                f"<tr><td class='token-name'>{tok}</td>"
                f"<td><img class='heatmap' src='data:image/png;base64,{output_img}' "
                f"title='{tooltip}' height='18'></td></tr>"
            )
        
        html = (
            "<div class='card-scroll'>"
            "<table class='token-table'>"
            "<tr><th>Token</th><th>Linear(768→3072) + GELU</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )
        return ui.HTML(html)
    
    # Add & Norm 2 (Post-FFN)
    @output
    @render.ui
    def add_norm_2_table():
        result = compute_all()
        if not result:
            return ui.HTML("<p>Click <b>Generate All</b> to visualize.</p>")
        tokens, embeddings, pos_enc, attentions, all_hidden_states = result
        
        layer = int(input.layer_select())
        
        # hidden_states[layer+1] é o output final da camada (após todo o bloco transformer)
        final_output = all_hidden_states[layer + 1][0].numpy()
        
        rows = []
        for i, tok in enumerate(tokens):
            output = final_output[i]
            output_display = output[:50]
            output_img = array_to_base64_img(output_display, "Cividis")
            
            output_values = ", ".join([f"{v:.3f}" for v in output])
            tooltip = f"Final layer {layer} output (768 dims): [{output_values}]"
            
            rows.append(
                f"<tr><td class='token-name'>{tok}</td>"
                f"<td><img class='heatmap' src='data:image/png;base64,{output_img}' "
                f"title='{tooltip}' height='18'></td></tr>"
            )
        
        html = (
            "<div class='card-scroll'>"
            "<table class='token-table'>"
            "<tr><th>Token</th><th>After FFN + Residual + Norm</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )
        return ui.HTML(html)


# ===============================
# 5. Create App
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()