"""Auditor Notebook UI for Attention Atlas.

This module renders the Notebook as a **floating drawer** rather than as
a top-level navbar section. A circular hamburger-style FAB (floating
action button) is anchored to the bottom-right of every page; clicking
it slides a panel in from the right with the five-field entry form and
the chronological list of saved entries.

Each entry follows the construct introduced in Chapter 6 of the thesis:

    hypothesis, conditions tested, signals observed, uncertainty
    acknowledged, next steps.

The reactive logic lives in
``attention_app/server/notebook_handlers.py``; this module only emits
the static layout and the input ids.
"""

from shiny import ui


NOTEBOOK_CSS = """
<style>
/* ── Floating action button (3-line hamburger) ────────────────── */
.nb-fab {
    position: fixed;
    top: 28px;
    right: 56px;
    width: 42px;
    height: 42px;
    border-radius: 50%;
    background: #ff5ca9;
    border: none;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 4px;
    box-shadow: none;
    z-index: 9998;
    transition: transform 0.18s ease, background 0.18s ease, box-shadow 0.22s ease;
    padding: 0;
}
.nb-fab:hover {
    transform: translateY(-1px);
    background: #ff74b8;
    box-shadow: 0 10px 22px rgba(255, 92, 169, 0.40),
                0 3px 6px rgba(255, 92, 169, 0.22);
}
.nb-fab:focus-visible {
    outline: 2px solid #ff74b8;
    outline-offset: 3px;
}
.nb-fab-line {
    width: 17px;
    height: 2px;
    background: white;
    border-radius: 2px;
}

/* ── Backdrop overlay ─────────────────────────────────────────── */
.nb-drawer-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(15, 23, 42, 0.45);
    z-index: 9998;
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.25s ease, visibility 0.25s ease;
}
.nb-drawer-backdrop.nb-open {
    opacity: 1;
    visibility: visible;
}

/* ── Slide-in drawer ──────────────────────────────────────────── */
.nb-drawer {
    position: fixed;
    top: 0;
    right: 0;
    width: min(560px, 92vw);
    height: 100vh;
    background: #f0f4f8;
    box-shadow: -12px 0 32px rgba(15, 23, 42, 0.18);
    z-index: 9999;
    transform: translateX(100%);
    transition: transform 0.32s cubic-bezier(0.32, 0.72, 0, 1);
    display: flex;
    flex-direction: column;
    font-family: 'Inter', -apple-system, sans-serif;
    color: #1e293b;
}
.nb-drawer.nb-open { transform: translateX(0); }
.nb-drawer-header {
    padding: 22px 24px 18px 24px;
    border-bottom: 1px solid #e2e8f0;
    background: #ffffff;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-shrink: 0;
    position: relative;
}
.nb-drawer-header::after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    bottom: -1px;
    height: 2px;
    background: linear-gradient(90deg, #ff5ca9 0%, #ff74b8 60%, transparent 100%);
    opacity: 0.85;
}
.nb-drawer-title {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #ff5ca9;
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
}
.nb-drawer-subtitle {
    color: #64748b;
    font-size: 12px;
    line-height: 1.45;
    margin: 0;
    max-width: 420px;
}
.nb-drawer-close {
    background: none;
    border: none;
    font-size: 22px;
    color: #64748b;
    cursor: pointer;
    line-height: 1;
    padding: 4px 10px;
    border-radius: 6px;
    transition: background 0.15s, color 0.15s;
    margin-left: 12px;
}
.nb-drawer-close:hover {
    background: #fff0f8;
    color: #ff5ca9;
}
.nb-drawer-body {
    padding: 20px 24px 28px 24px;
    overflow-y: auto;
    flex: 1;
}

/* ── Cards inside the drawer (one per section) ───────────────── */
.nb-section {
    margin-bottom: 18px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 18px 20px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
}
.nb-section-head {
    display: flex !important;
    flex-direction: row;
    flex-wrap: nowrap;
    align-items: center;
    gap: 8px;
    margin: 0 0 4px 0;
}
.nb-section-head h3 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 12px;
    font-weight: 600;
    color: #1e293b;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin: 0;
    line-height: 1;
    flex: 0 0 auto;
}
.nb-section-head > .shiny-html-output {
    display: inline-flex !important;
    align-items: center !important;
    line-height: 1 !important;
    margin: 0 !important;
    height: auto !important;
    width: auto !important;
    flex: 0 0 auto;
}
.nb-section > h3 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 12px;
    font-weight: 600;
    color: #1e293b;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin: 0 0 4px 0;
    line-height: 1;
}
.nb-count {
    background: #fff0f8;
    color: #ff5ca9;
    font-size: 10.5px;
    padding: 2px 8px;
    border-radius: 9px;
    font-weight: 600;
    letter-spacing: 0.3px;
    line-height: 1.4;
    display: inline-flex;
    align-items: center;
}
.nb-section p.nb-section-sub {
    font-size: 11.5px;
    color: #94a3b8;
    margin: 6px 0 14px 0;
    line-height: 1.45;
}

/* ── Form fields ──────────────────────────────────────────────── */
.nb-field { margin-bottom: 11px; }
.nb-field label {
    display: block;
    font-size: 10.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #475569;
    margin-bottom: 3px;
}
.nb-field input[type="text"],
.nb-field textarea {
    width: 100%;
    padding: 8px 10px;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    font-family: 'Inter', sans-serif;
    font-size: 12.5px;
    line-height: 1.5;
    color: #1e293b;
    background: #f0f4f8;
    transition: border-color 0.15s, background 0.15s, box-shadow 0.15s;
    resize: vertical;
}
.nb-field input[type="text"]:focus,
.nb-field textarea:focus {
    outline: none;
    border-color: #ff5ca9;
    background: white;
    box-shadow: 0 0 0 3px rgba(255, 92, 169, 0.12);
}
.nb-field .nb-hint {
    display: block;
    font-size: 10.5px;
    color: #94a3b8;
    margin-top: 2px;
    font-style: italic;
}

/* ── Buttons ──────────────────────────────────────────────────── */
.nb-actions, .nb-export-row {
    display: flex;
    gap: 8px;
    margin: 14px 0 4px 0;
    flex-wrap: wrap;
}
.nb-btn {
    padding: 7px 14px;
    border-radius: 9999px;
    font-size: 12.5px;
    font-weight: 600;
    border: 1.5px solid transparent;
    cursor: pointer;
    font-family: 'Inter', sans-serif;
    transition: background 0.15s, border-color 0.15s, color 0.15s;
}
.nb-btn-primary {
    background: #ff5ca9;
    color: white;
    border-color: #ff5ca9;
}
.nb-btn-primary:hover { background: #ff74b8; border-color: #ff74b8; }
.nb-btn-secondary {
    background: transparent;
    color: #ff5ca9;
    border-color: #ff5ca9;
}
.nb-btn-secondary:hover { background: #fff0f8; }

/* ── Status banner ────────────────────────────────────────────── */
.nb-status {
    margin-top: 8px;
    font-size: 11.5px;
    color: #16a34a;
    min-height: 14px;
}
.nb-status.nb-status-error { color: #dc2626; }

/* ── Entries list ─────────────────────────────────────────────── */
.nb-empty {
    text-align: center;
    padding: 28px 16px;
    color: #94a3b8;
    font-size: 12px;
    font-style: italic;
    border: 1px dashed #e2e8f0;
    border-radius: 8px;
    background: #f0f4f8;
}
.nb-entry {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 14px 16px;
    background: #f0f4f8;
    margin-bottom: 10px;
}
.nb-entry-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 8px;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 6px;
    gap: 10px;
}
.nb-entry-title {
    font-size: 13px;
    font-weight: 600;
    color: #1e293b;
    flex: 1;
    min-width: 0;
    word-break: break-word;
}
.nb-entry-meta {
    font-size: 10.5px;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
    white-space: nowrap;
    display: flex;
    align-items: center;
    gap: 8px;
}
.nb-entry-field { margin: 6px 0; }
.nb-entry-field-label {
    font-size: 9.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #ff5ca9;
    margin-bottom: 1px;
    display: block;
}
.nb-entry-field-value {
    font-size: 12px;
    color: #334155;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
}
.nb-entry-delete {
    background: none;
    border: none;
    color: #94a3b8;
    font-size: 11px;
    cursor: pointer;
    padding: 2px 6px;
    border-radius: 4px;
    transition: background 0.15s, color 0.15s;
}
.nb-entry-delete:hover {
    background: #fee2e2;
    color: #dc2626;
}
</style>
"""

NOTEBOOK_JS = """
<script>
(function() {
    function bindNotebookDrawer() {
        var fab = document.getElementById('nb-fab');
        var drawer = document.getElementById('nb-drawer');
        var backdrop = document.getElementById('nb-drawer-backdrop');
        var closeBtn = document.getElementById('nb-drawer-close');
        if (!fab || !drawer || !backdrop || !closeBtn) {
            // Try again shortly: Shiny may not have inserted the DOM yet.
            setTimeout(bindNotebookDrawer, 120);
            return;
        }
        if (fab.dataset.bound === '1') return;
        fab.dataset.bound = '1';
        function openDrawer() {
            drawer.classList.add('nb-open');
            backdrop.classList.add('nb-open');
            document.body.style.overflow = 'hidden';
        }
        function closeDrawer() {
            drawer.classList.remove('nb-open');
            backdrop.classList.remove('nb-open');
            document.body.style.overflow = '';
        }
        fab.addEventListener('click', openDrawer);
        closeBtn.addEventListener('click', closeDrawer);
        backdrop.addEventListener('click', closeDrawer);
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && drawer.classList.contains('nb-open')) {
                closeDrawer();
            }
        });
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', bindNotebookDrawer);
    } else {
        bindNotebookDrawer();
    }
})();
</script>
"""


def _field(input_id, label, hint, kind="textarea", rows=3, placeholder=""):
    if kind == "text":
        control = ui.input_text(input_id, label=None, placeholder=placeholder)
    else:
        control = ui.input_text_area(
            input_id,
            label=None,
            placeholder=placeholder,
            rows=rows,
            width="100%",
            autoresize=False,
        )
    return ui.tags.div(
        ui.tags.label(label, **{"for": input_id}),
        control,
        ui.tags.span(hint, class_="nb-hint"),
        class_="nb-field",
    )


def create_notebook_drawer():
    """Return the FAB + slide-in drawer as a single body-level component.

    Intended to be passed to ``ui.page_navbar(..., footer=...)`` so that
    it is mounted once globally and accessible from every navbar tab.
    """
    return ui.tags.div(
        ui.HTML(NOTEBOOK_CSS),
        # ── FAB (3-line hamburger) ───────────────────────────────
        ui.tags.button(
            ui.tags.span(class_="nb-fab-line"),
            ui.tags.span(class_="nb-fab-line"),
            ui.tags.span(class_="nb-fab-line"),
            id="nb-fab",
            class_="nb-fab",
            type="button",
            **{"aria-label": "Open Auditor Notebook", "title": "Auditor Notebook"},
        ),
        # ── Backdrop ─────────────────────────────────────────────
        ui.tags.div(id="nb-drawer-backdrop", class_="nb-drawer-backdrop"),
        # ── Drawer ───────────────────────────────────────────────
        ui.tags.div(
            ui.tags.div(
                ui.tags.div(
                    ui.tags.h2("Auditor Notebook", class_="nb-drawer-title"),
                    ui.tags.p(
                        "Record one analytical move per entry. Stored in this "
                        "session and persisted to disk; export when you finish.",
                        class_="nb-drawer-subtitle",
                    ),
                ),
                ui.tags.button(
                    "×",
                    id="nb-drawer-close",
                    class_="nb-drawer-close",
                    type="button",
                    **{"aria-label": "Close"},
                ),
                class_="nb-drawer-header",
            ),
            ui.tags.div(
                # ── New-entry form ───────────────────────────────
                ui.tags.div(
                    ui.tags.h3("New entry"),
                    ui.tags.p(
                        "All fields except the title are required.",
                        class_="nb-section-sub",
                    ),
                    _field(
                        "nb_title",
                        "Title (optional)",
                        "A short label so the entry is easy to find later.",
                        kind="text",
                        placeholder="e.g. Gender swap on Winogender sentence 23",
                    ),
                    _field(
                        "nb_hypothesis",
                        "Hypothesis",
                        "What you expect the model to do, and why.",
                        kind="textarea",
                        rows=3,
                        placeholder="e.g. Head L5H3 will attend more to the gendered "
                        "pronoun on the stereotypical variant than on the counterfactual.",
                    ),
                    _field(
                        "nb_conditions",
                        "Conditions tested",
                        "Model, prompt(s), comparison setup, layers/heads selected.",
                        kind="textarea",
                        rows=3,
                        placeholder="e.g. BERT-base, compare-mode A=original, "
                        "B=counterfactual, layer 5, head 3, default temperature.",
                    ),
                    _field(
                        "nb_signals",
                        "Signals observed",
                        "What the dashboard showed: attention, IG, ablation, perturbation.",
                        kind="textarea",
                        rows=4,
                        placeholder="e.g. Attention on the pronoun is 0.42 in A and "
                        "0.39 in B; IG agrees in direction; head ablation drops the "
                        "bias-class probability by 7 pp on A only.",
                    ),
                    _field(
                        "nb_uncertainty",
                        "Uncertainty acknowledged",
                        "What this evidence cannot decide, and what could overturn it.",
                        kind="textarea",
                        rows=3,
                        placeholder="e.g. The ablation effect is within seed variance "
                        "for B; cross-validation with LRP not yet checked.",
                    ),
                    _field(
                        "nb_next_steps",
                        "Next steps",
                        "Concrete follow-ups: more seeds, other heads, other prompts.",
                        kind="textarea",
                        rows=2,
                        placeholder="e.g. Repeat with three more seeds and add an LRP cross-check.",
                    ),
                    ui.tags.div(
                        ui.input_action_button(
                            "nb_add",
                            "Add entry",
                            class_="nb-btn nb-btn-primary",
                        ),
                        ui.input_action_button(
                            "nb_clear",
                            "Clear form",
                            class_="nb-btn nb-btn-secondary",
                        ),
                        class_="nb-actions",
                    ),
                    ui.output_ui("nb_status"),
                    class_="nb-section",
                ),
                # ── Entries list ─────────────────────────────────
                ui.tags.div(
                    ui.tags.div(
                        ui.tags.h3("Entries"),
                        ui.output_ui("nb_count", inline=True),
                        class_="nb-section-head",
                    ),
                    ui.tags.p(
                        "Persisted to downloads/sessions/auditor_notebook.json.",
                        class_="nb-section-sub",
                    ),
                    ui.tags.div(
                        ui.download_button(
                            "nb_download_md",
                            "Export Markdown",
                            class_="nb-btn nb-btn-secondary",
                        ),
                        ui.download_button(
                            "nb_download_json",
                            "Export JSON",
                            class_="nb-btn nb-btn-secondary",
                        ),
                        ui.input_action_button(
                            "nb_clear_all",
                            "Clear all",
                            class_="nb-btn nb-btn-secondary",
                        ),
                        class_="nb-export-row",
                    ),
                    ui.output_ui("nb_entries"),
                    class_="nb-section",
                ),
                class_="nb-drawer-body",
            ),
            id="nb-drawer",
            class_="nb-drawer",
            role="dialog",
            **{"aria-modal": "true", "aria-labelledby": "nb-drawer-title"},
        ),
        ui.HTML(NOTEBOOK_JS),
    )
