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

from .components import AUDIT_ICON_DATA_URL


NOTEBOOK_CSS = """
<style>
/* ── Floating action button (3-line hamburger) ────────────────── */
/* Hidden by default; revealed once Attention or Bias has been run.   */
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
    display: none;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 4px;
    box-shadow: none;
    z-index: 9998;
    transition: transform 0.18s ease, background 0.18s ease, box-shadow 0.22s ease,
                opacity 0.25s ease;
    padding: 0;
    opacity: 0;
}
.nb-fab.nb-fab-visible {
    display: flex;
    opacity: 1;
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
.nb-fab-icon {
    width: 34px;
    height: 34px;
    display: block;
    object-fit: contain;
    pointer-events: none;
}

/* ── Unexported-entries warning icon (left of the FAB) ────────── */
/* A bare pink triangle, no circle behind it, breathing gently.     */
/* Shown only after something has been typed in the notebook this   */
/* session AND it holds entries that have not been exported yet.    */
.nb-export-warn {
    position: fixed;
    top: 28px;
    right: 108px;
    width: 42px;
    height: 42px;
    background: none;
    color: #ff5ca9;
    border: none;
    display: none;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    z-index: 9998;
    padding: 0;
    transition: transform 0.18s ease, color 0.18s ease;
}
.nb-export-warn.nb-warn-visible {
    display: flex;
    animation: nb-warn-pulse 2s ease-in-out infinite;
}
.nb-export-warn:hover {
    color: #ff74b8;
    transform: translateY(-1px);
    animation: none;
}
.nb-export-warn:focus-visible {
    outline: 2px solid #ff74b8;
    outline-offset: 3px;
    border-radius: 8px;
}
.nb-export-warn svg { pointer-events: none; }
@keyframes nb-warn-pulse {
    0%   { transform: scale(1);    opacity: 1; }
    50%  { transform: scale(1.12); opacity: 0.75; }
    100% { transform: scale(1);    opacity: 1; }
}

/* Click-to-explain popover under the icon */
.nb-export-warn-pop {
    position: fixed;
    top: 80px;
    right: 100px;
    width: min(300px, calc(100vw - 112px));
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    box-shadow: 0 16px 40px rgba(15, 23, 42, 0.16);
    padding: 14px 16px;
    display: none;
    z-index: 9998;
    font-family: 'Inter', -apple-system, sans-serif;
}
.nb-export-warn-pop.nb-warn-open { display: block; }
.nb-export-warn-pop-title {
    font-size: 13px;
    font-weight: 700;
    color: #1e293b;
    margin: 0 0 6px 0;
    display: flex;
    align-items: center;
    gap: 7px;
}
.nb-export-warn-pop-title svg {
    color: #ff5ca9;
    flex-shrink: 0;
}
.nb-export-warn-pop-msg {
    font-size: 12.5px;
    line-height: 1.5;
    color: #475569;
    margin: 0 0 12px 0;
}
.nb-export-warn-open-btn {
    width: 100%;
    background: #ff5ca9;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12.5px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.18s ease;
}
.nb-export-warn-open-btn:hover { background: #ff74b8; }

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
    /* Pin to top AND bottom (like .nb-drawer-backdrop's inset:0) instead of
       using height:100vh. Under the responsive `body { zoom }` hack for short
       screens, a 100vh element is computed against the unzoomed viewport and
       then scaled down, so it stops short of the bottom (visible on Hugging
       Face, whose iframe is shorter and triggers the zoom). Edge-pinning fills
       correctly regardless of the zoom factor. */
    top: 0;
    bottom: 0;
    right: 0;
    width: min(560px, 92vw);
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
/* Transient highlight when the export row is jumped to from the
   unexported-entries warning, so the eye lands on the right buttons. */
.nb-export-row {
    border-radius: 10px;
    padding: 4px;
    margin-left: -4px;
    margin-right: -4px;
    transition: background 0.4s ease, box-shadow 0.4s ease;
}
.nb-export-row.nb-export-flash {
    background: rgba(255, 92, 169, 0.12);
    box-shadow: 0 0 0 2px rgba(255, 92, 169, 0.35);
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
/* Clearing destroys all three local copies at once - the file on disk, the
   browser backup, and the unexported-entries reminder - so it is coloured
   apart from the exports it sits beside. The spacing is deliberately left
   even: the confirmation step below is what guards the action, and pushing
   the button to the far end only bought a few pixels in a row this width.
   Filled rather than outlined, so it reads as a different kind of control
   from the three outlined exports beside it and not as a fourth one. The red
   is kept in the rose family of the interface pink, so it separates by weight
   and not by clashing. */
.nb-btn-danger {
    background: #e11d48;
    color: #ffffff;
    border-color: #e11d48;
}
.nb-btn-danger:hover { background: #be123c; border-color: #be123c; }

/* ── Clear confirmation ───────────────────────────────────────── */
.nb-confirm {
    margin: 10px 0 4px 0;
    padding: 10px 12px;
    border-radius: 8px;
    background: #fff1f2;
    border: 1px solid #fda4af;
    border-left: 3px solid #e11d48;
    font-size: 12px;
    color: #881337;
    line-height: 1.5;
}
.nb-confirm p { margin: 0 0 8px 0; }
.nb-confirm-actions { display: flex; gap: 8px; flex-wrap: wrap; }
/* One step darker than the button that opened this box, so the two reds read
   as the same action escalating rather than as two unrelated warnings. */
.nb-btn-danger-solid {
    background: #be123c;
    color: #ffffff;
    border-color: #be123c;
}
.nb-btn-danger-solid:hover { background: #9f1239; border-color: #9f1239; }

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
    display: inline-flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
}
.nb-case-chip {
    display: inline-flex;
    align-items: center;
    background: #fff0f8;
    color: #ff5ca9;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 9999px;
    border: 1px solid #ffd5e7;
    letter-spacing: 0.2px;
    font-family: 'JetBrains Mono', monospace;
    text-transform: lowercase;
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
.nb-entry-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid #e2e8f0;
}
.nb-entry-restore,
.nb-entry-delete {
    background: none;
    border: none;
    font-size: 11px;
    cursor: pointer;
    padding: 4px 8px;
    border-radius: 4px;
    font-family: 'Inter', sans-serif;
    transition: background 0.15s, color 0.15s;
}
.nb-entry-restore {
    color: #ff5ca9;
    border: 1px solid #ff5ca9;
    font-weight: 600;
}
.nb-entry-restore:hover { background: #fff0f8; }
.nb-entry-delete { color: #94a3b8; }
.nb-entry-delete:hover { background: #fee2e2; color: #dc2626; }

/* ── Captured-context block ─────────────────────────────────────── */
.nb-ctx-empty {
    font-size: 11.5px;
    color: #94a3b8;
    font-style: italic;
    padding: 12px 14px;
    background: #f0f4f8;
    border-radius: 6px;
    border: 1px dashed #e2e8f0;
}
.nb-ctx-block {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 10px 12px;
    display: grid;
    grid-template-columns: minmax(140px, max-content) 1fr;
    gap: 4px 14px;
    font-size: 11.5px;
    line-height: 1.45;
}
.nb-ctx-block-entry {
    margin-top: 2px;
    background: #ffffff;
}
.nb-ctx-subsection {
    margin-top: 8px;
}
.nb-ctx-subsection:first-child {
    margin-top: 0;
}
.nb-ctx-subtitle {
    color: #334155;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.35px;
    margin: 0 0 4px 0;
    text-transform: uppercase;
}
.nb-ctx-empty-inline {
    color: #94a3b8;
    font-size: 11px;
    font-style: italic;
}
.nb-ctx-row {
    display: contents;
}
.nb-ctx-key {
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 10px;
    letter-spacing: 0.4px;
    padding-top: 2px;
    white-space: nowrap;
}
.nb-ctx-val {
    color: #1e293b;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    word-break: break-word;
}
.nb-ctx-preview-wrapper {
    margin: 8px 0 14px 0;
}
.nb-ctx-preview-wrapper > .nb-ctx-preview-title {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 10.5px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 5px;
}
.nb-ctx-preview-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #16a34a;
    box-shadow: 0 0 0 3px rgba(22, 163, 74, 0.15);
}

/* ── Regulatory-anchor (i) link beside captured-field labels ─── */
.nb-reg-link {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    height: 14px;
    margin-left: 4px;
    color: #94a3b8;
    text-decoration: none;
    font-size: 11px;
    font-weight: 600;
    vertical-align: middle;
    transition: color 0.15s, transform 0.15s;
    cursor: help;
}
.nb-reg-link:hover {
    color: #ff5ca9;
    transform: scale(1.15);
}
.nb-reg-link:focus-visible {
    outline: 2px solid #ff74b8;
    outline-offset: 2px;
    border-radius: 50%;
}

/* ── Disconfirming-evidence row + banner (DR7) ──────────────── */
.nb-ctx-row-warning .nb-ctx-key {
    color: #b45309;
}
.nb-ctx-row-warning .nb-ctx-val {
    color: #b45309;
    background: rgba(245, 158, 11, 0.08);
    border-left: 2px solid #f59e0b;
    padding-left: 6px;
    margin-left: -6px;
    border-radius: 2px;
}
.nb-ctx-warning-banner {
    background: rgba(245, 158, 11, 0.10);
    border: 1px solid rgba(245, 158, 11, 0.30);
    color: #92400e;
    font-size: 11.5px;
    font-weight: 500;
    border-radius: 6px;
    padding: 7px 10px;
    margin: 0 0 8px 0;
    line-height: 1.4;
}
</style>
"""

NOTEBOOK_JS = """
<script>
(function() {
    // The export-warning icon appears only once BOTH are true: the user
    // has typed something in the notebook this session, and there are
    // entries that have not been exported yet.
    function updateWarnBadge() {
        var badge = document.getElementById('nb-export-warn');
        var pop = document.getElementById('nb-export-warn-pop');
        if (!badge) return;
        var show = (window._nbUnexported || 0) > 0 && window._nbNotebookTouched;
        if (show) {
            badge.classList.add('nb-warn-visible');
        } else {
            badge.classList.remove('nb-warn-visible');
            if (pop) pop.classList.remove('nb-warn-open');
        }
    }

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
        // Any typing inside the drawer (the five entry fields, the
        // participant code, ...) counts as "the notebook is being used";
        // only then may the export warning appear.
        drawer.addEventListener('input', function() {
            if (window._nbNotebookTouched) return;
            window._nbNotebookTouched = true;
            updateWarnBadge();
        });
        function closeDrawer() {
            drawer.classList.remove('nb-open');
            backdrop.classList.remove('nb-open');
            document.body.style.overflow = '';
            if (window.Shiny && Shiny.setInputValue) {
                Shiny.setInputValue('nb_dismiss_status', Date.now(), {priority: 'event'});
            }
        }
        fab.addEventListener('click', openDrawer);
        closeBtn.addEventListener('click', closeDrawer);
        backdrop.addEventListener('click', closeDrawer);
        // Exposed so the server can minimise the drawer after "Restore
        // state": the restored run plays out on the dashboard behind it, so
        // staying in the notebook hides the very evidence being restored.
        window._nbCloseDrawer = closeDrawer;

        // ── Unexported-entries warning badge ──
        var warnBadge = document.getElementById('nb-export-warn');
        var warnPop = document.getElementById('nb-export-warn-pop');
        var warnOpenBtn = document.getElementById('nb-export-warn-open');
        if (warnBadge && warnPop) {
            warnBadge.addEventListener('click', function(e) {
                e.stopPropagation();
                warnPop.classList.toggle('nb-warn-open');
            });
            document.addEventListener('click', function(e) {
                if (!warnPop.classList.contains('nb-warn-open')) return;
                if (warnPop.contains(e.target) || warnBadge.contains(e.target)) return;
                warnPop.classList.remove('nb-warn-open');
            });
            if (warnOpenBtn) {
                warnOpenBtn.addEventListener('click', function() {
                    warnPop.classList.remove('nb-warn-open');
                    openDrawer();
                    // Land straight on the export buttons rather than at the
                    // top of the form: the warning is about exporting.
                    var row = document.getElementById('nb-export-row');
                    if (!row) return;
                    setTimeout(function() {
                        row.scrollIntoView({behavior: 'smooth', block: 'center'});
                        row.classList.add('nb-export-flash');
                        setTimeout(function() {
                            row.classList.remove('nb-export-flash');
                        }, 1600);
                    }, 340);  // after the drawer's slide-in transition
                });
            }
        }
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

    // Custom-message handler that toggles FAB visibility based on whether
    // the user has clicked Generate All / Analyze Bias yet.
    function registerFabToggle() {
        if (!window.Shiny || !Shiny.addCustomMessageHandler) {
            setTimeout(registerFabToggle, 120);
            return;
        }
        Shiny.addCustomMessageHandler('nb_fab_toggle', function(payload) {
            var fab = document.getElementById('nb-fab');
            if (!fab) return;
            if (payload && payload.visible) {
                fab.classList.add('nb-fab-visible');
            } else {
                fab.classList.remove('nb-fab-visible');
            }
        });
        Shiny.addCustomMessageHandler('nb_close_drawer', function() {
            if (window._nbCloseDrawer) {
                window._nbCloseDrawer();
            }
        });
        Shiny.addCustomMessageHandler('nb_restore_client_inputs', function(payload) {
            var values = (payload && payload.values) || {};
            Object.keys(values).forEach(function(inputId) {
                Shiny.setInputValue(inputId, values[inputId], {priority: 'event'});
            });

            // The four prompt boxes are plain <textarea> elements, not Shiny
            // text-area widgets, so update_text_area() has no binding to talk
            // to. Write the DOM value and push the input by hand instead.
            var textareas = (payload && payload.textareas) || {};
            Object.keys(textareas).forEach(function(elId) {
                var ta = document.getElementById(elId);
                if (!ta) {
                    console.warn('[Notebook] textarea not found on restore:', elId);
                    return;
                }
                ta.value = textareas[elId];
                Shiny.setInputValue(elId, ta.value, {priority: 'event'});
            });

            // Sync the bespoke toolbar widgets whose visual state lives in
            // the DOM rather than in a native Shiny widget.
            if (Object.prototype.hasOwnProperty.call(values, 'bias_attn_source') &&
                    window._setBiasAttnSource) {
                window._setBiasAttnSource(values.bias_attn_source);
            }
            if (Object.prototype.hasOwnProperty.call(values, 'bias_correction') &&
                    window._setBiasCorrection) {
                window._setBiasCorrection(values.bias_correction);
            }
            if (Object.prototype.hasOwnProperty.call(values, 'bias_alpha')) {
                var alphaSlider = document.getElementById('bias_alpha');
                if (alphaSlider) alphaSlider.value = values.bias_alpha;
                var alphaDisplay = document.getElementById('alpha_val_input');
                if (alphaDisplay) alphaDisplay.value = (+values.bias_alpha).toFixed(
                    (+values.bias_alpha) < 0.01 ? 3 : 2);
            }

            if (Object.prototype.hasOwnProperty.call(values, 'bias_selected_tokens_A')) {
                window.selectedBiasTokensA = new Set(values.bias_selected_tokens_A || []);
            }
            if (Object.prototype.hasOwnProperty.call(values, 'bias_selected_tokens_B')) {
                window.selectedBiasTokensB = new Set(values.bias_selected_tokens_B || []);
            }

            var hasBiasSelection =
                Object.prototype.hasOwnProperty.call(values, 'bias_selected_tokens_A') ||
                Object.prototype.hasOwnProperty.call(values, 'bias_selected_tokens_B');
            if (hasBiasSelection) {
                document.querySelectorAll('.bias-token-chip.selected').forEach(function(chip) {
                    chip.classList.remove('selected');
                });
                [
                    ['A', window.selectedBiasTokensA || new Set()],
                    ['B', window.selectedBiasTokensB || new Set()]
                ].forEach(function(pair) {
                    var prefix = pair[0];
                    pair[1].forEach(function(idx) {
                        document.querySelectorAll(
                            '.bias-token-chip[data-token-idx="' + idx + '"][data-prefix="' + prefix + '"]'
                        ).forEach(function(chip) {
                            chip.classList.add('selected');
                        });
                    });
                });
            }
        });

        // ── Local backup of the Notebook (user study) ──
        // The Notebook is deliberately never uploaded, so a crash or a stray
        // refresh would lose the session's primary coded artefact. Keep a
        // copy in this browser, keyed by participant: on a shared session
        // machine one participant must never restore another's entries.
        function nbBackupKey() {
            var m = /[?&](?:pid|participant|p)=([^&#]+)/.exec(window.location.search);
            return 'attention_atlas_notebook_' + (m ? decodeURIComponent(m[1]) : 'shared');
        }

        Shiny.addCustomMessageHandler('nb_backup', function(payload) {
            try {
                localStorage.setItem(
                    nbBackupKey(), JSON.stringify((payload && payload.entries) || []));
            } catch (e) {
                // Private mode or quota exceeded: the server-side copy stands.
            }
        });

        Shiny.addCustomMessageHandler('nb_unexported', function(payload) {
            window._nbUnexported = (payload && payload.n) || 0;
            var n = window._nbUnexported;
            var msgEl = document.getElementById('nb-export-warn-msg');
            if (msgEl && n > 0) {
                msgEl.textContent =
                    n + (n === 1 ? ' entry exists' : ' entries exist') +
                    ' only in this browser. If you close or refresh this ' +
                    'tab without exporting, ' +
                    (n === 1 ? 'it is' : 'they are') + ' lost.';
            }
            updateWarnBadge();
        });

        // Offer whatever this browser kept; the server ignores it unless the
        // notebook it loaded is empty, so a backup can never overwrite work.
        try {
            var saved = localStorage.getItem(nbBackupKey());
            if (saved) {
                Shiny.setInputValue('nb_restored_entries', JSON.parse(saved),
                                    {priority: 'event'});
            }
        } catch (e) { /* nothing usable stored */ }
    }
    registerFabToggle();

    // Closing with unexported entries loses them: the export is the only
    // copy that leaves this machine. Browsers show their own wording here.
    window.addEventListener('beforeunload', function(e) {
        if (!window._nbUnexported) return;
        e.preventDefault();
        e.returnValue = '';
        return '';
    });
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
        # ── FAB (notebook icon) ──────────────────────────────────
        ui.tags.button(
            ui.tags.img(
                src=AUDIT_ICON_DATA_URL,
                alt="",
                class_="nb-fab-icon",
                **{"aria-hidden": "true"},
            ),
            id="nb-fab",
            class_="nb-fab",
            type="button",
            **{"aria-label": "Open Auditor Notebook", "title": "Auditor Notebook"},
        ),
        # ── Unexported-entries warning badge + popover ───────────
        ui.tags.button(
            ui.HTML(
                '<svg viewBox="0 0 24 24" width="27" height="27" fill="none" '
                'stroke="currentColor" stroke-width="2.2" stroke-linecap="round" '
                'stroke-linejoin="round" aria-hidden="true">'
                '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 '
                '1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
                '<line x1="12" y1="9" x2="12" y2="13"/>'
                '<line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
            ),
            id="nb-export-warn",
            class_="nb-export-warn",
            type="button",
            **{
                "aria-label": "Unexported notebook entries — click for details",
                "title": "Unexported notebook entries",
            },
        ),
        ui.tags.div(
            ui.tags.p(
                ui.HTML(
                    '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" '
                    'stroke="currentColor" stroke-width="2.2" stroke-linecap="round" '
                    'stroke-linejoin="round" aria-hidden="true">'
                    '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 '
                    '1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
                    '<line x1="12" y1="9" x2="12" y2="13"/>'
                    '<line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
                ),
                "Not exported yet",
                class_="nb-export-warn-pop-title",
            ),
            ui.tags.p(
                "",
                id="nb-export-warn-msg",
                class_="nb-export-warn-pop-msg",
            ),
            ui.tags.button(
                "Open the notebook to export",
                id="nb-export-warn-open",
                class_="nb-export-warn-open-btn",
                type="button",
            ),
            id="nb-export-warn-pop",
            class_="nb-export-warn-pop",
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
                        "Title is optional. The five thesis elements are saved: "
                        "hypothesis, automatic conditions, automatic signals, "
                        "uncertainty, and next steps.",
                        class_="nb-section-sub",
                    ),
                    _field(
                        "nb_case_id",
                        "Audit case ID (optional)",
                        'Group related entries under one investigation, e.g. "crows-pairs-race-bert-2026-05".',
                        kind="text",
                        placeholder="e.g. crows-pairs-race-bert-2026-05",
                    ),
                    _field(
                        "nb_title",
                        "Title (optional)",
                        "A short label so the entry is easy to find later.",
                        kind="text",
                        placeholder="e.g. Race-axis stereotype on CrowS-Pairs item 47",
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
                        "Concrete follow-ups: another model, another axis, another prompt, a control.",
                        kind="textarea",
                        rows=2,
                        placeholder="e.g. Run the same item on GPT-2 to check whether the race-bias localisation in L5H3 generalises across architectures.",
                    ),
                    # ── Context preview (live read of dashboard state) ──
                    ui.tags.div(
                        ui.tags.div(
                            ui.tags.span(class_="nb-ctx-preview-dot"),
                            "Conditions and signals captured automatically",
                            class_="nb-ctx-preview-title",
                        ),
                        ui.output_ui("nb_context_preview"),
                        class_="nb-ctx-preview-wrapper",
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
                    ui.output_ui("nb_export_status"),
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
                        ui.download_button(
                            "nb_download_csv",
                            "Export CSV",
                            class_="nb-btn nb-btn-secondary",
                        ),
                        ui.input_action_button(
                            "nb_clear_all",
                            "Clear all",
                            class_="nb-btn nb-btn-danger",
                        ),
                        id="nb-export-row",
                        class_="nb-export-row",
                    ),
                    # Arms the confirmation strip; the clear itself only runs
                    # from the button inside it.
                    ui.output_ui("nb_clear_confirm"),
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
