JS_CODE = """
        // Define global functions immediately (before Shiny connects)
        window.switchView = function(containerId, tabName) {
            // 1. Hide all panes in this container
            var panes = document.querySelectorAll('#' + containerId + ' .view-pane');
            panes.forEach(function(pane) {
                pane.style.display = 'none';
            });
            
            // 2. Show target pane
            var target = document.getElementById(containerId + '_' + tabName);
            if(target) target.style.display = 'block';
            
            // 3. Update buttons
            var buttons = document.querySelectorAll('#' + containerId + ' .view-btn');
            buttons.forEach(function(btn) {
                btn.classList.remove('active');
                if(btn.dataset.tab === tabName) btn.classList.add('active');
            });
        };

        window.toggleMlmDetails = function(id) {
            var el = document.getElementById(id);
            if (el.style.display === 'none' || el.style.display === '') {
                el.style.display = 'block';
            } else {
                el.style.display = 'none';
            }
        };

        window.toggleTreeNode = function(nodeId) {
            var children = document.getElementById(nodeId + '-children');
            var toggle = document.getElementById(nodeId + '-toggle');
            if (children && toggle) {
                children.classList.toggle('collapsed');
                toggle.classList.toggle('collapsed');
            }
        };

        // Dynamic tooltip positioning for fixed-position tooltips
        (function() {
            function positionTooltip(wrapper, tooltip) {
                var rect = wrapper.getBoundingClientRect();
                var tooltipWidth = tooltip.offsetWidth || 380;
                var tooltipHeight = tooltip.offsetHeight || 200;
                
                // Position below the icon, centered
                var left = rect.left + (rect.width / 2) - (tooltipWidth / 2);
                var top = rect.bottom + 8;
                
                // Ensure tooltip doesn't go off-screen horizontally
                if (left < 10) left = 10;
                if (left + tooltipWidth > window.innerWidth - 10) {
                    left = window.innerWidth - tooltipWidth - 10;
                }
                
                // If tooltip would go below viewport, show above instead
                if (top + tooltipHeight > window.innerHeight - 10) {
                    top = rect.top - tooltipHeight - 8;
                }
                if (top < 10) top = 10;
                
                tooltip.style.left = left + 'px';
                tooltip.style.top = top + 'px';
            }
            
            document.addEventListener('mouseover', function(e) {
                var wrapper = e.target.closest('.info-tooltip-wrapper');
                if (wrapper) {
                    var tooltip = wrapper.querySelector('.info-tooltip-content');
                    if (tooltip) {
                        positionTooltip(wrapper, tooltip);
                    }
                }
            });
        })();


        // Show global metric info popup on hover
        window.showGlobalMetricInfo = function(iconElement) {
            // Remove any existing popup first
            var existingPopup = document.getElementById('global-metric-popup');
            if (existingPopup) {
                existingPopup.remove();
            }
            
            // Create popup
            var popup = document.createElement('div');
            popup.id = 'global-metric-popup';
            popup.innerHTML = '<strong style="color:#ff5ca9;">Global Metric</strong><br>Uses ALL layers to measure how attention patterns transform from first to last layer.';
            popup.style.cssText = 'position: fixed; z-index: 999999; background: linear-gradient(135deg, #1e293b 0%, #334155 100%); color: #f1f5f9; padding: 12px 16px; border-radius: 10px; font-size: 12px; line-height: 1.6; max-width: 250px; box-shadow: 0 10px 40px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,92,169,0.3);';
            
            // Position near the icon
            var rect = iconElement.getBoundingClientRect();
            popup.style.left = (rect.left + rect.width/2 - 125) + 'px'; // Center horizontally
            popup.style.top = (rect.bottom + 8) + 'px';
            
            // Ensure popup doesn't go off-screen
            document.body.appendChild(popup);
            var popupRect = popup.getBoundingClientRect();
            if (popupRect.right > window.innerWidth - 10) {
                popup.style.left = (window.innerWidth - popupRect.width - 10) + 'px';
            }
            if (popupRect.left < 10) {
                popup.style.left = '10px';
            }
        };

        // Hide global metric info popup
        window.hideGlobalMetricInfo = function() {
            var popup = document.getElementById('global-metric-popup');
            if (popup) {
                popup.remove();
            }
        };


        // Define showMetricModal early so it's available for onclick handlers
        window.showMetricModal = function(metricName, layer, head) {
            var modal = document.getElementById('metric-modal');
            var title = document.getElementById('modal-title');
            var body = document.getElementById('modal-body');

            title.textContent = metricName;

            var explanations = {
                'Syntax': {
                    formula: 'SYN<sup>l,h</sup> = (Σ<sub>i,j∈syntax</sub> A<sub>ij</sub><sup>l,h</sup>) / (Σ<sub>i,j</sub> A<sub>ij</sub><sup>l,h</sup>)',
                    description: 'Proportion of total attention mass directed toward function words (determiners, prepositions, auxiliaries, conjunctions, particles, pronouns). POS tags are identified using spaCy\\'s part-of-speech tagger and include: DET, ADP, AUX, CCONJ, SCONJ, PART, PRON. The metric sums all attention weights targeting these syntactic tokens and divides by the total attention mass.',
                    interpretation: 'Higher values (closer to 1) indicate the head specializes in syntactic structure, focusing on grammatical scaffolding rather than semantic content. These heads typically play a role in parsing sentence structure and establishing grammatical relationships. Low values suggest the head ignores function words in favor of content.'
                },
                'Semantics': {
                    formula: 'SEM<sup>l,h</sup> = (Σ<sub>i,j∈semantics</sub> A<sub>ij</sub><sup>l,h</sup>) / (Σ<sub>i,j</sub> A<sub>ij</sub><sup>l,h</sup>)',
                    description: 'Proportion of total attention mass directed toward content-bearing words (nouns, proper nouns, verbs, adjectives, adverbs, numerals). POS tags from spaCy include: NOUN, PROPN, VERB, ADJ, ADV, NUM. The metric sums all attention weights targeting these semantic tokens and divides by the total attention mass.',
                    interpretation: 'Higher values (closer to 1) indicate the head specializes in semantic content, tracking meaning-carrying words that convey the main ideas and concepts. These heads typically focus on topic words and key information. Low values suggest the head prioritizes structural elements over semantic ones.'
                },
                'CLS Focus': {
                    formula: 'CLS<sup>l,h</sup> = (1/n) Σ<sub>i=1</sub><sup>n</sup> A<sub>i,CLS</sub><sup>l,h</sup>',
                    description: 'Average attention weight from all tokens to the [CLS] token at position 0. Computed by taking column 0 of the attention matrix (all queries attending to [CLS]) and averaging across all query positions.',
                    interpretation: 'Higher values (closer to 1) indicate the head uses [CLS] as a central aggregation point, pulling information from the entire sequence into this special token. This is common in later layers where [CLS] accumulates sentence-level representations. Low values suggest the head doesn\\'t use [CLS] as a special aggregation point.'
                },
                'Punctuation': {
                    formula: 'PUNC<sup>l,h</sup> = (Σ<sub>i,j∈punct</sub> A<sub>ij</sub><sup>l,h</sup>) / (Σ<sub>i,j</sub> A<sub>ij</sub><sup>l,h</sup>)',
                    description: 'Proportion of total attention mass directed toward punctuation marks. Punctuation is identified using Python\\'s string.punctuation set (.,!?;:\\'\"()[]{}-/\\\\). The metric sums all attention weights targeting punctuation tokens and divides by the total attention mass.',
                    interpretation: 'Higher values (closer to 1) indicate the head uses punctuation as structural anchors or boundary markers, often for clause/phrase segmentation. These heads may help identify sentence boundaries or syntactic breaks. Low values suggest the head ignores punctuation entirely.'
                },
                'Entities': {
                    formula: 'ENT<sup>l,h</sup> = (Σ<sub>i,j∈entities</sub> A<sub>ij</sub><sup>l,h</sup>) / (Σ<sub>i,j</sub> A<sub>ij</sub><sup>l,h</sup>)',
                    description: 'Proportion of total attention mass directed toward named entities (people, organizations, locations, etc.). Named Entity Recognition tags are identified using spaCy\\'s NER tagger - any token with a tag other than "O" (outside) is considered an entity. The metric sums all attention weights targeting entity tokens and divides by the total attention mass.',
                    interpretation: 'Higher values (closer to 1) indicate the head specializes in tracking named entities and important noun phrases across the sequence. This suggests a role in coreference resolution or entity tracking. Low values (or 0 if no entities present) suggest the head doesn\\'t prioritize named entities.'
                },
                'Long-range': {
                    formula: 'LR<sup>l,h</sup> = mean(A<sub>ij</sub><sup>l,h</sup> | |i-j| ≥ 5)',
                    description: 'Average attention weight for token pairs separated by 5 or more positions. Only attention weights where the absolute distance between query position i and key position j is at least 5 are included in the calculation. This measures the head\\'s tendency to bridge distant tokens rather than focusing on local context.',
                    interpretation: 'Higher values indicate the head specializes in long-range dependencies, connecting tokens that are far apart in the sequence. This is important for capturing global context and long-distance relationships. Low values suggest the head focuses primarily on local neighborhoods and immediate context.'
                },
                'Self-attention': {
                    formula: 'SELF<sup>l,h</sup> = (1/n) Σ<sub>i=1</sub><sup>n</sup> A<sub>ii</sub><sup>l,h</sup>',
                    description: 'Average of the diagonal elements of the attention matrix, measuring how much each token attends to itself. Computed by extracting the diagonal (where i = j) and averaging these self-attention weights across all positions.',
                    interpretation: 'Higher values (closer to 1) indicate strong self-attention loops where tokens primarily attend to themselves. This often serves to preserve token identity or stabilize representations. Lower values suggest the head focuses on contextual relationships rather than self-preservation.'
                },
                'Confidence Max': {
                    formula: 'MaxA = max<sub>i,j</sub>(a<sub>ij</sub>)',
                    description: 'The maximum attention weight in the attention matrix (Eq. 5). Measures the strongest connection between any query-key pair.',
                    interpretation: 'Higher values indicate that this head has a very confident focus on a specific token. Values close to 1 suggest the head is highly specialized.',
                    typicalRange: '<b>Low:</b> < 0.2 (diffuse) · <b>Medium:</b> 0.2–0.5 · <b>High:</b> > 0.5 (confident)',
                    paper: 'From Attention to Assurance (Eq. 5)'
                },
                'Confidence Avg': {
                    formula: 'AvgMaxA = (1/d<sub>k</sub>) Σ<sub>i</sub> max<sub>j</sub>(a<sub>ij</sub>)',
                    description: 'Average of the maximum attention weight per row (Eq. 6). Each row represents how a query token attends to all key tokens.',
                    interpretation: 'This metric captures the overall confidence level. High values suggest the head consistently focuses strongly on specific tokens for each query.',
                    typicalRange: '<b>Low:</b> < 0.15 · <b>Medium:</b> 0.15–0.4 · <b>High:</b> > 0.4 (consistently confident)',
                    paper: 'From Attention to Assurance (Eq. 6)'
                },
                'Focus': {
                    formula: 'Focus = E / log(n²), where E = -Σ a<sub>ij</sub> log(a<sub>ij</sub>)',
                    description: 'Normalized attention entropy (Eq. 8). Divided by max entropy (log n²) to give value between 0 and 1.',
                    interpretation: '0 = fully focused (one token). 1 = fully uniform (all tokens equal). <b>Lower is more focused</b>.',
                    typicalRange: '<b>Focused:</b> < 0.3 · <b>Moderate:</b> 0.3–0.7 · <b>Diffuse:</b> > 0.7',
                    paper: 'From Attention to Assurance (Eq. 8)'
                },
                'Sparsity': {
                    formula: 'S = (1/n²) Σ<sub>ij</sub> 𝟙(a<sub>ij</sub> < τ), where τ = 1/n',
                    description: 'Proportion of attention weights below adaptive threshold τ = 1/seq_len (Eq. 11). Uses adaptive threshold for length-independence.',
                    interpretation: 'High sparsity = most tokens ignored (selective). Low sparsity = attention distributed across many tokens.',
                    typicalRange: '<b>Low:</b> < 30% · <b>Medium:</b> 30%–60% · <b>High:</b> > 60% (very selective)',
                    paper: 'From Attention to Assurance (Eq. 11)'
                },
                'Distribution': {
                    formula: 'Q<sub>0.5</sub> = median(A)',
                    description: 'The median (50th percentile) of all attention weights (Eq. 12).',
                    interpretation: 'Low median + high max = attention concentrated on few tokens. High median = more evenly distributed.',
                    typicalRange: '<b>Low:</b> < 0.005 · <b>Medium:</b> 0.005–0.02 · <b>High:</b> > 0.02',
                    paper: 'From Attention to Assurance (Eq. 12)'
                },
                'Uniformity': {
                    formula: 'U = std(A) = √[(1/n²) Σ<sub>ij</sub> (a<sub>ij</sub> - μ)²]',
                    description: 'Standard deviation of attention weights (Eq. 15). Measures variability in the attention distribution.',
                    interpretation: 'Low = uniform/homogeneous attention. High = variable attention (some strong, some weak).',
                    typicalRange: '<b>Low:</b> < 0.03 (uniform) · <b>Medium:</b> 0.03–0.10 · <b>High:</b> > 0.10 (variable)',
                    paper: 'From Attention to Assurance (Eq. 15)'
                },
                'Flow Change': {
                    formula: 'JSD(L<sub>first</sub>, L<sub>last</sub>) = √[ ½×KL(P||M) + ½×KL(Q||M) ]',
                    description: 'Jensen-Shannon Divergence between first and last layer attention distributions (Eq. 9). Measures how attention patterns transform through the model.',
                    interpretation: 'Low = static patterns (first≈last). High = effective feature extraction (diverse representations learned). Higher JSD correlates with fewer errors.',
                    typicalRange: '<b>Low:</b> < 0.10 (static) · <b>Medium:</b> 0.10–0.25 · <b>High:</b> > 0.25 (good transformation)',
                    paper: 'From Attention to Assurance (Eq. 9)'
                },
                'Balance': {
                    formula: 'Balance = attn_to_CLS / attn_total',
                    description: 'Proportion of attention directed to [CLS] token vs total attention. Normalized 0-1 range. Relevant for bias detection.',
                    interpretation: '0 = all to content. 0.5 = balanced. 1 = all to [CLS]. High values suggest shortcut learning (common in biased models).',
                    typicalRange: '<b>Low:</b> < 0.15 (content) · <b>Medium:</b> 0.15–0.40 · <b>High:</b> > 0.40 (CLS focus)',
                    paper: 'From Attention to Assurance (Eq. 14)'
                }
            };

            var info = explanations[metricName];
            if (info) {
                var referenceBlock = info.paper ? `
                    <div class="modal-section">
                        <h4>Reference</h4>
                        <p style="font-size:11px;line-height:1.6;">
                            Golshanrad, Pouria and Faghih, Fathiyeh, <em>From Attention to Assurance: Enhancing Transformer Encoder Reliability Through Advanced Testing and Online Error Prediction</em>.
                            <a href="https://ssrn.com/abstract=4856933" target="_blank" style="color:#ff5ca9;text-decoration:none;border-bottom:1px solid rgba(255,92,169,0.3);">Available at SSRN</a> or
                            <a href="http://dx.doi.org/10.2139/ssrn.4856933" target="_blank" style="color:#ff5ca9;text-decoration:none;border-bottom:1px solid rgba(255,92,169,0.3);">DOI</a>
                        </p>
                    </div>
                ` : '';

                var typicalRangeBlock = info.typicalRange ? `
                    <div class="modal-section">
                        <h4>Typical Ranges</h4>
                        <p style="font-size:12px;line-height:1.8;background:rgba(255,92,169,0.05);padding:10px 12px;border-radius:8px;border-left:3px solid #ff5ca9;">
                            ${info.typicalRange}
                        </p>
                    </div>
                ` : '';

                body.innerHTML = `
                    <div class="modal-section">
                        <h4>Formula</h4>
                        <div class="modal-formula">${info.formula}</div>
                    </div>
                    <div class="modal-section">
                        <h4>Description</h4>
                        <p>${info.description}</p>
                    </div>
                    <div class="modal-section">
                        <h4>Interpretation</h4>
                        <p>${info.interpretation}</p>
                    </div>
                    ${typicalRangeBlock}
                    ${referenceBlock}
                `;
            }

            modal.style.display = 'block';
        };
        """

# Main interactive JavaScript code
JS_INTERACTIVE = """
        // Handle spinner visibility
        $(document).on('shiny:busy', function() {
            // Only show if we are not already showing the custom spinner
            if ($('#loading_spinner').css('display') === 'none') {
                 // Optional: show a global spinner if needed, but we use the sidebar one
            }
        });

        // Custom message handlers
        Shiny.addCustomMessageHandler('start_loading', function(msg) {
            var btn = $('#generate_all');
            if (!btn.data('original-content')) {
                btn.data('original-content', btn.html());
            }
            btn.html('<div class="spinner" style="width:16px;height:16px;border-width:2px;display:inline-block;vertical-align:middle;margin-right:8px;"></div>Processing<span class="loading-dots"></span>');
            btn.prop('disabled', true).css('opacity', '0.8');
            $('#dashboard-container').addClass('content-hidden').removeClass('content-visible');
        });

        Shiny.addCustomMessageHandler('stop_loading', function(msg) {
            var btn = $('#generate_all');
            // Restore original content if available, otherwise default
            if (btn.data('original-content')) {
                btn.html(btn.data('original-content'));
            } else {
                // If checking dynamic label state is complex, just keep current text minus spinner?
                // Better: rely on update_button_label to have set the correct original-content
                btn.html('Generate All'); 
            }
            btn.prop('disabled', false).css('opacity', '1');
        });

        // NEW: Handle dynamic button label updates cleanly
        Shiny.addCustomMessageHandler('update_button_label', function(msg) {
            console.log("JS: update_button_label received:", msg);
            setTimeout(function() {
                var btn = $('#generate_all');
                
                if (btn.length === 0) {
                    console.error("JS: Button #generate_all NOT FOUND");
                    return;
                }

                var newLabel = msg.label;
                
                // Check current state
                var isDisabled = btn.prop('disabled');

                if (isDisabled) {
                    // If currently loading, update the stored original content
                    btn.data('original-content', newLabel);
                } else {
                    // Immediate update
                    btn.html(newLabel);
                    // Also update stored content just in case
                    btn.data('original-content', newLabel); 
                }
            }, 50); // Small delay to win any race conditions
        });

        // Bias Loading Handlers
        Shiny.addCustomMessageHandler('start_bias_loading', function(msg) {
            $('#bias_loading_spinner').css('display', 'flex');
            $('#analyze_bias_btn').prop('disabled', true).css('opacity', '0.7');
        });

        Shiny.addCustomMessageHandler('stop_bias_loading', function(msg) {
            $('#bias_loading_spinner').css('display', 'none');
            $('#analyze_bias_btn').prop('disabled', false).css('opacity', '1');
        });

        // showMetricModal is already defined above - no need to redefine

        // ISA Overlay handler
        function showISAOverlay(sentXIdx, sentYIdx, sentXText, sentYText, isaScore) {
            var modal = document.getElementById('isa-overlay-modal');
            var sentXEl = document.getElementById('isa-sentence-x');
            var sentYEl = document.getElementById('isa-sentence-y');
            var scoreEl = document.getElementById('isa-score');

            // Populate sentence data
            sentXEl.textContent = sentXText;
            sentYEl.textContent = sentYText;

            // Safe number parsing
            var safeScore = Number(isaScore);
            if (isNaN(safeScore)) {
                safeScore = 0.0;
            }
            scoreEl.textContent = safeScore.toFixed(4);

            // Store indices for Shiny to access
            window.isa_selected_pair = [sentXIdx, sentYIdx];

            // Trigger Shiny reactive to update visualization
            if (typeof Shiny !== 'undefined') {
                Shiny.setInputValue('isa_overlay_trigger', {
                    sentXIdx: sentXIdx,
                    sentYIdx: sentYIdx,
                    timestamp: Date.now()
                }, {priority: 'event'});
            }

            // Show modal
            modal.style.display = 'block';
        }
        window.showISAOverlay = showISAOverlay;

        window.onclick = function(event) {
            var modal = document.getElementById('metric-modal');
            if (event.target == modal) {
                modal.style.display = 'none';
            }
            var isaModal = document.getElementById('isa-overlay-modal');
            if (event.target == isaModal) {
                isaModal.style.display = 'none';
            }
        }

        // Handle radar mode change - show/hide head selector
        $(document).on('change', 'input[name="radar_mode"]', function() {
            const mode = $(this).val();
            if (mode === 'all') {
                $('#radar_head_selector').hide();
            } else {
                $('#radar_head_selector').show();
            }
        });

        // MLM details toggle
        function toggleMlmDetails(id) {
            const panel = document.getElementById(id);
            if (panel) {
                panel.style.display = panel.style.display === 'block' ? 'none' : 'block';
            }
        }
        window.toggleMlmDetails = toggleMlmDetails;

        // ISA Matrix Click Handler - attach to Plotly plot after it's rendered
        $(document).on('shiny:value', function(event) {
            if (event.name === 'isa_matrix') {
                setTimeout(function() {
                    var isaPlot = document.querySelector('#isa_matrix .js-plotly-plot');
                    if (isaPlot && !isaPlot.hasAttribute('data-isa-listener')) {
                        isaPlot.setAttribute('data-isa-listener', 'true');
                        isaPlot.on('plotly_click', function(data) {
                            if (data.points && data.points.length > 0) {
                                var point = data.points[0];
                                var customdata = point.customdata;
                                if (customdata && customdata.length >= 5) {
                                    var sentXIdx = customdata[0];
                                    var sentYIdx = customdata[1];
                                    var sentXText = customdata[2];
                                    var sentYText = customdata[3];
                                    var isaScore = customdata[4];
                                    showISAOverlay(sentXIdx, sentYIdx, sentXText, sentYText, isaScore);
                                }
                            }
                        });
                    }
                }, 500);
            }

            // Force horizontal scroll for Attention Flow plot
            if (event.name === 'attention_flow') {
                setTimeout(function() {
                    var flowContainer = document.getElementById('attention_flow');
                    if (flowContainer) {
                        // Set overflow styles
                        flowContainer.style.overflowX = 'auto';
                        flowContainer.style.overflowY = 'hidden';
                        flowContainer.style.width = '100%';

                        // Set parent styles
                        var parent = flowContainer.parentElement;
                        if (parent) {
                            parent.style.overflowX = 'auto';
                            parent.style.overflowY = 'hidden';
                            parent.style.width = '100%';
                        }

                        // Find and configure Plotly div
                        var plotlyDiv = flowContainer.querySelector('.js-plotly-plot');
                        if (plotlyDiv && window.Plotly) {
                            // Disable responsive behavior
                            window.Plotly.Plots.resize(plotlyDiv);
                        }
                    }
                }, 500);
            }
        });
        // Reliable Plotly click → Shiny input for ISA matrix
        $(document).on("shiny:connected", function () {
            function attachIsaClick() {
                const plot = document.querySelector("#isa_matrix .js-plotly-plot");
                if (plot && !plot.dataset.isaListener) {
                    plot.dataset.isaListener = "true";
                    plot.on("plotly_click", function (e) {
                        if (!e.points || !e.points[0]) return;
                        const pt = e.points[0];
                        const cd = pt.customdata;
                        if (!cd) return;

                        Shiny.setInputValue("isa_click", {
                            x: pt.x,  // source sentence index (Sentence B)
                            y: pt.y   // target sentence index (Sentence A)
                        }, {priority: "event"});
                    });
                }
            }

            // Re-attach every time the plot is re-rendered
            $(document).on("shiny:value", function (ev) {
                if (ev.name === "isa_matrix") setTimeout(attachIsaClick, 100);
            });

            attachIsaClick();
        });

        """

# D3.js tree visualization code
JS_TREE_VIZ = """
        // Token Influence Tree D3.js Visualization
        function renderInfluenceTree(treeData, containerId) {
            d3.select(`#${containerId}`).selectAll("*").remove();

            if (!treeData || !treeData.name) {
                d3.select(`#${containerId}`)
                    .append("p")
                    .style("font-size", "11px")
                    .style("color", "#9ca3af")
                    .style("padding", "20px")
                    .text("Generate attention data to view the influence tree.");
                return;
            }

            // Vertical tree configuration
            const container = document.getElementById(containerId);
            const containerWidth = container ? container.clientWidth : 600;

            const margin = {top: 60, right: 60, bottom: 60, left: 60};
            
            const colors = {
                root: '#ff5ca9',
                level1: '#3b82f6',
                level2: '#8b5cf6',
                level3: '#06b6d4'
            };

            // 1. Process Data & Layout first (before creating SVG)
            const root = d3.hierarchy(treeData);
            const maxDepth = root.height;
            
            // Calculate dynamic size
            // Height: based on depth (fixed pixels per level)
            // Width: based on max breadth (max nodes at any level)
            const depthStep = 140; // Pixels per depth level
            const nodeWidthSpacing = 80; // Pixels between nodes at same level
            
            // Calculate max breadth
            const levelCounts = {};
            root.each(d => {
                const depth = d.depth;
                levelCounts[depth] = (levelCounts[depth] || 0) + 1;
            });
            const values = Object.values(levelCounts);
            const maxBreadth = values.length > 0 ? Math.max(...values) : 1;
            
            // Determine SVG dimensions
            // Height = (depth + 1) * spacing + margins
            const fullHeight = ((maxDepth + 1) * depthStep) + margin.top + margin.bottom;
            
            // Width = maxBreadth * spacing + margins (or container width, whichever is larger)
            const requiredWidth = (maxBreadth * nodeWidthSpacing) + margin.left + margin.right;
            const fullWidth = Math.max(containerWidth, requiredWidth);

            const tree = d3.tree().size([fullWidth - margin.left - margin.right, fullHeight - margin.top - margin.bottom]);
            tree(root);

            // 4. Create SVG with scrollable dimensions
            const svg = d3.select(`#${containerId}`)
                .append("svg")
                .attr("width", fullWidth)
                .attr("height", fullHeight)
                .style("font", "12px 'Inter', sans-serif")
                .style("display", "block")
                .style("margin", "0 auto"); // Center horizontally if smaller than container

            const g = svg.append("g")
                .attr("transform", `translate(${margin.left},${margin.top})`);
            
            let i = 0;
            update(root);

            function update(source) {
                // Recompute layout
                tree(root);

                const nodes = root.descendants();
                const links = root.descendants().slice(1);

                // Start from center horizontal
                // But D3 tree layout calculates x, y based on .size()
                // We just use those x, y coordinates
                
                // Override y for strictly uniform depth spacing
                nodes.forEach(d => { d.y = d.depth * depthStep; });

                const node = g.selectAll('g.node')
                    .data(nodes, d => d.id || (d.id = ++i));

                const nodeEnter = node.enter().append('g')
                    .attr('class', 'node')
                    .attr("transform", d => `translate(${source.x0 || source.x},${source.y0 || source.y})`)
                    .on('click', click);

                nodeEnter.append('circle')
                    .attr('class', 'node-circle')
                    .attr('r', 1e-6)
                    .style("fill", d => getNodeColor(d))
                    .style("stroke", d => getNodeColor(d))
                    .style("stroke-width", d => 2 + (d.data.att || 0) * 3)
                    .style("opacity", d => 0.3 + (d.data.att || 0) * 0.7);

                nodeEnter.append('text')
                    .attr("dy", d => d.depth === 0 ? "-2.5em" : "-1.5em")
                    .attr("text-anchor", "middle")
                    .text(d => d.data.name)
                    .style("fill", d => getNodeColor(d))
                    .style("font-weight", d => d.depth === 0 ? "700" : "500")
                    .style("font-size", d => d.depth === 0 ? "14px" : "12px")
                    .style("background", "white") // Fake shadow for readability? No, just SVG text
                    .style("text-shadow", "0 1px 2px rgba(255,255,255,0.8)");

                nodeEnter.append('text')
                    .attr("dy", d => d.depth === 0 ? "-1em" : "1.8em") 
                    .attr("text-anchor", "middle")
                    .text(d => d.depth > 0 ? `${(d.data.att || 0).toFixed(3)}` : "")
                    .style("fill", "#64748b")
                    .style("font-size", "10px");

                nodeEnter.append("title")
                    .text(d => {
                        if (d.depth === 0) return `Root: ${d.data.name}`;
                        return `Token: ${d.data.name}\\nAttention: ${(d.data.att || 0).toFixed(4)}\\nQ·K Similarity: ${(d.data.qk_sim || 0).toFixed(4)}`;
                    });

                const nodeUpdate = nodeEnter.merge(node);

                nodeUpdate.transition()
                    .duration(750)
                    .attr("transform", d => `translate(${d.x},${d.y})`);

                nodeUpdate.select('circle.node-circle')
                    .attr('r', d => 6 + (d.data.att || 0) * 4)
                    .style("fill", d => d._children ? getNodeColor(d) : "#fff")
                    .style("cursor", "pointer");

                const nodeExit = node.exit().transition()
                    .duration(750)
                    .attr("transform", d => `translate(${source.x},${source.y})`)
                    .remove();

                nodeExit.select('circle').attr('r', 1e-6);
                nodeExit.select('text').style('fill-opacity', 1e-6);

                const link = g.selectAll('path.link')
                    .data(links, d => d.id);

                const linkEnter = link.enter().insert('path', "g")
                    .attr("class", "link")
                    .attr('d', d => {
                        const o = {x: source.x0 || source.x, y: source.y0 || source.y};
                        return diagonal(o, o);
                    })
                    .style("fill", "none")
                    .style("stroke", d => getNodeColor(d))
                    .style("stroke-width", d => 1 + (d.data.att || 0) * 4)
                    .style("opacity", d => 0.2 + (d.data.att || 0) * 0.6);

                const linkUpdate = linkEnter.merge(link);

                linkUpdate.transition()
                    .duration(750)
                    .attr('d', d => diagonal(d, d.parent));

                link.exit().transition()
                    .duration(750)
                    .attr('d', d => {
                        const o = {x: source.x, y: source.y};
                        return diagonal(o, o);
                    })
                    .remove();

                nodes.forEach(d => {
                    d.x0 = d.x;
                    d.y0 = d.y;
                });

                // Vertical diagonal path
                function diagonal(s, d) {
                    return `M ${s.x} ${s.y}
                            C ${s.x} ${(s.y + d.y) / 2},
                              ${d.x} ${(s.y + d.y) / 2},
                              ${d.x} ${d.y}`;
                }

                function click(event, d) {
                    if (d.children) {
                        d._children = d.children;
                        d.children = null;
                    } else {
                        d.children = d._children;
                        d._children = null;
                    }
                    update(d);
                }
            }

            function getNodeColor(d) {
                if (d.depth === 0) return colors.root;
                if (d.depth === 1) return colors.level1;
                if (d.depth === 2) return colors.level2;
                return colors.level3;
            }
            // Smart scroll: center the tree initially
            setTimeout(() => {
                if (container && container.scrollWidth > container.clientWidth) {
                    container.scrollLeft = (container.scrollWidth - container.clientWidth) / 2;
                }
            }, 100);
        }

        window.renderInfluenceTree = renderInfluenceTree;
        console.log('D3.js version:', typeof d3 !== 'undefined' ? d3.version : 'not loaded');
        console.log('renderInfluenceTree:', typeof renderInfluenceTree !== 'undefined' ? 'loaded' : 'not loaded');

        document.addEventListener("DOMContentLoaded", function() {
            const origError = console.error;
            console.error = function(...args) {
                if (args[0] && typeof args[0] === "string" && args[0].includes("anywidget")) {
                    return; // silencia
                }
                origError.apply(console, args);
            };
        });
        """

# Transition modal code
JS_TRANSITION_MODAL = """
        window.showTransitionModal = function(fromSection, toSection, modelType) {
            var modalId = 'transition-modal';
            var modal = document.getElementById(modalId);

            if (!modal) {
                modal = document.createElement('div');
                modal.id = modalId;
                modal.className = 'modal';
                document.body.appendChild(modal);

                // Close on outside click
                window.addEventListener('click', function(event) {
                    if (event.target == modal) {
                        modal.style.display = "none";
                    }
                });
            }

            var explanation = getTransitionExplanation(fromSection, toSection, modelType);

            var content = `
                <div class="modal-content" style="max-width: 650px; border: 1px solid rgba(255, 92, 169, 0.3);">
                    <div class="modal-header" style="border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 15px; margin-bottom: 15px;">
                        <div class="modal-title" style="font-size: 18px; display: flex; align-items: center; gap: 10px;">
                            <span style="color: #cbd5e1; font-weight: 500;">${fromSection}</span>
                            <span style="color: #64748b; font-size: 16px;">➜</span>
                            <span style="color: #ff5ca9; font-weight: 700;">${toSection}</span>
                        </div>
                        <span class="close-btn" onclick="document.getElementById('${modalId}').style.display='none'" style="color: #64748b; font-size: 24px; line-height: 1; cursor: pointer;">&times;</span>
                    </div>
                    <div class="modal-body" style="font-size: 14px; line-height: 1.7; color: #e2e8f0;">
                        ${explanation}
                    </div>
                </div>
            `;

            modal.innerHTML = content;
            modal.style.display = 'block';
        };

        window.showISACalcExplanation = function(modelType) {
            var modalId = 'isa-calc-modal';
            var modal = document.getElementById(modalId);

            if (!modal) {
                modal = document.createElement('div');
                modal.id = modalId;
                modal.className = 'modal';
                document.body.appendChild(modal);

                window.addEventListener('click', function(event) {
                    if (event.target == modal) {
                        modal.style.display = "none";
                    }
                });
            }

            var content = "";
            if (modelType === 'BERT') {
                content = `
                    <div class="modal-content" style="max-width: 780px; border: 1px solid rgba(255, 92, 169, 0.3);">
                        <div class="modal-header" style="border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px;">
                            <h3 class="modal-title" style="color: #ff5ca9;">Inter-Sentence Attention (ISA): BERT</h3>
                            <span class="close-btn" onclick="document.getElementById('${modalId}').style.display='none'" style="color: #64748b; cursor: pointer;">&times;</span>
                        </div>
                        <div class="modal-body" style="font-size: 13px; line-height: 1.7; color: #e2e8f0; padding-top: 6px;">
                            <h4 style="color: #ff5ca9; margin-top: 0;">Definition</h4>
                            <p>Inter-Sentence Attention (ISA) captures <strong>sentence-level relationships</strong> by aggregating token-level attention patterns. It reduces the complexity of attention analysis from O(n²) to O(m²), where n is the number of tokens and m is the number of sentences (n ≫ m).</p>
                            
                            <h4 style="color: #ff5ca9; margin-top: 16px;">Three-Step Computation</h4>
                            <p>Given sentences Sₐ and Sᵦ with token indices [iₐ, iₐ₊₁) and [iᵦ, iᵦ₊₁):</p>
                            
                            <div style="background: rgba(255,255,255,0.05); padding: 14px; border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 11px; margin: 12px 0; border-left: 3px solid #ff5ca9;">
                                <strong style="color:#ff5ca9;">Step 1 — Layer Integration:</strong><br>
                                A(i,j) = max<sub>l∈L</sub> α<sub>l</sub>(i,j)<br><br>
                                <span style="color:#94a3b8;">where α<sub>l</sub>(i,j) = softmax(Q<sub>l,i</sub>K<sub>l,j</sub><sup>T</sup>/√d<sub>k</sub>)</span>
                            </div>
                            
                            <div style="background: rgba(255,255,255,0.05); padding: 14px; border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 11px; margin: 12px 0; border-left: 3px solid #3b82f6;">
                                <strong style="color:#3b82f6;">Step 2 — Token Pair Aggregation:</strong><br>
                                β<sub>h</sub>(Sₐ, Sᵦ) = max<sub>(i,j)∈Sₐ×Sᵦ</sub> A(i,j)<br><br>
                                <span style="color:#94a3b8;">Maximum attention between any token pair from the two sentences</span>
                            </div>
                            
                            <div style="background: rgba(255,255,255,0.05); padding: 14px; border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 11px; margin: 12px 0; border-left: 3px solid #8b5cf6;">
                                <strong style="color:#8b5cf6;">Step 3 — Head Aggregation:</strong><br>
                                ISA(Sₐ, Sᵦ) = max<sub>h∈H</sub> β<sub>h</sub>(Sₐ, Sᵦ)<br><br>
                                <span style="color:#94a3b8;">Maximum across all attention heads to preserve specialized patterns</span>
                            </div>

                            <h4 style="color: #ff5ca9; margin-top: 16px;">Why Maximum Aggregation?</h4>
                            <p>We use <strong>max</strong> rather than averaging to preserve strong signals from individual attention heads. Research shows different heads specialize in capturing specific linguistic patterns (Clark et al., 2019), so averaging would dilute these specialized relationships.</p>

                            <h4 style="color: #ff5ca9; margin-top: 16px;">BERT Properties (Bidirectional)</h4>
                            <ul style="padding-left: 20px; margin-bottom: 12px;">
                                <li><strong>Full Attention:</strong> Every token can attend to all other tokens (no causal mask)</li>
                                <li><strong>Near-Symmetric Matrix:</strong> ISA(Sₐ,Sᵦ) ≈ ISA(Sᵦ,Sₐ)</li>
                                <li><strong>Interpretation:</strong> Measures mutual semantic/syntactic relationship strength</li>
                            </ul>
                            
                            <p style="font-size: 11px; color: #64748b; margin-top: 16px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1);">📚 Reference: Seo, S., Yoo, S., Lee, H., Jang, Y., Park, J.H., & Kim, J. (2024). "A Sentence-Level Visualization of Attention in Large Language Models." SAVIS: <a href="https://pypi.org/project/savis" target="_blank" style="color:#ff5ca9;">pypi.org/project/savis</a></p>
                        </div>
                    </div>
                `;
            } else {
                content = `
                    <div class="modal-content" style="max-width: 780px; border: 1px solid rgba(255, 92, 169, 0.3);">
                        <div class="modal-header" style="border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px;">
                            <h3 class="modal-title" style="color: #ff5ca9;">Inter-Sentence Attention (ISA): GPT-2 (Causal)</h3>
                            <span class="close-btn" onclick="document.getElementById('${modalId}').style.display='none'" style="color: #64748b; cursor: pointer;">&times;</span>
                        </div>
                        <div class="modal-body" style="font-size: 13px; line-height: 1.7; color: #e2e8f0; padding-top: 6px;">
                            <h4 style="color: #ff5ca9; margin-top: 0;">Definition</h4>
                            <p>For <strong>causal (autoregressive) models</strong> like GPT-2, ISA computation must respect the causal mask: token i can only attend to tokens j where j ≤ i. This fundamentally changes the interpretation of cross-sentence attention.</p>
                            
                            <h4 style="color: #ff5ca9; margin-top: 16px;">Causal ISA Computation</h4>
                            
                            <div style="background: rgba(255,255,255,0.05); padding: 14px; border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 11px; margin: 12px 0; border-left: 3px solid #ff5ca9;">
                                <strong style="color:#ff5ca9;">Step 1 — Layer Integration (with causal mask):</strong><br>
                                A(i,j) = max<sub>l∈L</sub> α<sub>l</sub>(i,j) &nbsp;&nbsp;<strong style="color:#ef4444;">if i ≥ j, else 0</strong><br><br>
                                <span style="color:#94a3b8;">Causal constraint: future tokens are masked (attention = 0)</span>
                            </div>
                            
                            <div style="background: rgba(255,255,255,0.05); padding: 14px; border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 11px; margin: 12px 0; border-left: 3px solid #3b82f6;">
                                <strong style="color:#3b82f6;">Step 2 — Token Pair Aggregation (valid pairs only):</strong><br>
                                β<sub>h</sub>(Sₐ, Sᵦ) = max<sub>(i,j)∈Sₐ×Sᵦ, i≥j</sub> A(i,j)<br><br>
                                <span style="color:#94a3b8;">Only token pairs where query position ≥ key position</span>
                            </div>
                            
                            <div style="background: rgba(255,255,255,0.05); padding: 14px; border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 11px; margin: 12px 0; border-left: 3px solid #8b5cf6;">
                                <strong style="color:#8b5cf6;">Step 3 — Head Aggregation:</strong><br>
                                ISA(Sₐ, Sᵦ) = max<sub>h∈H</sub> β<sub>h</sub>(Sₐ, Sᵦ)
                            </div>

                            <h4 style="color: #ff5ca9; margin-top: 16px;">GPT-2 Properties (Unidirectional)</h4>
                            <ul style="padding-left: 20px; margin-bottom: 12px;">
                                <li><strong>Causal Mask:</strong> Tokens only attend to previous tokens (left context)</li>
                                <li><strong>Lower Triangular Matrix:</strong> ISA(Sₐ,Sᵦ) = 0 when Sₐ comes before Sᵦ</li>
                                <li><strong>Asymmetric:</strong> ISA(Sₐ,Sᵦ) ≠ ISA(Sᵦ,Sₐ) — directionality matters!</li>
                                <li><strong>Interpretation:</strong> Measures how much later sentence Sₐ <em>depends on</em> earlier sentence Sᵦ</li>
                            </ul>
                            
                            <div style="background: rgba(239,68,68,0.1); padding: 12px; border-radius: 8px; border: 1px solid rgba(239,68,68,0.3); margin-top: 12px;">
                                <strong style="color:#ef4444;">⚠️ Key Difference from BERT:</strong><br>
                                <span style="font-size: 12px;">In GPT-2, sentence A can only attend to sentence B if sentence B appears <em>before</em> sentence A in the text. The ISA matrix is lower-triangular, not symmetric.</span>
                            </div>
                            
                            <p style="font-size: 11px; color: #64748b; margin-top: 16px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1);">📚 Reference: Seo, S., Yoo, S., Lee, H., Jang, Y., Park, J.H., & Kim, J. (2024). "A Sentence-Level Visualization of Attention in Large Language Models." SAVIS: <a href="https://pypi.org/project/savis" target="_blank" style="color:#ff5ca9;">pypi.org/project/savis</a></p>
                        </div>
                    </div>
                `;
            }

            modal.innerHTML = content;
            modal.style.display = 'block';
        };

        function getTransitionExplanation(from, to, modelType) {
            // Dynamic handling for Input -> Token Embeddings based on Model Type
            if (from === 'Input' && to === 'Token Embeddings') {
                if (modelType === 'gpt2') {
                    return `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Tokenization & Embedding Lookup</h4>
                    <p>The input text is first processed by the <strong>Byte-Pair Encoding (BPE) tokenizer</strong>, which breaks words into subword units (tokens) based on frequency patterns. Each token is mapped to a unique integer ID.</p>
                    <p>These IDs are then used to look up dense vectors from the <strong>Token Embedding Matrix</strong> (size: <code>vocab_size × hidden_dim</code>).</p>
                    <p>GPT-2 uses no special start token — generation begins directly from the input. The end-of-text token <code>&lt;|endoftext|&gt;</code> marks sequence boundaries.</p>
                    <ul>
                        <li><strong>Output Shape:</strong> <code>(batch_size, seq_len, hidden_dim)</code></li>
                    </ul>
                    `;
                } else {
                    // Default / BERT
                    return `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Tokenization & Embedding Lookup</h4>
                    <p>The input text is first processed by the <strong>WordPiece tokenizer</strong>, which breaks words into subword units (tokens). Each token is mapped to a unique integer ID.</p>
                    <p>These IDs are then used to look up dense vectors from the <strong>Token Embedding Matrix</strong> (size: <code>vocab_size × hidden_dim</code>).</p>
                    <p>Special tokens <code>[CLS]</code> (start) and <code>[SEP]</code> (separator) are added.</p>
                    <ul>
                        <li><strong>Output Shape:</strong> <code>(batch_size, seq_len, hidden_dim)</code></li>
                    </ul>
                    `;
                }
            }

            // Dynamic handling for Add & Norm (post-FFN) -> Exit
            if (from === 'Add & Norm (post-FFN)' && to === 'Exit') {
                if (modelType === 'gpt2') {
                    return `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Final Prediction</h4>
                    <p>The <strong>Add & Norm</strong> layer outputs the final <strong>Hidden States</strong>.</p>
                    <p>These vectors are then projected to the vocabulary size to predict the <strong>Next Token</strong> in the sequence (Causal Language Modeling).</p>
                    <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:6px; font-family:monospace; margin:10px 0;">P(token_{t+1}) = Softmax(Hidden · W_vocab)</div>
                    `;
                } else {
                    // BERT
                    return `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Final Output & Prediction</h4>
                    <p>The <strong>Add & Norm</strong> layer outputs the final <strong>Hidden States</strong>.</p>
                    <p>In the pre-training phase, these are used for <strong>Masked Token Predictions (MLM)</strong>, reconstructing masked words from context.</p>
                    <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:6px; font-family:monospace; margin:10px 0;">P(mask) = Softmax(Hidden · W_vocab)</div>
                    <p>For fine-tuning tasks (e.g., classification), the [CLS] token's hidden state is typically used.</p>
                    `;
                }
            }

            // Dynamic handling for Token Embeddings -> Positional Embeddings
            if (from === 'Token Embeddings' && to === 'Positional Embeddings') {
                if (modelType === 'gpt2') {
                    return `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Adding Position Information</h4>
                    <p>In GPT-2, <strong>Positional Embeddings</strong> are added directly to the Token Embeddings. Unlike BERT, there are no Segment Embeddings because GPT-2 is typically trained on a continuous stream of text.</p>
                    <ul>
                        <li><strong>Token Embeddings:</strong> Represent the meaning of each word/subword.</li>
                        <li><strong>Positional Embeddings:</strong> Indicate the order of tokens in the sequence.</li>
                        <li><strong>Summation:</strong> The two vectors are added element-wise to create the initial input representation.</li>
                    </ul>
                    `;
                } else {
                    return `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Adding Position & Segment Info</h4>
                    <p>For BERT, the final input representation is the sum of <strong>three</strong> components. Even if the Segment Embeddings details are hidden in this comparison view, they are included in the calculation.</p>
                    <ul>
                        <li><strong>Token Embeddings:</strong> Represent the meaning of each word/subword.</li>
                        <li><strong>Segment Embeddings:</strong> Distinguish between Sentence A and Sentence B.</li>
                        <li><strong>Positional Embeddings:</strong> Indicate the order of tokens.</li>
                    </ul>
                    <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:6px; font-family:monospace; margin:10px 0;">Input = Token + Segment + Position</div>
                    <p>All three vectors are summed element-wise to form the model's input.</p>
                    `;
                }
            }

            const explanations = {
                // 'Input_Token Embeddings' is handled dynamically above
                'Token Embeddings_Segment Embeddings': `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Adding Sentence Context</h4>
                    <p><strong>Segment Embeddings</strong> distinguish between different sentences in the input (e.g., Sentence A vs. Sentence B). This is crucial for tasks like Question Answering or Next Sentence Prediction.</p>
                    <ul>
                        <li><strong>Token Type IDs:</strong> 0 for the first sentence, 1 for the second.</li>
                        <li>Embeddings are looked up from a learned matrix of size <code>2 × hidden_dim</code>.</li>
                        <li>These are added element-wise to the Token Embeddings.</li>
                    </ul>
                `,
                'Segment Embeddings_Positional Embeddings': `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Injecting Position Information</h4>
                    <p>Since the Transformer architecture has no inherent sense of order (unlike RNNs), <strong>Positional Embeddings</strong> are added to give the model information about the absolute position of each token.</p>
                    <ul>
                        <li>Learned embeddings for each position index (0, 1, 2, ...).</li>
                        <li>Matrix size: <code>max_position_embeddings × hidden_dim</code> (typically 512 × 768 for BERT-base).</li>
                        <li>Added element-wise to the previous sum of Token and Segment embeddings.</li>
                    </ul>
                `,
                'Positional Embeddings_Sum & Layer Normalization': `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Combination & Normalization</h4>
                    <p>The final input representation is the sum of the three embedding types:</p>
                    <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:6px; font-family:monospace; margin:10px 0;">Embedding = Token + Segment + Position</div>
                    <p><strong>Layer Normalization</strong> is then applied to stabilize training:</p>
                    <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:6px; font-family:monospace; margin:10px 0;">LN(x) = γ((x - μ) / σ) + β</div>
                    <p>This normalizes the values across the hidden dimension for each token independently.</p>
                `,
                'Sum & Layer Normalization_Q/K/V Projections': `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Linear Projections</h4>
                    <p>The input vectors are projected into three different spaces using learned linear transformations (dense layers) to create <strong>Query (Q)</strong>, <strong>Key (K)</strong>, and <strong>Value (V)</strong> vectors.</p>
                    <ul>
                        <li><strong>Q:</strong> What the token is looking for.</li>
                        <li><strong>K:</strong> What the token "advertises" about itself.</li>
                        <li><strong>V:</strong> The actual content information to be aggregated.</li>
                    </ul>
                    <p>Each projection uses a weight matrix of size <code>hidden_dim × hidden_dim</code>.</p>
                `,
                'Segment Embeddings_Sum & Layer Normalization': `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Input Aggregation</h4>
                    <p>The Segment Embeddings are combined with the other embedding components (Token and Positional) to form the sequence representation.</p>
                    <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:6px; font-family:monospace; margin:10px 0;">Input = Token + Segment + Position</div>
                    <p>All three vectors are summed element-wise, meaning they must occupy the same vector space. The result is then normalized (LayerNorm) to prepare it for the Transformer layers.</p>
                `,
                'Q/K/V Projections_Add & Norm': `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Q/K/V ➜ Scaled Attention ➜ Add & Norm</h4>
                    
                    <strong style="color:#3b82f6; display:block; margin-bottom:4px;">Step 1: Q/K/V ➜ Scaled Dot-Product Attention</strong>
                    <p>The projected Query, Key, and Value vectors are fed into the attention mechanism. The model calculates the compatibility (scores) between Q and K to decide "where to look".</p>
                    <div style="background:rgba(255,255,255,0.05); padding:8px; border-radius:4px; margin:8px 0; font-size:12px;">Scores = (Q · K^T) / √d_k</div>

                    <strong style="color:#3b82f6; display:block; margin-top:16px; margin-bottom:4px;">Step 2: Scaled Dot-Product Attention ➜ Add & Norm</strong>
                    <p>The scores are converted to probabilities and used to aggregate context.</p>
                    <ol>
                        <li><strong>Softmax:</strong> Converts scores to attention weights (probabilities).</li>
                        <li><strong>Weighted Sum:</strong> Multiplies Values (V) by these weights to get the Context Vector.</li>
                        <li><strong>Residual:</strong> The Context Vector is added back to the original input (x + Sublayer(x)).</li>
                        <li><strong>Norm:</strong> The result is normalized (LayerNorm).</li>
                    </ol>
                    <div style="background:rgba(255,255,255,0.05); padding:8px; border-radius:4px; margin:8px 0; font-size:12px;">Output = LayerNorm(x + Attention(Q,K,V))</div>
                `,
                'Scaled Dot-Product Attention_Global Attention Metrics': `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Aggregating Statistics</h4>
                    <p>We compute global metrics across all attention heads and layers to understand the model's behavior. This step doesn't change the data flow but analyzes the attention patterns produced in the previous step.</p>
                    <ul>
                        <li><strong>Entropy:</strong> How focused or diffuse the attention is.</li>
                        <li><strong>Confidence:</strong> The magnitude of the maximum attention weight.</li>
                        <li><strong>Sparsity:</strong> How many tokens receive significant attention.</li>
                    </ul>
                `,
                'Global Attention Metrics_Multi-Head Attention': `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Visualizing Attention Heads</h4>
                    <p>This visualization allows us to inspect the raw attention matrices for individual heads. In <strong>Multi-Head Attention</strong>, the model runs the attention mechanism multiple times in parallel (12 heads for BERT-base).</p>
                    <p>Each head can learn to focus on different relationships (e.g., one head might track next-token relationships, another might track subject-verb dependencies).</p>
                `,
                'Multi-Head Attention_Attention Flow': `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Flow Visualization</h4>
                    <p>The <strong>Attention Flow</strong> view provides a Sankey-style diagram to visualize the flow of information between tokens.</p>
                    <ul>
                        <li><strong>Lines:</strong> Represent attention weights.</li>
                        <li><strong>Thickness:</strong> Proportional to the attention strength.</li>
                        <li><strong>Filtering:</strong> Low-weight connections are often hidden to reduce clutter and reveal the most significant dependencies.</li>
                    </ul>
                `,
                'Attention Flow_Attention Head Specialization': `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Analyzing Head Roles</h4>
                    <p>We analyze the attention patterns to determine what linguistic features each head specializes in. This is done by correlating attention weights with known linguistic properties.</p>
                    <ul>
                        <li><strong>Syntax:</strong> Attention to syntactic dependencies.</li>
                        <li><strong>Positional:</strong> Attention to previous/next tokens.</li>
                        <li><strong>Long-range:</strong> Attention to distant tokens.</li>
                    </ul>
                    <p>The Radar Chart visualizes this "profile" for each head.</p>
                `,
                'Attention Head Specialization_Attention Dependency Tree': `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Hierarchical Influence</h4>
                    <p>The <strong>Dependency Tree</strong> visualizes the chain of influence starting from a selected root token. It shows how attention propagates through the sequence.</p>
                    <ul>
                        <li><strong>Root:</strong> The token being analyzed.</li>
                        <li><strong>Children:</strong> Tokens that the root attends to most strongly.</li>
                        <li><strong>Depth:</strong> Shows multi-hop attention (tokens attending to tokens that attend to the root).</li>
                    </ul>
                `,
                'Attention Dependency Tree_Inter-Sentence Attention': `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Cross-Sentence Analysis</h4>
                    <p><strong>Inter-Sentence Attention (ISA)</strong> specifically isolates and quantifies the attention flowing between the two input sentences (Sentence A and Sentence B).</p>
                    <ul>
                        <li><strong>ISA Score:</strong> Aggregates attention weights where the Source is in one sentence and the Target is in the other.</li>
                        <li>High ISA indicates strong interaction or information exchange between the sentences.</li>
                    </ul>
                `,
                'Positional Embeddings_Sum & Layer Normalization': `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Embedding Summation & Normalization</h4>
                    <p>The three embedding components are element-wise summed:</p>
                    <div class="modal-formula">
                        Input = Token_Embeddings + Segment_Embeddings + Positional_Embeddings
                    </div>
                    <p>This combined representation is then passed through <strong>Layer Normalization</strong> to stabilize training.</p>
                    <ul>
                        <li><strong>Summation:</strong> Combines semantic meaning, sentence segment info, and position info.</li>
                        <li><strong>LayerNorm:</strong> Normalizes the vector for each token to have mean 0 and variance 1.</li>
                    </ul>
                `,
                'Add & Norm (post-FFN)_Hidden States': `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Final Layer Output</h4>
                    <p>After the Feed-Forward Network, another residual connection and normalization step is applied.</p>
                    <div class="modal-formula">
                        Output = LayerNorm(FFN_Output + Input_to_FFN)
                    </div>
                    <p>This produces the final <strong>Hidden States</strong> for the current layer, which are passed to the next layer or used for final predictions.</p>
                `,
                'Hidden States_Token Output Predictions': `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Unembedding & Prediction</h4>
                    <p>The final hidden states are projected back to the vocabulary size to predict the next token (or masked token).</p>
                    <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:6px; font-family:monospace; margin:10px 0;">Logits = LayerNorm(Hidden) · W_vocab + b</div>
                    <p><strong>Softmax</strong> is then applied to convert these logits into probabilities:</p>
                    <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:6px; font-family:monospace; margin:10px 0;">P(token_i) = exp(logit_i) / Σ exp(logit_j)</div>
                `,
                'Inter-Sentence Attention_Add & Norm': `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Residual Connection & Norm</h4>
                    <p>After the attention mechanism, a <strong>Residual (Skip) Connection</strong> adds the original input back to the attention output, followed by another Layer Normalization.</p>
                    <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:6px; font-family:monospace; margin:10px 0;">Output = LayerNorm(x + Attention(x))</div>
                    <p>This allows gradients to flow through the network more easily and preserves information from the lower layers.</p>
                `,
                'Add & Norm_Feed-Forward Network': `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Feed-Forward Processing</h4>
                    <p>The output is passed through a position-wise <strong>Feed-Forward Network (FFN)</strong>.</p>
                    <p>This is where the model "thinks" about the information it has gathered.</p>
                    <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:6px; font-family:monospace; margin:10px 0;">FFN(x) = GELU(xW_1 + b_1)W_2 + b_2</div>
                    <ul>
                        <li><strong>Column 1 ("Intermediate"):</strong> The model expands information into a massive space (3072 dimensions in BERT) to disentangle complex concepts. The heatmap shows these neurons firing.</li>
                        <li><strong>Column 2 ("Projection"):</strong> It compresses this back to the standard size (768 dimensions) to pass it to the next layer.</li>
                    </ul>
                `,
                'Feed-Forward Network_Add & Norm (post-FFN)': `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Second Residual & Norm</h4>
                    <p>A second <strong>Residual Connection</strong> and <strong>Layer Normalization</strong> are applied after the FFN.</p>
                    <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:6px; font-family:monospace; margin:10px 0;">Output = LayerNorm(x + FFN(x))</div>
                    <p>This completes one full Transformer Encoder Block. In BERT-base, this entire process (Attention → Add&Norm → FFN → Add&Norm) repeats 12 times.</p>
                `,
                'Add & Norm (post-FFN)_Hidden States': `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Final Layer Output</h4>
                    <p>The output of the final (12th) encoder layer constitutes the <strong>Hidden States</strong> (or contextualized embeddings) for the sequence.</p>
                    <ul>
                        <li><strong>Shape:</strong> <code>(batch_size, seq_len, hidden_dim)</code></li>
                        <li>These vectors contain rich, contextual information aggregated from all previous layers and are used for downstream tasks or the pre-training objectives.</li>
                    </ul>
                `,
                'Hidden States_Token Output Predictions (MLM)': `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Masked Language Modeling</h4>
                    <p>For the pre-training objective, an <strong>MLM Head</strong> projects the hidden states back to the vocabulary size to predict the original identity of masked tokens.</p>
                    <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:6px; font-family:monospace; margin:10px 0;">Logits = Linear(HiddenStates)</div>
                    <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:6px; font-family:monospace; margin:10px 0;">Probabilities = Softmax(Logits)</div>
                    <p>This gives a probability distribution over the entire vocabulary for each token position.</p>
                `,
                'Hidden States_Next Token Predictions': `
                    <h4 style="color:#ff5ca9; margin-top:0;">What Happens: Causal Language Modeling</h4>
                    <p>The final hidden states are projected to the vocabulary size to predict the <strong>next token</strong> in the sequence.</p>
                    <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:6px; font-family:monospace; margin:10px 0;">P(token_{t+1} | token_1...token_t)</div>
                    <p>This is the core objective of GPT-2: predicting the future based on the past context.</p>
                `,
            };

            var key = from + '_' + to;
            return explanations[key] || '<p>Explanation not available for this transition.</p>';
        }
        """

__all__ = ["JS_CODE", "JS_INTERACTIVE", "JS_TREE_VIZ", "JS_TRANSITION_MODAL"]
