JS_CODE = """
        // Define global functions immediately (before Shiny connects)
        window.switchView = function(containerId, tabName) {
            console.log("switchView called:", containerId, tabName);
            // 1. Hide all panes in this container
            var panes = document.querySelectorAll('#' + containerId + ' .view-pane');
            console.log("Found panes:", panes.length);
            panes.forEach(function(pane) {
                pane.style.display = 'none';
            });
            
            // 2. Show target pane
            var targetId = containerId + '_' + tabName;
            var target = document.getElementById(targetId);
            console.log("Target ID:", targetId, "Target found:", !!target);
            if(target) target.style.display = 'block';
            else console.error("Target NOT found:", targetId);
            
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

        // Download SVG as file (for D3 visualizations)
        window.downloadSVG = function(containerId, filename) {
            var container = document.getElementById(containerId);
            if (!container) {
                console.error('Container not found:', containerId);
                return;
            }

            var svg = container.querySelector('svg');
            if (!svg) {
                console.error('No SVG found in container:', containerId);
                return;
            }

            // Clone the SVG to avoid modifying the original
            var svgClone = svg.cloneNode(true);

            // Add necessary namespaces
            svgClone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
            svgClone.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');

            // Get computed styles and inline them
            var styles = document.createElement('style');
            styles.textContent = `
                text { font-family: 'Inter', sans-serif; }
                .node-circle { stroke-width: 2px; }
                .link { fill: none; }
            `;
            svgClone.insertBefore(styles, svgClone.firstChild);

            // Serialize and download
            var serializer = new XMLSerializer();
            var svgString = serializer.serializeToString(svgClone);
            var blob = new Blob([svgString], {type: 'image/svg+xml;charset=utf-8'});
            var url = URL.createObjectURL(blob);

            var link = document.createElement('a');
            link.href = url;
            link.download = filename || 'visualization.svg';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        };

        // Helper to formatting timestamp
        function getTimestampedFilename(baseName) {
            baseName = baseName || 'export';
            // Remove extension if present to append timestamp before it
            var ext = '';
            if (baseName.lastIndexOf('.') !== -1) {
                ext = baseName.substring(baseName.lastIndexOf('.'));
                baseName = baseName.substring(0, baseName.lastIndexOf('.'));
            }
            
            var now = new Date();
            var year = now.getFullYear();
            var month = String(now.getMonth() + 1).padStart(2, '0');
            var day = String(now.getDate()).padStart(2, '0');
            var hour = String(now.getHours()).padStart(2, '0');
            var min = String(now.getMinutes()).padStart(2, '0');
            var sec = String(now.getSeconds()).padStart(2, '0');
            
            return `${baseName}_${year}-${month}-${day}_${hour}-${min}-${sec}${ext}`;
        }

        // Download D3 SVG as PNG
        window.downloadD3PNG = function(containerId, filename) {
            filename = getTimestampedFilename(filename || 'visualization.png');
            var container = document.getElementById(containerId);
            if (!container) return;
            var svg = container.querySelector('svg');
            if (!svg) return;

            // Clone to avoid modifying
            var svgClone = svg.cloneNode(true);
            svgClone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
            
            // Inline styles (borrowed from downloadSVG)
            var styles = document.createElement('style');
            styles.textContent = `
                text { font-family: 'Inter', sans-serif; }
                .node-circle { stroke-width: 2px; }
                .link { fill: none; }
            `;
            svgClone.insertBefore(styles, svgClone.firstChild);

            // Serialize
            var serializer = new XMLSerializer();
            var svgString = serializer.serializeToString(svgClone);
            
            // Create Image
            var img = new Image();
            var svgBlob = new Blob([svgString], {type: 'image/svg+xml;charset=utf-8'});
            var url = URL.createObjectURL(svgBlob);
            
            img.onload = function() {
                var canvas = document.createElement('canvas');
                // Use actual SVG attributes for dimensions, or falling back to bbox
                var w = parseInt(svg.getAttribute('width')) || svg.getBoundingClientRect().width || 800;
                var h = parseInt(svg.getAttribute('height')) || svg.getBoundingClientRect().height || 600;

                // Ensure minimum dimensions
                if (w < 100) w = 800;
                if (h < 100) h = 600;

                canvas.width = w * 2; // 2x density for sharpness
                canvas.height = h * 2;
                var ctx = canvas.getContext('2d');
                ctx.scale(2, 2);

                // White background
                ctx.fillStyle = '#ffffff';
                ctx.fillRect(0, 0, w, h);

                // Draw image at original size (scale is already applied via ctx.scale)
                ctx.drawImage(img, 0, 0, w, h);

                var pngUrl = canvas.toDataURL('image/png');

                var link = document.createElement('a');
                link.href = pngUrl;
                link.download = filename || 'visualization.png';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                URL.revokeObjectURL(url);

                // Save a copy to the server-side images/ folder
                if (typeof Shiny !== 'undefined' && Shiny.setInputValue) {
                    Shiny.setInputValue('_save_png', {
                        filename: filename || 'visualization.png',
                        data: pngUrl
                    }, {priority: 'event'});
                }
            };
            img.onerror = function(e) {
                console.error('Error loading SVG for PNG conversion:', e);
                URL.revokeObjectURL(url);
            };
            img.src = url;
        };

        // Download Plotly chart as PNG
        window.downloadPlotlyPNG = function(containerId, filename) {
            filename = getTimestampedFilename(filename || 'chart.png');
            var container = document.getElementById(containerId);
            if (!container) {
                console.error('Container not found:', containerId);
                return;
            }

            var plotlyDiv = container.querySelector('.js-plotly-plot');
            if (!plotlyDiv) {
                plotlyDiv = container.querySelector('[class*="plotly"]');
            }
            if (!plotlyDiv) {
                console.error('No Plotly chart found in container:', containerId);
                return;
            }

            Plotly.toImage(plotlyDiv, {
                format: 'png',
                width: 1200,
                height: 800
            }).then(function(dataUrl) {
                var img = new Image();
                img.onload = function() {
                    var canvas = document.createElement('canvas');
                    canvas.width = img.width;
                    canvas.height = img.height;
                    var ctx = canvas.getContext('2d');
                    ctx.fillStyle = '#ffffff';
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0);
                    
                    var finalUrl = canvas.toDataURL('image/png');
                    var link = document.createElement('a');
                    link.href = finalUrl;
                    link.download = filename || 'chart.png';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);

                    // Save a copy to the server-side images/ folder
                    if (typeof Shiny !== 'undefined' && Shiny.setInputValue) {
                        Shiny.setInputValue('_save_png', {
                            filename: filename || 'chart.png',
                            data: finalUrl
                        }, {priority: 'event'});
                    }
                };
                img.src = dataUrl;
            }).catch(function(error) {
                console.error('Error exporting plot:', error);
            });
        };

        // Session restoration handler - restores custom textareas
        $(document).ready(function() {
            Shiny.addCustomMessageHandler('restore_session_text', function(data) {
                if (data.text_input) {
                    var textInput = document.getElementById('text_input');
                    if (textInput) {
                        textInput.value = data.text_input;
                        Shiny.setInputValue('text_input', data.text_input, {priority: 'event'});
                    }
                }
                if (data.text_input_B) {
                    var textInputB = document.getElementById('text_input_B');
                    if (textInputB) {
                        textInputB.value = data.text_input_B;
                        Shiny.setInputValue('text_input_B', data.text_input_B, {priority: 'event'});
                    }
                }
            });

            // Restore custom control items (sliders/radio)
            Shiny.addCustomMessageHandler('restore_session_controls', function(data) {
                console.log("Restoring controls:", data);
                
                // Restore Layer
                if (data.layer !== undefined) {
                    Shiny.setInputValue('global_layer', data.layer, {priority: 'event'});
                    // Update visual slider if it exists (might not if controls hidden)
                    var layerSlider = document.getElementById('layer-slider');
                    if (layerSlider) {
                        layerSlider.value = data.layer;
                        var layerVal = document.getElementById('layer-value');
                        if (layerVal) layerVal.textContent = data.layer;
                    }
                }
                
                // Restore Head
                if (data.head !== undefined) {
                    Shiny.setInputValue('global_head', data.head, {priority: 'event'});
                    var headSlider = document.getElementById('head-slider');
                    if (headSlider) {
                        headSlider.value = data.head;
                        var headVal = document.getElementById('head-value');
                        if (headVal) headVal.textContent = data.head;
                    }
                }
                
                // Restore TopK
                if (data.topk !== undefined) {
                     Shiny.setInputValue('global_topk', data.topk, {priority: 'event'});
                     var topkSlider = document.getElementById('topk-slider');
                     if (topkSlider) {
                         topkSlider.value = data.topk;
                         var topkVal = document.getElementById('topk-value');
                         if (topkVal) topkVal.textContent = data.topk;
                     }
                }
                
                // Restore Norm
                if (data.norm !== undefined) {
                    Shiny.setInputValue('global_norm', data.norm, {priority: 'event'});
                    // Update visual radio button state
                    var normGroup = document.getElementById('norm-radio-group');
                    if (normGroup) {
                        var buttons = normGroup.querySelectorAll('.radio-option');
                        buttons.forEach(function(btn) {
                            btn.classList.remove('active');
                            if (btn.getAttribute('data-value') === data.norm) {
                                btn.classList.add('active');
                            }
                        });
                        
                        // Handle rollout visibility
                        var rolloutControl = document.getElementById('rollout-layers-control');
                        if (rolloutControl) {
                             if (data.norm === 'rollout') rolloutControl.classList.add('visible');
                             else rolloutControl.classList.remove('visible');
                        }
                    }
                }
            });

            // Trigger generation programmatically
            Shiny.addCustomMessageHandler('trigger_generate', function(message) {
                console.log("Triggering generation from server...");
                var btn = document.getElementById('generate_all');
                if (btn) {
                    btn.click();
                } else {
                    console.error("Generate button not found");
                }
            });
        });

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
            var btn = $('#analyze_bias_btn');
            if (btn.length) {
                if (!btn.data('original-content')) {
                    btn.data('original-content', btn.html());
                }
                btn.html('<div class="spinner" style="width:14px;height:14px;border-width:2px;display:inline-block;vertical-align:middle;margin-right:8px;"></div>Analyzing<span class="loading-dots"></span>');
                btn.prop('disabled', true).css('opacity', '0.8');
            }
        });
 
        Shiny.addCustomMessageHandler('stop_bias_loading', function(msg) {
            var btn = $('#analyze_bias_btn');
            if (btn.length) {
                if (btn.data('original-content')) {
                    btn.html(btn.data('original-content'));
                } else {
                    btn.html('Analyze Bias');
                }
                btn.prop('disabled', false).css('opacity', '1');
            }
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
                    <h4 style="color:#ff5ca9; margin-top:0;">Tokenization & Embedding Lookup</h4>
                    <p>GPT-2 uses <strong>Byte-Pair Encoding (BPE)</strong>, a compression-based subword algorithm operating on UTF-8 byte sequences. This ensures complete coverage of any input without unknown tokens.</p>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0; border-left:3px solid #3b82f6;">
                        <strong style="color:#3b82f6;">BPE Algorithm:</strong><br>
                        <span style="font-size:12px; color:#94a3b8;">1. Initialize with 256 byte tokens → 2. Merge most frequent pairs → 3. Repeat to 50,257 tokens</span>
                    </div>

                    <p>Token IDs index into the <strong>Token Embedding Matrix</strong> <code>E ∈ ℝ<sup>50,257 × 768</sup></code>.</p>

                    <ul style="font-size:13px;">
                        <li><strong>Special Token:</strong> <code>&lt;|endoftext|&gt;</code> marks sequence boundaries</li>
                        <li><strong>No start token</strong> — generation begins directly from context</li>
                        <li><strong>Output:</strong> <code>(batch, seq_len, 768)</code> — context-independent vectors</li>
                    </ul>
                    `;
                } else {
                    // Default / BERT
                    return `
                    <h4 style="color:#ff5ca9; margin-top:0;">Tokenization & Embedding Lookup</h4>
                    <p>BERT uses <strong>WordPiece tokenization</strong>, a subword segmentation that balances vocabulary size with rare word coverage.</p>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0; border-left:3px solid #3b82f6;">
                        <strong style="color:#3b82f6;">WordPiece Process:</strong><br>
                        <span style="font-size:12px; color:#94a3b8;">1. Normalize text → 2. Split by whitespace → 3. Decompose into subwords (## prefix for continuations)</span><br>
                        <span style="font-size:11px; color:#64748b;">Example: "unbelievable" → ["un", "##believ", "##able"]</span>
                    </div>

                    <p>Token IDs index into the <strong>Token Embedding Matrix</strong> <code>E ∈ ℝ<sup>30,522 × 768</sup></code>.</p>

                    <ul style="font-size:13px;">
                        <li><strong>[CLS]:</strong> Prepended to every sequence — aggregates sequence-level info</li>
                        <li><strong>[SEP]:</strong> Appended after each sentence — demarcates boundaries</li>
                        <li><strong>Output:</strong> <code>(batch, seq_len, 768)</code> — context-independent vectors</li>
                    </ul>
                    `;
                }
            }

            // Dynamic handling for Add & Norm (post-FFN) -> Exit
            if (from === 'Add & Norm (post-FFN)' && to === 'Exit') {
                if (modelType === 'gpt2') {
                    return `
                    <h4 style="color:#ff5ca9; margin-top:0;">Hidden States → Token Predictions (Causal LM)</h4>
                    <p>The final layer outputs <strong>contextualized hidden states</strong> — dense vectors encoding both the token's meaning and information gathered from all preceding tokens.</p>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0; border-left:3px solid #f97316;">
                        <strong style="color:#f97316;">Autoregressive Prediction:</strong><br>
                        <code style="font-size:11px;">P(token<sub>t+1</sub> | token<sub>1</sub>...token<sub>t</sub>) = Softmax(H<sub>t</sub> · W<sub>vocab</sub>)</code>
                    </div>

                    <ul style="font-size:13px;">
                        <li><strong>Generation:</strong> Sample next token → append → repeat</li>
                        <li><strong>Loss:</strong> Cross-entropy summed over all positions</li>
                        <li>Each token can only see <em>left context</em> (causal mask)</li>
                    </ul>
                    `;
                } else {
                    // BERT
                    return `
                    <h4 style="color:#ff5ca9; margin-top:0;">Hidden States → Token Predictions (MLM)</h4>
                    <p>The final layer outputs <strong>contextualized hidden states</strong> — dense vectors encoding both the token's meaning and information from the entire sequence (bidirectional).</p>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0; border-left:3px solid #3b82f6;">
                        <strong style="color:#3b82f6;">MLM Head Architecture:</strong><br>
                        <span style="font-size:11px; color:#94a3b8;">Hidden → Linear → GELU → LayerNorm → Projection → Softmax</span>
                    </div>

                    <ul style="font-size:13px;">
                        <li><strong>Pre-training:</strong> 15% tokens masked (80% [MASK], 10% random, 10% unchanged)</li>
                        <li><strong>Fine-tuning:</strong> [CLS] hidden state used for classification tasks</li>
                        <li><strong>W<sub>vocab</sub></strong> often shared (tied) with input embeddings</li>
                    </ul>
                    `;
                }
            }

            // Dynamic handling for Token Embeddings -> Positional Embeddings
            if (from === 'Token Embeddings' && to === 'Positional Embeddings') {
                if (modelType === 'gpt2') {
                    return `
                    <h4 style="color:#ff5ca9; margin-top:0;">Adding Positional Information</h4>
                    <p>Self-attention is <strong>permutation-equivariant</strong> — it has no inherent sense of token order. Positional embeddings inject sequence position information.</p>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0; border-left:3px solid #f97316;">
                        <strong style="color:#f97316;">GPT-2 Embedding Sum:</strong><br>
                        <code style="font-size:12px;">Input[i] = Token_Embed[i] + Position_Embed[i]</code>
                    </div>

                    <ul style="font-size:13px;">
                        <li><strong>Position Matrix:</strong> <code>E<sub>pos</sub> ∈ ℝ<sup>1024 × 768</sup></code> (learned, not sinusoidal)</li>
                        <li><strong>No Segment Embeddings:</strong> GPT-2 processes continuous text streams</li>
                        <li><strong>Max length:</strong> Fixed at 1024 tokens (architectural limit)</li>
                    </ul>
                    `;
                } else {
                    return `
                    <h4 style="color:#ff5ca9; margin-top:0;">Adding Position & Segment Information</h4>
                    <p>Self-attention is <strong>permutation-equivariant</strong> — it has no inherent sense of token order. BERT combines three embedding types.</p>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0; border-left:3px solid #3b82f6;">
                        <strong style="color:#3b82f6;">BERT Embedding Sum:</strong><br>
                        <code style="font-size:12px;">Input[i] = Token[i] + Segment[i] + Position[i]</code>
                    </div>

                    <ul style="font-size:13px;">
                        <li><strong>Token:</strong> Semantic meaning from vocabulary</li>
                        <li><strong>Segment:</strong> Sentence A (0) vs Sentence B (1)</li>
                        <li><strong>Position:</strong> Absolute index (0, 1, 2...511)</li>
                    </ul>
                    <p style="font-size:12px; color:#94a3b8;">All three matrices are learned during pre-training and summed element-wise.</p>
                    `;
                }
            }

            const explanations = {
                // 'Input_Token Embeddings' is handled dynamically above
                'Token Embeddings_Segment Embeddings': `
                    <h4 style="color:#ff5ca9; margin-top:0;">Adding Sentence Membership (BERT only)</h4>
                    <p><strong>Segment Embeddings</strong> provide explicit information about which sentence each token belongs to — essential for sentence-pair tasks.</p>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0; border-left:3px solid #3b82f6;">
                        <strong style="color:#3b82f6;">Token Type Assignment:</strong><br>
                        <span style="font-size:11px; color:#94a3b8;">[CLS] What is AI? [SEP] AI is... [SEP]</span><br>
                        <span style="font-size:11px; color:#64748b;">&nbsp;&nbsp;0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0&nbsp;&nbsp;&nbsp;0&nbsp;&nbsp;0&nbsp;&nbsp;&nbsp;0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp;&nbsp;1</span>
                    </div>

                    <ul style="font-size:13px;">
                        <li><strong>Segment Matrix:</strong> <code>E<sub>seg</sub> ∈ ℝ<sup>2 × 768</sup></code></li>
                        <li><strong>ID 0:</strong> Sentence A (incl. [CLS] and first [SEP])</li>
                        <li><strong>ID 1:</strong> Sentence B (incl. final [SEP])</li>
                    </ul>
                    <p style="font-size:12px; color:#94a3b8;">GPT-2 omits this — it processes continuous text without explicit boundaries.</p>
                `,
                'Segment Embeddings_Positional Embeddings': `
                    <h4 style="color:#ff5ca9; margin-top:0;">Injecting Position Information</h4>
                    <p>Transformers are <strong>permutation-equivariant</strong> — unlike RNNs, they process all tokens in parallel with no inherent ordering. Positional embeddings solve this.</p>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0; border-left:3px solid #8b5cf6;">
                        <strong style="color:#8b5cf6;">Position Embedding Lookup:</strong><br>
                        <code style="font-size:11px;">Position_Embed[i] = E<sub>position</sub>[i]</code><br>
                        <span style="font-size:11px; color:#94a3b8;">where i is the absolute position index (0, 1, 2, ...)</span>
                    </div>

                    <ul style="font-size:13px;">
                        <li><strong>Matrix:</strong> <code>E<sub>pos</sub> ∈ ℝ<sup>512 × 768</sup></code> (BERT-base)</li>
                        <li><strong>Learned</strong> (not sinusoidal) — optimized during pre-training</li>
                        <li><strong>Limitation:</strong> Cannot generalize beyond max_seq_len</li>
                    </ul>
                `,
                'Positional Embeddings_Sum & Layer Normalization': `
                    <h4 style="color:#ff5ca9; margin-top:0;">Embedding Aggregation & Layer Normalization</h4>
                    <p>All embedding components are summed element-wise, then normalized to stabilize training and improve gradient flow.</p>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0; border-left:3px solid #22c55e;">
                        <strong style="color:#22c55e;">Layer Normalization:</strong><br>
                        <code style="font-size:11px;">LN(x) = γ ⊙ (x - μ) / √(σ² + ε) + β</code><br>
                        <span style="font-size:11px; color:#94a3b8;">μ, σ² computed across hidden dim; γ, β are learned</span>
                    </div>

                    <ul style="font-size:13px;">
                        <li><strong>Effect:</strong> Each token vector → ~zero mean, ~unit variance</li>
                        <li><strong>γ (scale):</strong> Initialized to 1</li>
                        <li><strong>β (shift):</strong> Initialized to 0</li>
                        <li><strong>ε:</strong> Small constant (1e-12) for numerical stability</li>
                    </ul>
                    <p style="font-size:12px; color:#94a3b8;">Unlike BatchNorm, LayerNorm works on each sample independently — ideal for variable-length sequences.</p>
                `,
                'Sum & Layer Normalization_Q/K/V Projections': `
                    <h4 style="color:#ff5ca9; margin-top:0;">Query, Key, Value Projections</h4>
                    <p>The input is projected into three specialized representation spaces using learned linear transformations.</p>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0; font-family:monospace; font-size:11px;">
                        Q = X · W<sub>Q</sub> + b<sub>Q</sub><br>
                        K = X · W<sub>K</sub> + b<sub>K</sub><br>
                        V = X · W<sub>V</sub> + b<sub>V</sub>
                    </div>

                    <ul style="font-size:13px;">
                        <li><strong>Query (Q):</strong> "What information am I looking for?"</li>
                        <li><strong>Key (K):</strong> "What information can I provide?"</li>
                        <li><strong>Value (V):</strong> "Here is my actual content"</li>
                    </ul>

                    <p style="font-size:12px; color:#94a3b8;"><strong>Weight matrices:</strong> W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub> ∈ ℝ<sup>768 × 768</sup><br>
                    The separation allows different transformations for "searching" vs "being found" vs "providing content".</p>
                `,
                'Segment Embeddings_Sum & Layer Normalization': `
                    <h4 style="color:#ff5ca9; margin-top:0;">Embedding Aggregation & Normalization</h4>
                    <p>All three embedding components are summed into a single representation, then normalized.</p>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0; border-left:3px solid #3b82f6;">
                        <strong style="color:#3b82f6;">BERT Input Formula:</strong><br>
                        <code style="font-size:11px;">Input[i] = E<sub>token</sub>[i] + E<sub>segment</sub>[i] + E<sub>position</sub>[i]</code>
                    </div>

                    <ul style="font-size:13px;">
                        <li>All embeddings share the same dimension (768)</li>
                        <li>Element-wise addition preserves vector space structure</li>
                        <li><strong>LayerNorm</strong> stabilizes values before attention</li>
                    </ul>

                    <p style="font-size:12px; color:#94a3b8;">The summed representation is the model's initial understanding — context-independent at this stage. Contextualization happens in subsequent attention layers.</p>
                `,
                'Q/K/V Projections_Add & Norm': `
                    <h4 style="color:#ff5ca9; margin-top:0;">Scaled Dot-Product Attention → Residual → Norm</h4>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0; border-left:3px solid #3b82f6;">
                        <strong style="color:#3b82f6;">Step 1: Compute Attention Scores</strong><br>
                        <code style="font-size:11px;">Scores = Q · K<sup>T</sup> / √d<sub>k</sub></code><br>
                        <span style="font-size:11px; color:#94a3b8;">Scaling by √d<sub>k</sub> prevents softmax saturation for large d<sub>k</sub></span>
                    </div>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0; border-left:3px solid #8b5cf6;">
                        <strong style="color:#8b5cf6;">Step 2: Apply Mask (GPT-2 only)</strong><br>
                        <span style="font-size:11px; color:#94a3b8;">Scores[i,j] = -∞ if j > i (future tokens masked)</span>
                    </div>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0; border-left:3px solid #22c55e;">
                        <strong style="color:#22c55e;">Step 3: Softmax → Weighted Sum → Residual</strong><br>
                        <code style="font-size:11px;">Attn = Softmax(Scores) · V</code><br>
                        <code style="font-size:11px;">Output = LayerNorm(X + Attn)</code>
                    </div>

                    <p style="font-size:12px; color:#94a3b8;"><strong>Residual connection:</strong> Allows gradients to flow directly backward, mitigating vanishing gradients in deep networks.</p>
                `,
                'Scaled Dot-Product Attention_Global Attention Metrics': `
                    <h4 style="color:#ff5ca9; margin-top:0;">Attention Analysis (Interpretability)</h4>
                    <p>Quantitative measures computed across attention distributions — <em>does not alter computation</em>, only analyzes patterns.</p>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0;">
                        <strong style="color:#94a3b8; font-size:12px;">Key Metrics:</strong><br>
                        <span style="font-size:11px;"><strong style="color:#22c55e;">Entropy:</strong> H(A) = -Σ A[i,j]·log(A[i,j]) — high = diffuse, low = focused</span><br>
                        <span style="font-size:11px;"><strong style="color:#3b82f6;">Confidence:</strong> max(A[i,:]) — strength of strongest connection</span><br>
                        <span style="font-size:11px;"><strong style="color:#f59e0b;">Sparsity:</strong> % of weights below threshold τ</span>
                    </div>

                    <p style="font-size:12px; color:#94a3b8;">These metrics describe distribution <em>shape</em> but don't indicate whether attention is "correct" or task-relevant.</p>
                `,
                'Global Attention Metrics_Multi-Head Attention': `
                    <h4 style="color:#ff5ca9; margin-top:0;">Multi-Head Attention Visualization</h4>
                    <p>Multiple attention functions run in parallel, each potentially learning different relationship types.</p>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0; border-left:3px solid #3b82f6;">
                        <strong style="color:#3b82f6;">Multi-Head Formula:</strong><br>
                        <code style="font-size:11px;">MultiHead(Q,K,V) = Concat(head<sub>1</sub>,...,head<sub>h</sub>) · W<sub>O</sub></code><br>
                        <span style="font-size:11px; color:#94a3b8;">h=12 heads, each with d<sub>k</sub>=64 dims (768/12)</span>
                    </div>

                    <ul style="font-size:13px;">
                        <li><strong>Positional heads:</strong> Attend to adjacent tokens</li>
                        <li><strong>Syntactic heads:</strong> Follow dependency structure</li>
                        <li><strong>Delimiter heads:</strong> Attend to [CLS]/[SEP]</li>
                        <li><strong>Long-range heads:</strong> Connect distant tokens</li>
                    </ul>
                `,
                'Multi-Head Attention_Attention Flow': `
                    <h4 style="color:#ff5ca9; margin-top:0;">Attention Flow (Sankey Diagram)</h4>
                    <p>Directed graph visualization showing information flow between tokens based on attention weights.</p>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0;">
                        <strong style="color:#94a3b8; font-size:12px;">Visual Encoding:</strong><br>
                        <span style="font-size:11px;"><strong style="color:#22c55e;">●</strong> Nodes = tokens in sequence</span><br>
                        <span style="font-size:11px;"><strong style="color:#3b82f6;">―</strong> Edges = attention weights (Query→Key)</span><br>
                        <span style="font-size:11px;"><strong style="color:#f59e0b;">━</strong> Edge thickness ∝ weight magnitude</span>
                    </div>

                    <ul style="font-size:13px;">
                        <li><strong>Threshold:</strong> Connections below 0.04 filtered out</li>
                        <li><strong>BERT:</strong> Bidirectional edges (both directions)</li>
                        <li><strong>GPT-2:</strong> Leftward edges only (causal mask)</li>
                    </ul>

                    <p style="font-size:12px; color:#94a3b8;">⚠️ Shows Query→Key relationships, not true information flow or causal influence.</p>
                `,
                'Attention Flow_Attention Head Specialization': `
                    <h4 style="color:#ff5ca9; margin-top:0;">Head Specialization Analysis</h4>
                    <p>Attention patterns are analyzed to identify linguistic features captured by individual heads — correlating weights with annotated structures.</p>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0;">
                        <strong style="color:#94a3b8; font-size:12px;">7 Analysis Dimensions:</strong><br>
                        <span style="font-size:11px;"><strong style="color:#3b82f6;">Syntax:</strong> Overlap with dependency edges</span><br>
                        <span style="font-size:11px;"><strong style="color:#22c55e;">Semantics:</strong> Attention to content words</span><br>
                        <span style="font-size:11px;"><strong style="color:#f59e0b;">CLS Focus:</strong> Attention to [CLS] token</span><br>
                        <span style="font-size:11px;"><strong style="color:#8b5cf6;">Punctuation:</strong> Attention to delimiters</span><br>
                        <span style="font-size:11px;"><strong style="color:#ef4444;">Entities:</strong> Attention to named entities</span><br>
                        <span style="font-size:11px;"><strong style="color:#06b6d4;">Long-range:</strong> Mean attention distance</span><br>
                        <span style="font-size:11px;"><strong style="color:#ec4899;">Self-attention:</strong> Diagonal attention strength</span>
                    </div>

                    <p style="font-size:12px; color:#94a3b8;">⚠️ POS-based heuristic — heads may capture patterns not aligned with traditional categories.</p>
                `,
                'Attention Head Specialization_Attention Dependency Tree': `
                    <h4 style="color:#ff5ca9; margin-top:0;">Attention Dependency Tree</h4>
                    <p>Hierarchical tree visualization rooted at a selected token, showing attention propagation through the sequence.</p>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0; border-left:3px solid #8b5cf6;">
                        <strong style="color:#8b5cf6;">Construction Algorithm:</strong><br>
                        <span style="font-size:11px;">1. Select root token of interest</span><br>
                        <span style="font-size:11px;">2. Find top-k tokens root attends to (children)</span><br>
                        <span style="font-size:11px;">3. Recursively find what children attend to</span><br>
                        <span style="font-size:11px;">4. Continue to desired depth</span>
                    </div>

                    <ul style="font-size:13px;">
                        <li><strong>Node size:</strong> Proportional to attention weight</li>
                        <li><strong>Edge labels:</strong> Show exact attention values</li>
                        <li><strong>Depth:</strong> Reveals transitive information flow</li>
                    </ul>

                    <p style="font-size:12px; color:#94a3b8;">⚠️ Tree structure imposes hierarchy on non-hierarchical attention — multiple strong connections may be underrepresented.</p>
                `,
                'Attention Dependency Tree_Inter-Sentence Attention': `
                    <h4 style="color:#ff5ca9; margin-top:0;">Inter-Sentence Attention (ISA)</h4>
                    <p>Quantifies attention flow between distinct text segments — reduces token-level complexity O(n²) to sentence-level O(m²).</p>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0; border-left:3px solid #ff5ca9;">
                        <strong style="color:#ff5ca9;">Three-Level Max Pooling:</strong><br>
                        <code style="font-size:10px;">ISA(S<sub>a</sub>,S<sub>b</sub>) = max<sub>heads</sub>(max<sub>tokens∈S<sub>a</sub>×S<sub>b</sub></sub>(max<sub>layers</sub>(α<sub>ij</sub>)))</code>
                    </div>

                    <ul style="font-size:13px;">
                        <li><strong>&gt;0.8:</strong> Strong cross-sentence coupling</li>
                        <li><strong>0.4-0.8:</strong> Moderate interaction</li>
                        <li><strong>&lt;0.4:</strong> Independent processing</li>
                    </ul>

                    <p style="font-size:12px; color:#94a3b8;"><strong>BERT:</strong> Symmetric matrix (bidirectional)<br>
                    <strong>GPT-2:</strong> Lower triangular (causal — later→earlier only)</p>
                `,
                'Positional Embeddings_Sum & Layer Normalization': `
                    <h4 style="color:#ff5ca9; margin-top:0;">Embedding Aggregation & Layer Normalization</h4>
                    <p>All embedding components are summed element-wise, then normalized to stabilize training and improve gradient flow.</p>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0; border-left:3px solid #22c55e;">
                        <strong style="color:#22c55e;">Layer Normalization:</strong><br>
                        <code style="font-size:11px;">LN(x) = γ ⊙ (x - μ) / √(σ² + ε) + β</code><br>
                        <span style="font-size:11px; color:#94a3b8;">μ, σ² computed across hidden dim; γ, β are learned</span>
                    </div>

                    <ul style="font-size:13px;">
                        <li><strong>Effect:</strong> Each token vector → ~zero mean, ~unit variance</li>
                        <li><strong>γ (scale):</strong> Initialized to 1</li>
                        <li><strong>β (shift):</strong> Initialized to 0</li>
                        <li><strong>ε:</strong> Small constant (1e-12) for numerical stability</li>
                    </ul>
                    <p style="font-size:12px; color:#94a3b8;">Unlike BatchNorm, LayerNorm works on each sample independently — ideal for variable-length sequences.</p>
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
                    <h4 style="color:#ff5ca9; margin-top:0;">Unembedding & Token Prediction</h4>
                    <p>Hidden states are projected to vocabulary size to produce token probability distributions.</p>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0; border-left:3px solid #8b5cf6;">
                        <strong style="color:#8b5cf6;">Prediction Pipeline:</strong><br>
                        <code style="font-size:11px;">Logits = H · W<sub>vocab</sub> + b</code><br>
                        <code style="font-size:11px;">P(token<sub>i</sub>) = exp(logit<sub>i</sub>) / Σ<sub>j</sub> exp(logit<sub>j</sub>)</code>
                    </div>

                    <ul style="font-size:13px;">
                        <li><strong>Logits:</strong> Raw scores for each vocabulary token</li>
                        <li><strong>Softmax:</strong> Normalizes to probability distribution</li>
                        <li><strong>W<sub>vocab</sub>:</strong> Often shared (tied) with input embeddings</li>
                    </ul>

                    <p style="font-size:12px; color:#94a3b8;">The model outputs a distribution over 30k+ tokens — the predicted token is typically the argmax (greedy) or sampled from this distribution.</p>
                `,
                'Inter-Sentence Attention_Add & Norm': `
                    <h4 style="color:#ff5ca9; margin-top:0;">Residual Connection & Layer Normalization</h4>
                    <p>After attention computation, the original input is added back (skip connection) and normalized.</p>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0; border-left:3px solid #22c55e;">
                        <strong style="color:#22c55e;">Residual Formula:</strong><br>
                        <code style="font-size:11px;">Output = LayerNorm(X + Attention(X))</code>
                    </div>

                    <ul style="font-size:13px;">
                        <li><strong>Skip connection:</strong> Provides direct gradient path, mitigates vanishing gradients</li>
                        <li><strong>Identity mapping:</strong> Layers learn <em>modifications</em> to identity, not complete transforms</li>
                        <li><strong>Information preservation:</strong> Earlier layer info remains accessible</li>
                    </ul>

                    <p style="font-size:12px; color:#94a3b8;">This "post-norm" pattern (residual + LN) is applied consistently throughout BERT/GPT-2.</p>
                `,
                'Add & Norm_Feed-Forward Network': `
                    <h4 style="color:#ff5ca9; margin-top:0;">Position-wise Feed-Forward Network</h4>
                    <p>Applied <strong>independently to each token</strong> — provides non-linear transformation capacity beyond what attention offers.</p>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0; border-left:3px solid #f59e0b;">
                        <strong style="color:#f59e0b;">FFN Architecture:</strong><br>
                        <code style="font-size:11px;">FFN(x) = GELU(x · W<sub>1</sub> + b<sub>1</sub>) · W<sub>2</sub> + b<sub>2</sub></code>
                    </div>

                    <ul style="font-size:13px;">
                        <li><strong>Expansion:</strong> 768 → 3072 dims (4× bottleneck)</li>
                        <li><strong>Activation:</strong> GELU ≈ x · Φ(x) — smooth approximation of ReLU</li>
                        <li><strong>Projection:</strong> 3072 → 768 dims</li>
                    </ul>

                    <div style="background:rgba(139,92,246,0.1); padding:10px; border-radius:6px; margin-top:12px; border:1px solid rgba(139,92,246,0.3);">
                        <strong style="color:#8b5cf6; font-size:12px;">💡 Interpretability insight:</strong><br>
                        <span style="font-size:11px;">FFN neurons often correspond to interpretable concepts — functioning as a form of key-value memory storing factual knowledge.</span>
                    </div>
                `,
                'Feed-Forward Network_Add & Norm (post-FFN)': `
                    <h4 style="color:#ff5ca9; margin-top:0;">Second Residual Connection & Normalization</h4>
                    <p>Completes the Transformer block with another skip connection and normalization.</p>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0; border-left:3px solid #22c55e;">
                        <strong style="color:#22c55e;">Post-FFN Formula:</strong><br>
                        <code style="font-size:11px;">X<sub>out</sub> = LayerNorm(X<sub>in</sub> + FFN(X<sub>in</sub>))</code>
                    </div>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0;">
                        <strong style="color:#94a3b8; font-size:12px;">Complete Transformer Block:</strong><br>
                        <span style="font-size:11px;">X<sub>1</sub> = LayerNorm(X<sub>0</sub> + MultiHeadAttn(X<sub>0</sub>))</span><br>
                        <span style="font-size:11px;">X<sub>2</sub> = LayerNorm(X<sub>1</sub> + FFN(X<sub>1</sub>))</span>
                    </div>

                    <p style="font-size:12px; color:#94a3b8;"><strong>Layer Stacking:</strong> This block repeats N times (N=12 for base, N=24 for large). Each layer operates on the previous layer's output: X<sup>(l)</sup> = Block(X<sup>(l-1)</sup>)</p>
                `,
                'Add & Norm (post-FFN)_Hidden States': `
                    <h4 style="color:#ff5ca9; margin-top:0;">Final Hidden States</h4>
                    <p>The output of the final Transformer layer — dense, <strong>contextualized representations</strong> encoding both token meaning and information gathered from the entire sequence.</p>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0;">
                        <strong style="color:#94a3b8; font-size:12px;">Hidden State Properties:</strong><br>
                        <span style="font-size:11px;">• Shape: <code>(batch, seq_len, 768)</code></span><br>
                        <span style="font-size:11px;">• Each H[i] = token i in full context</span><br>
                        <span style="font-size:11px;">• Earlier layers → local/syntactic info</span><br>
                        <span style="font-size:11px;">• Later layers → global/semantic info</span>
                    </div>

                    <ul style="font-size:13px;">
                        <li><strong>Token tasks:</strong> Use H[i] directly (NER, POS tagging)</li>
                        <li><strong>Sequence tasks:</strong> Use H<sub>[CLS]</sub> (BERT) or H<sub>final</sub> (GPT-2)</li>
                        <li><strong>Generation:</strong> Use H to predict next tokens</li>
                    </ul>
                `,
                'Hidden States_Token Output Predictions (MLM)': `
                    <h4 style="color:#ff5ca9; margin-top:0;">Masked Language Model Predictions</h4>
                    <p>BERT's pre-training objective: predict the original identity of masked tokens from bidirectional context.</p>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0; border-left:3px solid #3b82f6;">
                        <strong style="color:#3b82f6;">MLM Head Pipeline:</strong><br>
                        <span style="font-size:11px;">Hidden → Linear(768→768) → GELU → LayerNorm → W<sub>vocab</sub>(768→30k) → Softmax</span>
                    </div>

                    <ul style="font-size:13px;">
                        <li><strong>Masking strategy:</strong> 15% of tokens selected</li>
                        <li>80% → [MASK], 10% → random, 10% → unchanged</li>
                        <li><strong>Loss:</strong> Cross-entropy over masked positions only</li>
                        <li><strong>W<sub>vocab</sub></strong> often tied with input embeddings</li>
                    </ul>
                `,
                'Hidden States_Next Token Predictions': `
                    <h4 style="color:#ff5ca9; margin-top:0;">Causal Language Model Predictions</h4>
                    <p>GPT-2's objective: predict the next token given all preceding tokens (autoregressive).</p>

                    <div style="background:rgba(255,255,255,0.05); padding:12px; border-radius:6px; margin:12px 0; border-left:3px solid #f97316;">
                        <strong style="color:#f97316;">CLM Objective:</strong><br>
                        <code style="font-size:11px;">P(token<sub>t</sub> | token<sub>1</sub>, ..., token<sub>t-1</sub>) = Softmax(H<sub>t-1</sub> · W<sub>vocab</sub>)</code>
                    </div>

                    <ul style="font-size:13px;">
                        <li><strong>Training:</strong> All positions predicted in parallel (teacher forcing)</li>
                        <li><strong>Generation:</strong> Sequential sampling, append, repeat</li>
                        <li><strong>Loss:</strong> Cross-entropy summed over all positions</li>
                        <li><strong>Causal mask:</strong> Each token sees only left context</li>
                    </ul>
                `,
            };

            var key = from + '_' + to;
            return explanations[key] || '<p>Explanation not available for this transition.</p>';
        }
        """

__all__ = ["JS_CODE", "JS_INTERACTIVE", "JS_TREE_VIZ", "JS_TRANSITION_MODAL"]
