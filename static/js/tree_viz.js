// Token Influence Tree D3.js Visualization
// Horizontal collapsible tree with attention-based styling

function renderInfluenceTree(treeData, containerId) {
    // Clear any existing SVG
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

    // Configuration
    // Configuration
    const container = document.getElementById(containerId);
    const containerWidth = container.clientWidth || 960;
    const containerHeight = container.clientHeight || 600;

    const margin = { top: 40, right: 60, bottom: 40, left: 60 };

    // Dynamic width calculation
    // We need to count leaf nodes to determine required width
    const rootForCount = d3.hierarchy(treeData);
    const leafCount = rootForCount.leaves().length;
    const minNodeWidth = 60; // Reduced from 100 to make it more compact
    // We want it to fit in container if possible, but expand if needed
    const requiredWidth = Math.max(containerWidth - margin.left - margin.right, leafCount * minNodeWidth);

    const width = requiredWidth;
    const height = containerHeight - margin.top - margin.bottom;

    // Color palette
    const colors = {
        root: '#ff5ca9',
        level1: '#3b82f6',
        level2: '#8b5cf6',
        level3: '#06b6d4'
    };

    // Create SVG
    const svg = d3.select(`#${containerId}`)
        .append("svg")
        .attr("width", width + margin.right + margin.left)
        .attr("height", height + margin.top + margin.bottom)
        .style("font", "12px 'Inter', sans-serif")
        .style("display", "inline-block") // Allows centering via text-align: center on parent
        .style("min-width", "100%"); // Ensure it fills at least the container

    const g = svg.append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Create tree layout - Vertical (size([width, height]))
    const tree = d3.tree().size([width, height]);

    // Auto-scroll to center if content exceeds container
    if (width > (containerWidth - margin.left - margin.right)) {
        setTimeout(() => {
            const scrollLeft = (width + margin.left + margin.right - containerWidth) / 2;
            container.scrollLeft = scrollLeft;
        }, 100);
    }

    // Create hierarchy
    const root = d3.hierarchy(treeData);

    // Collapse all children initially except first level
    root.children.forEach(collapse);

    function collapse(d) {
        if (d.children) {
            d._children = d.children;
            d._children.forEach(collapse);
            d.children = null;
        }
    }

    update(root);

    function update(source) {
        // Compute the new tree layout
        const treeData = tree(root);
        const nodes = treeData.descendants();
        const links = treeData.descendants().slice(1);

        // Normalize for fixed-depth - Vertical spacing
        nodes.forEach(d => { d.y = d.depth * 80; }); // Reduced vertical spacing from 120 to 80

        // ****************** Nodes section ***************************

        // Update the nodes
        const node = g.selectAll('g.node')
            .data(nodes, d => d.id || (d.id = ++i));

        // Enter any new nodes at the parent's previous position
        const nodeEnter = node.enter().append('g')
            .attr('class', 'node')
            .attr("transform", d => `translate(${source.x0 || width / 2},${source.y0 || 0})`)
            .on('click', click);

        // Add Circle for the nodes
        nodeEnter.append('circle')
            .attr('class', 'node-circle')
            .attr('r', 1e-6)
            .style("fill", d => getNodeColor(d))
            .style("stroke", d => getNodeColor(d))
            .style("stroke-width", d => 2 + (d.data.att || 0) * 3)
            .style("opacity", d => 0.3 + (d.data.att || 0) * 0.7);

        // Add labels for the nodes
        nodeEnter.append('text')
            .attr("dy", ".35em")
            .attr("dy", ".35em")
            .attr("x", d => d.children || d._children ? -13 : 13)
            .attr("text-anchor", d => d.children || d._children ? "end" : "start")
            .text(d => d.data.name)
            .style("fill", d => getNodeColor(d))
            .style("font-weight", d => d.depth === 0 ? "700" : "500")
            .style("font-size", d => d.depth === 0 ? "12px" : "10px"); // Reduced font size

        // Add attention score label
        nodeEnter.append('text')
            .attr("dy", "1.5em")
            .attr("dy", "1.8em")
            .attr("x", d => d.children || d._children ? -13 : 13)
            .attr("text-anchor", d => d.children || d._children ? "end" : "start")
            .text(d => d.depth > 0 ? `att: ${d.data.att.toFixed(3)}` : "")
            .style("fill", "#64748b")
            .style("font-size", "10px");

        // Add tooltip
        nodeEnter.append("title")
            .text(d => {
                if (d.depth === 0) return `Root: ${d.data.name}`;
                return `Token: ${d.data.name}\nAttention: ${d.data.att.toFixed(4)}\nQ·K Similarity: ${d.data.qk_sim.toFixed(4)}`;
            });

        // UPDATE
        const nodeUpdate = nodeEnter.merge(node);

        // Transition to the proper position for the node
        nodeUpdate.transition()
            .duration(750)
            .attr("transform", d => `translate(${d.x},${d.y})`);

        // Update the node attributes and style
        nodeUpdate.select('circle.node-circle')
            .attr('r', d => 4 + (d.data.att || 0) * 3) // Reduced radius from 6 to 4
            .style("fill", d => d._children ? getNodeColor(d) : "#fff")
            .style("cursor", "pointer");

        // Remove any exiting nodes
        const nodeExit = node.exit().transition()
            .duration(750)
            .attr("transform", d => `translate(${source.x},${source.y})`)
            .remove();

        // On exit reduce the node circles size to 0
        nodeExit.select('circle')
            .attr('r', 1e-6);

        // On exit reduce the opacity of text labels
        nodeExit.select('text')
            .style('fill-opacity', 1e-6);

        // ****************** links section ***************************

        // Update the links
        const link = g.selectAll('path.link')
            .data(links, d => d.id);

        // Enter any new links at the parent's previous position
        const linkEnter = link.enter().insert('path', "g")
            .attr("class", "link")
            .attr('d', d => {
                const o = { x: source.x0 || width / 2, y: source.y0 || 0 };
                return diagonal(o, o);
            })
            .style("fill", "none")
            .style("stroke", d => getNodeColor(d))
            .style("stroke-width", d => 1 + (d.data.att || 0) * 4)
            .style("opacity", d => 0.2 + (d.data.att || 0) * 0.6);

        // UPDATE
        const linkUpdate = linkEnter.merge(link);

        // Transition back to the parent element position
        linkUpdate.transition()
            .duration(750)
            .attr('d', d => diagonal(d, d.parent));

        // Remove any exiting links
        link.exit().transition()
            .duration(750)
            .attr('d', d => {
                const o = { x: source.x, y: source.y };
                return diagonal(o, o);
            })
            .remove();

        // Store the old positions for transition
        nodes.forEach(d => {
            d.x0 = d.x;
            d.y0 = d.y;
        });

        // Creates a curved (diagonal) path from parent to the child nodes - Vertical
        function diagonal(s, d) {
            return `M ${s.x} ${s.y}
                    C ${s.x} ${(s.y + d.y) / 2},
                      ${d.x} ${(s.y + d.y) / 2},
                      ${d.x} ${d.y}`;
        }

        // Toggle children on click
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
}

// Counter for node IDs
let i = 0;

// Make function globally available
window.renderInfluenceTree = renderInfluenceTree;
