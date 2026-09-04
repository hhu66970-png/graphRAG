# visualize_standalone.py
"""
Standalone visualization generator for knowledge graph
Generates a self-contained HTML file with embedded data from Neo4j
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
import json

load_dotenv()


def generate_standalone_visualization(output_path="graphragexpr/vis/output/graph.html"):
    """Generate standalone HTML visualization with embedded Neo4j data"""
    print("正在连接到 Neo4j 数据库...")

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URL"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
    )

    try:
        driver.verify_connectivity()
        print("✓ Neo4j 数据库连接成功")
    except Exception as e:
        print(f"✗ 无法连接到 Neo4j: {e}")
        return

    print("正在查询知识图谱数据...")

    query = """
    MATCH (n)-[r]->(m)
    WHERE NOT n:Chunk AND NOT n:Document
      AND NOT m:Chunk AND NOT m:Document
    RETURN elementId(n) AS source_id, labels(n)[0] AS source_label,
           coalesce(n.name, 'unknown') AS source_name,
           type(r) AS rel_type,
           elementId(m) AS target_id, labels(m)[0] AS target_label,
           coalesce(m.name, 'unknown') AS target_name
    """

    nodes = {}
    edges = []

    with driver.session() as session:
        result = session.run(query)

        if not result.peek():
            print("⚠ 没有找到实体数据，请先运行 build_kg_dyn.ps1")
            driver.close()
            return

        for record in result:
            # Add source node
            source_id = record["source_id"]
            if source_id not in nodes:
                nodes[source_id] = {
                    "id": source_id,
                    "label": record["source_name"],
                    "type": record["source_label"]
                }

            # Add target node
            target_id = record["target_id"]
            if target_id not in nodes:
                nodes[target_id] = {
                    "id": target_id,
                    "label": record["target_name"],
                    "type": record["target_label"]
                }

            # Add edge
            edges.append({
                "source": source_id,
                "target": target_id,
                "type": record["rel_type"]
            })

    driver.close()

    print(f"✓ 获取到 {len(nodes)} 个节点和 {len(edges)} 条关系")

    # Generate HTML with embedded data
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>知识图谱可视化</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            overflow: hidden;
        }}

        #container {{
            display: flex;
            height: 100vh;
        }}

        #sidebar {{
            width: 300px;
            background: rgba(26, 26, 46, 0.95);
            padding: 20px;
            overflow-y: auto;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }}

        #graph-container {{
            flex: 1;
            position: relative;
        }}

        h1 {{
            font-size: 24px;
            margin-bottom: 20px;
            color: #4fc3f7;
        }}

        .section {{
            margin-bottom: 25px;
        }}

        .section-title {{
            font-size: 14px;
            color: #4fc3f7;
            margin-bottom: 10px;
            font-weight: 600;
        }}

        .stat {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}

        .stat-label {{
            color: #b0b0b0;
            font-size: 13px;
        }}

        .stat-value {{
            color: #fff;
            font-weight: 600;
            font-size: 13px;
        }}

        .legend {{
            margin-top: 10px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            margin: 8px 0;
            font-size: 12px;
        }}

        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            margin-right: 10px;
        }}

        svg {{
            width: 100%;
            height: 100%;
        }}

        .node circle {{
            stroke: #fff;
            stroke-width: 2px;
            cursor: pointer;
        }}

        .node text {{
            font-size: 12px;
            fill: #fff;
            text-anchor: middle;
            pointer-events: none;
        }}

        .link {{
            stroke: rgba(255, 255, 255, 0.3);
            stroke-width: 2px;
        }}

        .link-label {{
            font-size: 10px;
            fill: #b0b0b0;
            text-anchor: middle;
            pointer-events: none;
        }}

        .node:hover circle {{
            stroke: #4fc3f7;
            stroke-width: 3px;
        }}

        #info {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.7);
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 13px;
            color: #4caf50;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="sidebar">
            <h1>知识图谱</h1>

            <div class="section">
                <div class="section-title">统计信息</div>
                <div class="stat">
                    <span class="stat-label">实体数量</span>
                    <span class="stat-value" id="node-count">-</span>
                </div>
                <div class="stat">
                    <span class="stat-label">关系数量</span>
                    <span class="stat-value" id="edge-count">-</span>
                </div>
            </div>

            <div class="section">
                <div class="section-title">实体类型</div>
                <div class="legend" id="legend"></div>
            </div>
        </div>

        <div id="graph-container">
            <div id="info">独立可视化</div>
            <svg id="graph"></svg>
        </div>
    </div>

    <script>
        // Embedded data from Neo4j
        const graphData = {{
            nodes: {json.dumps(list(nodes.values()), ensure_ascii=False)},
            edges: {json.dumps(edges, ensure_ascii=False)}
        }};

        // Color map for entity types
        const colorMap = {{
            "中药": "#2e7d32",
            "功效": "#66bb6a",
            "主治": "#ef6c00",
            "性味": "#8d6e63",
            "归经": "#5e35b1",
            "用法用量": "#00838f",
            "禁忌": "#c62828",
            "配伍": "#1565c0",
            "药材来源": "#6d4c41",
            "炮制": "#ad8b00",
            "化学成分": "#455a64",
            "方剂": "#7b1fa2",
            "典籍": "#0277bd",
            "人物": "#d81b60",
            "概念": "#78909c"
        }};

        // Update statistics
        document.getElementById('node-count').textContent = graphData.nodes.length;
        document.getElementById('edge-count').textContent = graphData.edges.length;

        // Create legend
        const types = [...new Set(graphData.nodes.map(n => n.type))];
        const legend = document.getElementById('legend');
        types.forEach(type => {{
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `
                <div class="legend-color" style="background: ${{colorMap[type] || '#95a5a6'}}"></div>
                <span>${{type}}</span>
            `;
            legend.appendChild(item);
        }});

        // Render graph
        const svg = d3.select('#graph');
        const width = document.getElementById('graph-container').clientWidth;
        const height = document.getElementById('graph-container').clientHeight;

        svg.attr('viewBox', [0, 0, width, height]);

        // Create force simulation
        const simulation = d3.forceSimulation(graphData.nodes)
            .force('link', d3.forceLink(graphData.edges).id(d => d.id).distance(150))
            .force('charge', d3.forceManyBody().strength(-500))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(30));

        // Create links
        const link = svg.append('g')
            .selectAll('line')
            .data(graphData.edges)
            .join('line')
            .attr('class', 'link')
            .attr('marker-end', 'url(#arrowhead)');

        // Create link labels
        const linkLabel = svg.append('g')
            .selectAll('text')
            .data(graphData.edges)
            .join('text')
            .attr('class', 'link-label')
            .text(d => d.type);

        // Create nodes
        const node = svg.append('g')
            .selectAll('g')
            .data(graphData.nodes)
            .join('g')
            .attr('class', 'node')
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));

        node.append('circle')
            .attr('r', 15)
            .attr('fill', d => colorMap[d.type] || '#95a5a6');

        node.append('text')
            .attr('dy', 30)
            .text(d => d.label);

        // Define arrow marker
        svg.append('defs').append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '-0 -5 10 10')
            .attr('refX', 25)
            .attr('refY', 0)
            .attr('orient', 'auto')
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .append('svg:path')
            .attr('d', 'M 0,-5 L 10,0 L 0,5')
            .attr('fill', 'rgba(255, 255, 255, 0.3)');

        // Update positions on tick
        simulation.on('tick', () => {{
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            linkLabel
                .attr('x', d => (d.source.x + d.target.x) / 2)
                .attr('y', d => (d.source.y + d.target.y) / 2);

            node.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
        }});

        // Drag functions
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}

        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}

        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
    </script>
</body>
</html>"""

    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n✓ 独立可视化已生成: {output_file}")
    print("直接在浏览器中打开即可查看（不需要运行服务器）")


if __name__ == "__main__":
    generate_standalone_visualization()
