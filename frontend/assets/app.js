// 后端接口前缀；图谱节点/边来自 /api/graphrag 的 subgraph，不是浏览器直连数据库。
const API_BASE = '/api';
const MAX_SUBGRAPH_NODES = 80;

let graphData = { nodes: [], edges: [] };
let svg, g, simulation, link, node, zoom;
let nodeDragged = false;

const colorScheme = {
    '中药': '#3D6A4F',
    '功效': '#5E7F6C',
    '主治': '#8A6D4B',
    '性味': '#4F735F',
    '归经': '#3D6A4F',
    '用法用量': '#8A6D4B',
    '禁忌': '#7A5C40',
    '配伍': '#3D6A4F',
    '药材来源': '#3D6A4F',
    '炮制': '#5C7A66',
    '化学成分': '#4A6E58',
    '方剂': '#D8E4DC',
    '典籍': '#8A6D4B',
    '人物': '#8A6D4B',
    '概念': '#7A8A80',
    'default': '#7A8A80'
};

/** 按实体中文类型取节点颜色。 */
function getNodeColor(type) {
    return colorScheme[type] || colorScheme['default'];
}

/** 控制空态提示与演示动画的显示。 */
function setEmptyHintVisible(visible) {
    const hint = document.getElementById('empty-hint');
    if (hint) hint.style.display = visible ? 'block' : 'none';
    const demo = document.getElementById('demo-graph');
    if (demo) demo.style.display = visible ? 'block' : 'none';
}

/** 把接口返回的节点边拷成 D3 可用的纯数据（去掉对象引用）。 */
function cloneGraphPayload(data) {
    const nodes = (data.nodes || []).map(n => ({
        id: n.id,
        label: n.label || n.name || 'unknown',
        type: n.type || '概念'
    }));
    const nodeIds = new Set(nodes.map(n => n.id));
    const edges = (data.edges || [])
        .map(e => ({
            source: typeof e.source === 'object' ? e.source.id : e.source,
            target: typeof e.target === 'object' ? e.target.id : e.target,
            type: e.type || '相关'
        }))
        .filter(e => nodeIds.has(e.source) && nodeIds.has(e.target));
    return { nodes, edges };
}

/** 子图节点过多时截断，避免浏览器卡死。 */
function limitGraph(data) {
    if (data.nodes.length <= MAX_SUBGRAPH_NODES) return data;
    const keep = new Set(data.nodes.slice(0, MAX_SUBGRAPH_NODES).map(n => n.id));
    return {
        nodes: data.nodes.filter(n => keep.has(n.id)),
        edges: data.edges.filter(e => keep.has(e.source) && keep.has(e.target))
    };
}

/** 转义 HTML 特殊字符，防止答案文本被当成标签。 */
function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

/** 在答案里高亮当前子图出现过的实体名。 */
function highlightEntities(text, nodes) {
    let html = escapeHtml(text);
    const labels = [...new Set((nodes || []).map(n => n.label || n.name).filter(Boolean))]
        .filter(label => label.length >= 2)
        .sort((a, b) => b.length - a.length);
    labels.forEach((label) => {
        const token = escapeHtml(label);
        html = html.split(token).join(`<mark class="entity-mark">${token}</mark>`);
    });
    return html.replace(/\n/g, '<br>');
}

/* 行首 Markdown 标题标记与独占一行的分隔线只做**展示**清理：模型输出里的
   「# 标题」「---」在页面上是纯噪声。两条规则都只吃行首 1～6 个 #（且 # 后
   必须有空格）或整行只有分隔线的情况，行内与行尾的 # 一律保留
   （「编号#123」「#1」不会被误删）。

   前导空白用 [ \t] 而不用 \s：\s 会连行尾换行一起吃掉，把标题并到上一段
   末尾（「…汇总\n\n# 人参」会变成「…汇总人参」），这是这类清理最常见的错。 */
const MARKDOWN_HEADING_RE = /^[ \t]{0,3}#{1,6}[ \t]+/gm;
const MARKDOWN_RULE_RE = /^[ \t]{0,3}(?:-{3,}|\*{3,}|_{3,})[ \t]*$/gm;

/** 去掉答案里轻量 Markdown 标记，便于纯文本展示。 */
function stripMarkdownLite(text) {
    return String(text ?? '')
        .replace(MARKDOWN_HEADING_RE, '')
        .replace(MARKDOWN_RULE_RE, '')
        // 只合并多余空行（3 行以上折成 1 个空行），段间分隔仍然保留，
        // 不把正常段落压成一段。
        .replace(/\n{3,}/g, '\n\n');
}

/* 成对的 Markdown 加粗标记只做**展示**处理：成对才去掉，落单的 ** 原样
   保留——宁可多显示两个星号，也不能把内容吃掉。整个规范化都在
   escapeHtml 之前完成（纯字符串处理，不产出 HTML），模型文本因此仍然
   不会未经转义进入 innerHTML。 */
/** 规整答案空白与换行。 */
function normalizeAnswerText(text) {
    return stripMarkdownLite(text).replace(/\*\*([^*\n]+)\*\*/g, '$1');
}

/* 高风险回答按后端给的分块渲染：样式由 kind 决定，不靠查字符串。
   没有分块时（普通回答）走原来的整体渲染，行为不变。
   程序核验说明单独成块，位于全部结论与原文之后。 */
/** 把接口答案渲染成聊天气泡 HTML。 */
function renderAnswerBody(data, nodes) {
    const blocks = data.answer_blocks || [];
    if (!blocks.length) {
        return highlightEntities(normalizeAnswerText(data.answer || ''), nodes);
    }
    const body = blocks.map((block) => {
        const text = highlightEntities(normalizeAnswerText(block.text || ''), nodes);
        const kind = block.kind || 'text';
        if (kind === 'decision') return `<p class="answer-decision">${text}</p>`;
        if (kind === 'conclusion') return `<p class="answer-conclusion">${text}</p>`;
        if (kind === 'quote') return `<p class="answer-evidence-quote">${text}</p>`;
        if (kind === 'note') return `<p class="answer-note">${text}</p>`;
        return `<p class="answer-text">${text}</p>`;
    }).join('');
    const note = data.verification_note
        ? `<div class="answer-verification-note">${escapeHtml(normalizeAnswerText(data.verification_note))}</div>`
        : '';
    return body + note;
}

/** 把来源记录格式化成「第 x 页」。 */
function pageLabel(source) {
    if (source.page_start == null) return '页码未标注';
    return source.page_start === source.page_end
        ? `第 ${source.page_start} 页`
        : `第 ${source.page_start}-${source.page_end} 页`;
}

/** 渲染页码、Chunk 与子图规模等溯源信息。 */
function renderProvenance(data, subgraph) {
    const sources = data.sources || [];
    const nodes = subgraph.nodes || [];
    const intent = data.intent ? `<div class="source-meta">检索意图 ${escapeHtml(data.intent)}</div>` : '';
    const warnings = (data.warnings || []).length
        ? `<div class="source-meta">${escapeHtml(data.warnings.join('；'))}</div>`
        : '';
    const confidence = data.trace && data.trace.route && data.trace.route.confidence != null
        ? `<div class="source-meta">路由置信度 ${escapeHtml(data.trace.route.confidence)}</div>`
        : '';
    const chips = nodes.slice(0, 18).map(n => `<span class="chip">${escapeHtml(n.label || n.name)}</span>`).join('');
    const sourceCards = sources.length
        ? sources.map((source) => {
            const quote = source.evidence_quote || source.text || '';
            const score = source.score != null ? ` · 分数 ${escapeHtml(source.score)}` : '';
            const terms = (source.matched_terms || []).filter(Boolean);
            return `
                <article class="source-card">
                    <div class="source-meta">教材原文 · ${escapeHtml(pageLabel(source))}${score}</div>
                    <p class="source-text">${quote ? escapeHtml(quote) : (terms.length ? `命中词：${escapeHtml(terms.join('、'))}` : '该来源未返回原文片段，仅保留页码溯源。')}</p>
                </article>
            `;
        }).join('')
        : '<article class="source-card"><p class="source-text">本次回答未返回可展开的原文片段。</p></article>';

    const isVector = data.method === 'vector';
    const summary = isVector
        ? `向量检索溯源 · ${sources.length} 个文本块（不使用知识图谱扩展）`
        : `图检索溯源 · 子图 ${nodes.length} 个实体 / ${(subgraph.edges || []).length} 条关系`;
    return `
        <details class="trace-block">
            <summary>${summary}</summary>
            <div class="trace-body">
                ${intent}${confidence}${warnings}
                ${chips ? `<div class="chip-row">${chips}</div>` : ''}
                ${sourceCards}
            </div>
        </details>
    `;
}

/* 高风险回答的状态文案。四档语义不同，展示上必须区分：
   grounded 才代表用户问的高风险结论都给出了；
   source_only / abstain 表示只有原文或什么都不能说，绝不是「安全」。 */
const HR_STATUS_LABEL = {
    'grounded': '已核验',
    'partial_grounded': '部分核验',
    'source_only': '仅有原文，未形成结论',
    'abstain': '证据不足，不予结论',
    'verification_unavailable': '未通过核验（核验层不可用）'
};

const HR_STATUS_CLASS = {
    'grounded': 'hr-ok',
    'partial_grounded': 'hr-warn',
    'source_only': 'hr-warn',
    'abstain': 'hr-bad',
    'verification_unavailable': 'hr-bad'
};

const HR_SUB_STATUS_LABEL = {
    'grounded': '已核验',
    'partial_grounded': '部分核验',
    'source_only': '仅有原文',
    'abstain': '无证据'
};

/* 结构化卡片放在哪里，按 answer_status 决定。**纯展示层策略**：
   后端字段一个不动，`evidence_units` / `claims` / `sources` / `subgraph`
   照常返回，溯源面板、接口契约、同源核验与评测都不受影响。

   grounded 不再列卡：程序化回答块已经把同一批证据逐条写出（结论行 +
   「原文依据：…（第 N 页）」+ 灰色核验说明），再列一次 evidence_units 就是
   同一批证据的两套展示——实测附子用量页面「3～15g／原文依据／第 231 页」
   各出现两次，读者要自己比对两边是否一致。

   其余状态**不删卡片**，改为默认折叠：partial_grounded 的冲突要看卡片里的
   条件与来源类型才能对照，source_only 的原文也仍然可查，只是不再与正文
   重复展开。 */
const HR_CARDS_MODE = {
    'grounded': 'none',
    'partial_grounded': 'folded',
    'source_only': 'folded',
    'abstain': 'folded',
    'verification_unavailable': 'folded'
};

/** 渲染剂量/禁忌等核验块；普通问题返回空字符串。 */
function renderHighRiskBlock(data) {
    const hr = data.high_risk;
    if (!hr || !hr.is_high_risk) return '';

    const status = data.answer_status || 'abstain';
    const badge = `<span class="hr-badge ${HR_STATUS_CLASS[status] || 'hr-bad'}">
        高风险 · ${escapeHtml(HR_STATUS_LABEL[status] || status)}
    </span>`;

    const subStatuses = (data.sub_questions || []).map(item => `
        <span class="hr-substatus">
            ${escapeHtml(item.question)}：${escapeHtml(HR_SUB_STATUS_LABEL[item.status] || item.status)}
        </span>
    `).join('');

    /* 未成立的部分要写出原因，而不是留给读者猜。 */
    const reasons = (data.sub_questions || [])
        .filter(item => item.status !== 'grounded')
        .map(item => `<p class="hr-note">${escapeHtml(item.question)}：${escapeHtml(item.reason)}</p>`)
        .join('');

    /* 条件证据卡：条件、程度、来源类型与原文同卡。
       数值绝不与它自己的用法／生制品／人群分开显示。
       原文一律带「原文依据：」标签，与结论区分开。 */
    const renderCard = (unit) => `
        <article class="hr-card">
            <div class="hr-card-head">
                <span class="hr-card-cat">${escapeHtml(unit.category_label)}</span>
                <span class="hr-card-subject">${escapeHtml(unit.subject_name)}</span>
                ${unit.object_name ? `<span class="hr-card-meta">→ ${escapeHtml(unit.object_name)}</span>` : ''}
            </div>
            <p class="hr-card-conditions">
                ${escapeHtml(unit.conditions || '原文未给出附加条件')} · 程度：${escapeHtml(unit.severity_label)}
            </p>
            <p class="hr-card-quote"><span class="hr-card-quote-label">原文依据：</span>${escapeHtml(unit.evidence_quote || '（未取到原文）')}</p>
            <div class="hr-card-meta">
                ${escapeHtml(unit.page_label)} · 来源类型：${unit.source_kind === 'ASSERTION' ? '已校验断言' : '教材原文（未经断言校验）'}
                ${unit.is_citable ? '' : ' · 原文不可定位'}
            </div>
            <!-- 说明行由后端定稿（flags：原文说明／证据存在差异／当前证据不足以确定）。
                 前端不再对 conflicts 各自套模板：套错会把解析层说明说成「证据存在差异」，
                 与同一张卡的「已核验」徽标自相矛盾。 -->
            ${(unit.flags || []).map(flag => `<div class="hr-flag">${escapeHtml(flag)}</div>`).join('')}
        </article>
    `;
    const units = data.evidence_units || [];
    const cardsMode = HR_CARDS_MODE[status] || 'folded';
    const renderedCards = units.map(renderCard).join('');
    const cards = cardsMode === 'none'
        ? ''
        : (cardsMode === 'folded' && units.length
            ? `<details class="hr-structured">
                   <summary>查看结构化证据 · ${units.length} 条（条件、程度与来源类型）</summary>
                   ${renderedCards}
               </details>`
            : renderedCards);

    /* 单元级风险说明（原文说明／证据存在差异／当前证据不足以确定）不跟着
       卡片一起消失：grounded 不列卡时，这些说明必须仍然出现在页面上，
       否则「证据存在差异」这类真正影响结论的信息会因为展示去重而丢掉。
       卡片还在（折叠或展开）时说明留在卡内，不重复列出。 */
    const unitFlags = cardsMode === 'none'
        ? units
            .map(unit => unit.flags || [])
            .reduce((all, flags) => all.concat(flags), [])
            .filter((flag, index, all) => all.indexOf(flag) === index)
            .map(flag => `<div class="hr-flag">${escapeHtml(flag)}</div>`)
            .join('')
        : '';

    /* 与本题指定条件不同的证据：默认折叠，明确标注「与本题指定人群不同」。
       它们不参与结论，也不在主要回答里出现。 */
    const others = (data.other_conditions || []).length ? `
        <details class="hr-other">
            <summary>其他使用注意（与本题指定人群不同）· ${(data.other_conditions || []).length} 条</summary>
            ${(data.other_conditions || []).map(renderCard).join('')}
        </details>
    ` : '';

    /* 被驳回的候选结论与硬性失败原因同样展示，使「为什么没采用」可追溯，
       也让模型自评无法在界面上悄悄翻案。 */
    const rejections = (data.rejections || []).length ? `
        <details class="hr-rejected">
            <summary>本回答中 ${(data.rejections || []).length} 条候选结论未通过程序核验</summary>
            <ul>
                ${(data.rejections || []).map(item => `
                    <li>${escapeHtml(item.claim_id)}：${escapeHtml([].concat(item.notes || [], item.hard_failures || []).join('；'))}</li>
                `).join('')}
            </ul>
        </details>
    ` : '';

    return `
        <div class="hr-block">
            <div class="hr-head">${badge}${subStatuses}</div>
            ${reasons}${unitFlags}${cards}${others}${rejections}
        </div>
    `;
}

/** 用户首次提问后收起欢迎语。 */
function hideWelcome() {
    const welcome = document.getElementById('chat-welcome');
    if (welcome) welcome.remove();
}

/** 在对话区追加一条用户或系统消息。 */
function appendMessage(role, html) {
    hideWelcome();
    const thread = document.getElementById('chat-thread');
    const wrap = document.createElement('div');
    wrap.className = `msg msg-${role}`;
    wrap.innerHTML = html;
    thread.appendChild(wrap);
    thread.scrollTop = thread.scrollHeight;
    return wrap;
}

/** 初始化 D3 力导向图画布、箭头和缩放。 */
function initGraph() {
    svg = d3.select('#graph');
    const container = document.getElementById('graph-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    svg.attr('width', width).attr('height', height);

    svg.selectAll('*').remove();
    svg.append('defs').append('marker')
        .attr('id', 'arrowhead')
        .attr('viewBox', '-0 -5 10 10')
        .attr('refX', 18)
        .attr('refY', 0)
        .attr('orient', 'auto')
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .append('path')
        .attr('d', 'M 0,-5 L 10,0 L 0,5')
        .attr('fill', '#3D6A4F');

    zoom = d3.zoom()
        .scaleExtent([0.1, 10])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });

    svg.call(zoom);
    g = svg.append('g');

    simulation = d3.forceSimulation()
        .force('link', d3.forceLink().id(d => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(30));

    document.getElementById('zoom-in').addEventListener('click', () => {
        svg.transition().call(zoom.scaleBy, 1.3);
    });

    document.getElementById('zoom-out').addEventListener('click', () => {
        svg.transition().call(zoom.scaleBy, 0.7);
    });

    document.getElementById('zoom-reset').addEventListener('click', () => {
        svg.transition().call(zoom.transform, d3.zoomIdentity);
    });
}

/** 窗口变化时重算画布中心。 */
function resizeGraph() {
    if (!svg || !simulation) return;
    const container = document.getElementById('graph-container');
    const width = container.clientWidth;
    const height = container.clientHeight;
    svg.attr('width', width).attr('height', height);
    simulation.force('center', d3.forceCenter(width / 2, height / 2));
    simulation.alpha(0.25).restart();
}

/** 鼠标悬停节点时显示名称与类型。 */
function showTooltip(event, d) {
    const tip = document.getElementById('graph-tooltip');
    const container = document.getElementById('graph-container');
    const rect = container.getBoundingClientRect();
    tip.innerHTML = `<div class="detail-title">${escapeHtml(d.label)}</div><div class="detail-type">${escapeHtml(d.type)}</div>`;
    tip.classList.add('visible');
    const x = event.clientX - rect.left + 14;
    const y = event.clientY - rect.top + 14;
    tip.style.left = `${Math.min(x, rect.width - 200)}px`;
    tip.style.top = `${Math.min(y, rect.height - 80)}px`;
}

/** 隐藏节点悬浮提示。 */
function hideTooltip() {
    document.getElementById('graph-tooltip').classList.remove('visible');
}

/** 点击节点后在侧栏列出相邻关系。 */
function showNodeDetails(d) {
    const panel = document.getElementById('details-panel');
    const related = graphData.edges
        .map(e => {
            const sourceId = typeof e.source === 'object' ? e.source.id : e.source;
            const targetId = typeof e.target === 'object' ? e.target.id : e.target;
            const sourceNode = graphData.nodes.find(n => n.id === sourceId);
            const targetNode = graphData.nodes.find(n => n.id === targetId);
            if (sourceId !== d.id && targetId !== d.id) return null;
            const other = sourceId === d.id ? targetNode : sourceNode;
            return `${escapeHtml(e.type)} → ${escapeHtml((other && other.label) || '未命名')}`;
        })
        .filter(Boolean)
        .slice(0, 8);
    panel.innerHTML = `
        <div class="detail-title">${escapeHtml(d.label)}</div>
        <div class="detail-type">${escapeHtml(d.type)}</div>
        ${related.length ? related.map(item => `<div>${item}</div>`).join('') : '<div class="hint-text">当前子图中暂无相邻关系</div>'}
    `;
    panel.classList.add('visible');
    panel.style.display = 'block';

    g.selectAll('.node').classed('highlighted', n => n.id === d.id);
    g.selectAll('.link').classed('highlighted', e => {
        const sourceId = typeof e.source === 'object' ? e.source.id : e.source;
        const targetId = typeof e.target === 'object' ? e.target.id : e.target;
        return sourceId === d.id || targetId === d.id;
    });
}

/** 用新的 subgraph 重绘力导向图。 */
function updateGraph(data) {
    const prepared = limitGraph(cloneGraphPayload(data || { nodes: [], edges: [] }));
    graphData = prepared;
    setEmptyHintVisible(prepared.nodes.length === 0);
    document.getElementById('details-panel').style.display = 'none';
    document.getElementById('details-panel').classList.remove('visible');
    hideTooltip();

    link = g.selectAll('.link')
        .data(prepared.edges, d => `${d.source}-${d.target}-${d.type}`)
        .join('path')
        .attr('class', 'link')
        .attr('marker-end', 'url(#arrowhead)');

    const linkLabel = g.selectAll('.link-label')
        .data(prepared.edges, d => `${d.source}-${d.target}-${d.type}`)
        .join('text')
        .attr('class', 'link-label')
        .text(d => d.type);

    const nodeGroup = g.selectAll('.node')
        .data(prepared.nodes, d => d.id)
        .join('g')
        .attr('class', 'node')
        .call(d3.drag()
            .on('start', dragStarted)
            .on('drag', dragged)
            .on('end', dragEnded));

    nodeGroup.selectAll('*').remove();

    nodeGroup.append('circle')
        .attr('r', 8)
        .attr('fill', d => getNodeColor(d.type))
        .attr('stroke', d => d.type === '方剂' ? '#3D6A4F' : '#FCFAF5');

    nodeGroup.append('text')
        .attr('dy', 20)
        .text(d => d.label);

    nodeGroup
        .on('mouseenter', (event, d) => showTooltip(event, d))
        .on('mousemove', (event, d) => showTooltip(event, d))
        .on('mouseleave', hideTooltip)
        .on('click', (event, d) => {
            if (nodeDragged) return;
            event.stopPropagation();
            showNodeDetails(d);
        });

    node = nodeGroup;

    simulation.nodes(prepared.nodes);
    simulation.force('link').links(prepared.edges);
    simulation.alpha(1).restart();

    simulation.on('tick', () => {
        link.attr('d', d => {
            const dx = d.target.x - d.source.x;
            const dy = d.target.y - d.source.y;
            const dr = Math.sqrt(dx * dx + dy * dy) || 1;
            return `M${d.source.x},${d.source.y}A${dr},${dr} 0 0,1 ${d.target.x},${d.target.y}`;
        });

        linkLabel
            .attr('x', d => (d.source.x + d.target.x) / 2)
            .attr('y', d => (d.source.y + d.target.y) / 2);

        node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    updateLegend(prepared.nodes);
}

/** 按当前子图类型更新图例。 */
function updateLegend(nodes) {
    const typeCounts = {};
    nodes.forEach(nodeItem => {
        typeCounts[nodeItem.type] = (typeCounts[nodeItem.type] || 0) + 1;
    });

    const legend = document.getElementById('legend');
    legend.innerHTML = '';

    Object.entries(typeCounts).sort((a, b) => b[1] - a[1]).forEach(([type, count]) => {
        const item = document.createElement('div');
        item.className = 'legend-item';
        item.innerHTML = `
            <div class="legend-color" style="background: ${getNodeColor(type)}"></div>
            <div class="legend-label">${type}</div>
            <div class="legend-count">${count}</div>
        `;
        legend.appendChild(item);
    });
}

/** 用 /api/stats 的类型计数更新图例。 */
function updateLegendFromStats(nodesByType) {
    const legend = document.getElementById('legend');
    legend.innerHTML = '';
    Object.entries(nodesByType || {})
        .sort((a, b) => b[1] - a[1])
        .forEach(([type, count]) => {
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `
                <div class="legend-color" style="background: ${getNodeColor(type)}"></div>
                <div class="legend-label">${type}</div>
                <div class="legend-count">${count}</div>
            `;
            legend.appendChild(item);
        });
}

/** 开始拖拽节点。 */
function dragStarted(event, d) {
    nodeDragged = false;
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
}

/** 拖拽过程中固定节点位置。 */
function dragged(event, d) {
    nodeDragged = true;
    d.fx = event.x;
    d.fy = event.y;
}

/** 结束拖拽并释放节点。 */
function dragEnded(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
}

/** 调用 /api/health，检查 Flask 能否连上 Neo4j。 */
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        document.getElementById('status').textContent = data.status === 'ok' ? '已连接' : '连接失败';
        document.getElementById('status').style.color = data.status === 'ok' ? '#3D6A4F' : '#A35A45';
    } catch (error) {
        document.getElementById('status').textContent = '无法连接';
        document.getElementById('status').style.color = '#A35A45';
    }
}

/** 调用 /api/stats，显示节点/边数量。 */
async function loadStats() {
    document.getElementById('loading').style.display = 'block';
    document.getElementById('loading').textContent = '加载图谱统计...';
    try {
        const response = await fetch(`${API_BASE}/stats`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || '统计接口失败');

        document.getElementById('node-count').textContent = data.total_nodes ?? '-';
        document.getElementById('edge-count').textContent = data.total_relationships ?? '-';
        updateLegendFromStats(data.nodes_by_type);
        setEmptyHintVisible(true);
    } catch (error) {
        console.error('Error loading stats:', error);
        document.getElementById('loading').textContent = '统计加载失败';
        document.getElementById('node-count').textContent = '-';
        document.getElementById('edge-count').textContent = '-';
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

// 提问：POST /api/graphrag，再渲染答案、高风险块与子图。
document.getElementById('ask-button').addEventListener('click', async () => {
    const question = document.getElementById('question-input').value.trim();
    const method = document.getElementById('retrieval-method').value;

    if (!question) return;

    const askButton = document.getElementById('ask-button');
    document.getElementById('question-input').value = '';

    askButton.disabled = true;
    askButton.innerHTML = '<span class="loading-spinner"></span> 思考中...';

    appendMessage('user', `
        <div class="msg-label">提问</div>
        <div class="bubble">${escapeHtml(question)}</div>
    `);
    const loadingHint = method === 'auto'
        ? '正在检索知识图谱...'
        : method === 'keyword_graph'
            ? '正在图检索（跨块补证据）...'
            : '正在向量检索（首次会加载本地模型，可能需要一些时间）...';
    const aiMsg = appendMessage('ai', `
        <div class="msg-label">回答</div>
        <div class="bubble"><span class="loading-spinner"></span> ${loadingHint}</div>
    `);

    try {
        const response = await fetch(`${API_BASE}/graphrag`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                method: method
            })
        });

        const data = await response.json();

        if (response.ok) {
            const methodLabel = {
                'auto': `自动路由${data.intent ? ` · ${data.intent}` : ''}`,
                'keyword_graph': '图检索',
                'vector': '向量检索'
            }[data.method] || data.method;

            const subgraph = data.method === 'vector'
                ? { nodes: [], edges: [] }
                : (data.subgraph || { nodes: [], edges: [] });
            const nodeCount = subgraph.nodes ? subgraph.nodes.length : 0;
            const edgeCount = subgraph.edges ? subgraph.edges.length : 0;

            aiMsg.innerHTML = `
                <div class="msg-label">回答 · ${escapeHtml(methodLabel)}</div>
                <div class="bubble">${renderAnswerBody(data, subgraph.nodes || [])}</div>
                ${renderHighRiskBlock(data)}
                ${renderProvenance(data, subgraph)}
                <div class="source-meta" style="margin-top:8px;">画布展示相关子图：${nodeCount} 个节点，${edgeCount} 条关系</div>
            `;

            if (nodeCount > 0) {
                updateGraph(subgraph);
            } else {
                updateGraph({ nodes: [], edges: [] });
            }
        } else {
            aiMsg.innerHTML = `<div class="bubble error-text">错误: ${escapeHtml(data.error || '请求失败')}</div>`;
        }
    } catch (error) {
        console.error('Error asking question:', error);
        aiMsg.innerHTML = `<div class="bubble error-text">错误: ${escapeHtml(error.message)}</div>`;
    } finally {
        askButton.disabled = false;
        askButton.textContent = '提问';
        const thread = document.getElementById('chat-thread');
        thread.scrollTop = thread.scrollHeight;
    }
});

document.getElementById('question-input').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        document.getElementById('ask-button').click();
    }
});

document.querySelectorAll('.quick-prompt').forEach((prompt) => {
    prompt.addEventListener('click', () => {
        const input = document.getElementById('question-input');
        input.value = prompt.dataset.question || '';
        input.focus();
    });
});

document.getElementById('graph-container').addEventListener('click', (event) => {
    if (event.target.closest('.node') || event.target.closest('.zoom-controls')) return;
    const panel = document.getElementById('details-panel');
    panel.style.display = 'none';
    panel.classList.remove('visible');
    if (g) {
        g.selectAll('.node').classed('highlighted', false);
        g.selectAll('.link').classed('highlighted', false);
    }
});

window.addEventListener('resize', resizeGraph);

initGraph();
checkHealth();
loadStats();
setInterval(checkHealth, 10000);
