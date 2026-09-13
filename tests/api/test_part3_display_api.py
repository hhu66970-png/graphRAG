"""第三部分（回答组织与前端展示）的接口层与前端回归。

为什么单独一组：第三部分改的是**已经核验通过的证据怎么组织给用户看**，
不碰检索、不碰核验算法。这类改动最容易「后端单元测试全绿、用户在页面上
看到的还是老样子」，因此这里必须从两个真实边界去看：

1. **接口边界**：`/api/graphrag` 的真实响应里有 `decision`／`decision_text`／
   `verification_note`／`answer_blocks`／`other_conditions`，
   且与本题条件不同的证据确实被挡在 `answer`／`claims`／`evidence_units`／
   `sources`／`subgraph` 之外，只出现在折叠区。
2. **浏览器边界**：把 `frontend/index.html` 里的**纯函数原样抽出来，用
   Node 真实执行**（不是读源码猜字符串），检查成对 `**` 被去掉、实体高亮仍
   生效、`<script>` 仍被转义、普通回答仍走原来的整体渲染。

环境隔离：与 `test_high_risk_api.py` 同一套替身（检索／Neo4j／普通 LLM 全部
被替换，嵌入器一旦被调用即失败），本文件在没有数据库、没有网络、没有本地模型
的机器上必须全绿。唯一的外部依赖是 Node.js；缺失时前端执行用例明确 skip，
**不会**被记成通过。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import (
    BENIGN_QUESTION,
    MAHUANG_ID,
    POPULATION_QUESTION,
    ask,
    assertion_row,
)

FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "index.html"

#: 「与本题指定条件不同」的人群。真实库里附子/麻黄都有这一条，
#: 也是拆改前抢占第一屏的那一条。
OTHER_POPULATION = "阴虚阳亢者"

#: 单元级说明整句（后端 `EvidenceCard.flags` 定稿，前端原样显示）。
#: 与 Node 夹具里的 `FLAG` 必须逐字一致：这几条说明在 grounded 不列卡时
#: 仍要出现在页面上，是「去重不得丢掉风险说明」的判据。
DOSE_FLAG = "原文说明：后续小句不是另一处独立剂量"


def rows_with_other_population() -> list[dict]:
    """一条与问题人群精确匹配的证据 + 一条**别的人群**的证据。

    两行的主语都是麻黄，所以它们都会被本题读进来；区分的只有宾语人群。
    """
    return [
        assertion_row(
            assertion_id="a-p231",
            predicate="CONTRAINDICATED_IN",
            polarity="NEGATIVE",
            object_id="P:孕妇",
            object_name="孕妇",
            evidence_quote="孕妇慎用。",
            chunk_texts=["孕妇慎用。"],
            chunk_page_starts=[231],
            chunk_page_ends=[231],
        ),
        assertion_row(
            assertion_id="a-hot",
            predicate="CONTRAINDICATED_IN",
            polarity="NEGATIVE",
            object_id="P:阴虚阳亢",
            object_name=OTHER_POPULATION,
            evidence_quote="阴虚阳亢者忌用。",
            chunk_texts=["阴虚阳亢者忌用。"],
            chunk_page_starts=[230],
            chunk_page_ends=[230],
        ),
    ]


# ------------------------------------------------------------ 接口字段
def test_first_sentence_answers_the_question_through_the_api(client, wire):
    """第一句直接回答，且核验说明在**末尾**。

    拆改前响应里的第一句是「以下结论均通过程序核验…」这一行元信息，
    用户要先读完它才看到答案；核验说明本身没有错，位置错了。
    """
    wire(rows=rows_with_other_population())

    code, data = ask(client, POPULATION_QUESTION, "auto")

    assert code == 200, data
    assert data["answer_status"] == "grounded", data["coverage_notes"]
    assert data["decision"] == "caution"
    assert data["decision_text"].startswith("不能直接判断为可以使用。")
    assert "慎用" in data["decision_text"]
    assert data["answer"].startswith(data["decision_text"])
    assert not data["answer"].startswith("以下结论")
    # 核验说明是收尾担保：位于答案末尾，而不是开头
    assert data["verification_note"].startswith("以上结论")
    assert data["answer"].rstrip().endswith(data["verification_note"])
    assert data["abstained"] is False


def test_answer_blocks_carry_kind_and_order(client, wire):
    """分块让前端按种类套样式；顺序即任务书 §7 的展示顺序。"""
    wire(rows=rows_with_other_population())

    code, data = ask(client, POPULATION_QUESTION, "auto")

    assert code == 200, data
    blocks = data["answer_blocks"]
    assert blocks, "高风险答案必须给出分块"
    for block in blocks:
        assert set(block) == {"kind", "text", "unit_ids", "page_label"}
    kinds = [block["kind"] for block in blocks]
    assert kinds[0] == "decision", kinds
    assert "quote" in kinds, kinds
    assert kinds.index("decision") < kinds.index("quote")
    # 原文依据统一带标签，且与结论分开成块
    quotes = [b for b in blocks if b["kind"] == "quote"]
    assert all(b["text"].startswith("原文依据：") for b in quotes)
    assert all("原文依据：" not in b["text"] for b in blocks if b["kind"] != "quote")
    # 分块与整段文字描述同一批内容：每条引文都能在 answer 里逐字找到
    for block in quotes:
        assert block["text"] in data["answer"]


def test_other_population_evidence_stays_out_of_the_main_answer(client, wire):
    """只显示与问题条件相关的证据：别的人群的证据只进折叠区。

    「只显示」在这里有确切含义——它不得出现在 `answer`、`claims`、
    `evidence_units`、`sources`、`subgraph` 中的任何一处，但仍要**可查**，
    所以保留在 `other_conditions` 里。
    """
    wire(rows=rows_with_other_population())

    code, data = ask(client, POPULATION_QUESTION, "auto")

    assert code == 200, data
    assert OTHER_POPULATION not in data["answer"]
    assert {claim["object_name"] for claim in data["claims"]} == {"孕妇"}
    assert {unit["object_name"] for unit in data["evidence_units"]} == {"孕妇"}
    assert all(OTHER_POPULATION not in json.dumps(s, ensure_ascii=False) for s in data["sources"])
    nodes = {node["id"] for node in data["subgraph"]["nodes"]}
    assert nodes == {MAHUANG_ID, "P:孕妇"}, nodes

    # 不丢：形状与 evidence_units 完全一致，前端可用同一个渲染函数处理
    others = data["other_conditions"]
    assert [card["object_name"] for card in others] == [OTHER_POPULATION]
    assert set(others[0]) == set(data["evidence_units"][0])
    assert others[0]["unit_id"].startswith("unit:")
    assert others[0]["unit_id"] not in {
        unit["unit_id"] for unit in data["evidence_units"]
    }
    assert others[0]["evidence_quote"] == "阴虚阳亢者忌用。"


def test_other_population_evidence_cannot_change_the_decision(client, wire):
    """别的人群的「忌用」不得把本题的首句拔高成「不宜使用」。"""
    wire(rows=rows_with_other_population())

    code, data = ask(client, POPULATION_QUESTION, "auto")

    assert code == 200, data
    # 阴虚阳亢者是「忌用」，比孕妇的「慎用」更重；首句仍必须是慎用这一档
    assert data["decision"] == "caution"
    assert data["decision_text"].startswith("不能直接判断为可以使用。")


def test_only_foreign_evidence_abstains_instead_of_answering(client, wire):
    """本题人群没有任何证据时，宁可弃权，也不拿别的人群的证据作答。

    用户的问法是哺乳期；读到的两行证据分别讲孕妇与阴虚阳亢者。拆改前这些
    证据会以「结论」的样子出现在哺乳期的回答里并标记已核验。
    """
    wire(rows=rows_with_other_population())

    code, data = ask(client, "麻黄哺乳期能用吗？", "auto")

    assert code == 200, data
    assert data["answer_status"] == "source_only", data["answer_status"]
    assert data["abstained"] is True
    assert data["claims"] == []
    assert data["decision"] == "unknown"
    assert data["decision_text"].startswith("根据当前教材证据，无法判断哺乳期能否使用")
    # 首句不得读成许可
    assert "可以" not in data["decision_text"]
    # 没有结论就不显示「通过核验」说明
    assert data["verification_note"] == ""
    # 正文说明「有证据，但说的是别人」，而不是让人以为教材里什么都没有
    assert "没有与本题指定条件一致的可核对原文" in data["answer"]
    assert "教材中另有 2 条证据" in data["answer"]
    # 别的人群的证据仍在折叠区，可查
    assert {card["object_name"] for card in data["other_conditions"]} == {
        "孕妇",
        OTHER_POPULATION,
    }
    assert data["sources"] == []
    assert data["subgraph"] == {"nodes": [], "edges": []}


# ------------------------------------------------ 异常与无批次路径的字段形状
def test_verification_unavailable_has_no_verification_note(
    client, wire, api_module, monkeypatch
):
    """核验不可用时字段形状不变，且**不显示**通过核验的说明。"""
    wire(rows=rows_with_other_population())

    def _boom(*args, **kwargs):
        raise RuntimeError("核验层不可用（替身）")

    monkeypatch.setattr(api_module, "_high_risk_answer", _boom)

    code, data = ask(client, POPULATION_QUESTION, "auto")

    assert code == 200, data
    assert data["answer_status"] == "verification_unavailable"
    assert data["abstained"] is True
    assert data["verification_note"] == ""
    assert data["decision"] == "not_applicable"
    assert data["decision_text"] == ""
    assert data["answer_blocks"] == []
    assert data["other_conditions"] == []
    assert "通过程序核验" not in data["answer"]


def test_no_active_run_has_no_verification_note(client, wire):
    """没有 ACTIVE 批次时同样只补字段、不加说明。

    批次号由检索结果（`auto_run_id`）与库里的 ACTIVE 查询共同决定，两处都要
    置空才能构造出「没有生效批次」这一情形。
    """
    wire(rows=rows_with_other_population(), active_run_id=None, auto_run_id=None)

    code, data = ask(client, POPULATION_QUESTION, "auto")

    assert code == 200, data
    assert data["answer_status"] == "abstain"
    assert data["abstained"] is True
    assert data["verification_note"] == ""
    assert data["answer_blocks"] == []
    assert data["other_conditions"] == []


def test_benign_response_gains_no_high_risk_fields(client, wire):
    """⑮ 非高风险响应不新增字段，前端因此仍走原来的整体渲染路径。"""
    wire()

    code, data = ask(client, BENIGN_QUESTION, "auto")

    assert code == 200, data
    for field in (
        "decision",
        "decision_text",
        "verification_note",
        "answer_blocks",
        "other_conditions",
    ):
        assert field not in data, field
    assert data["answer"], "普通回答的内容不变"


# ------------------------------------------------------------ 前端纯函数执行
def _node_binary() -> str:
    found = shutil.which("node")
    if found:
        return found
    for candidate in (
        r"C:\nvm4w\nodejs\node.exe",
        r"C:\Program Files\nodejs\node.exe",
    ):
        if Path(candidate).exists():
            return candidate
    pytest.skip("未找到 Node.js：前端渲染函数的真实执行检查无法进行")


#: 抽取区段的锚点。从 `escapeHtml` 到 `hideWelcome` 之间全部是**纯函数**
#: （只依赖彼此，不碰 DOM），因此可以在 Node 里独立执行。`renderProvenance`
#: 与 `renderHighRiskBlock` 也在这一段里：回答正文、溯源入口与证据卡的渲染
#: 必须一起验，否则「卡片不见了」与「说明行一起丢了」分不开。
_SNIPPET_START = "        function escapeHtml(value) {"
_SNIPPET_END = "        function hideWelcome() {"

_DRIVER = r"""
const fs = require('fs');
const snippet = fs.readFileSync(process.argv[2], 'utf8');
const fns = new Function(
    snippet + `
return {escapeHtml, highlightEntities, stripMarkdownLite, normalizeAnswerText,
        renderAnswerBody, renderProvenance, renderHighRiskBlock, HR_CARDS_MODE};`
)();
const nodes = [{label: '防风'}, {label: '附子'}];
const blocks = [
  {kind: 'decision', text: '不能直接判断为可以使用。教材记载：孕妇应慎用附子。'},
  {kind: 'quote', text: '原文依据：「孕妇慎用。」（第 231 页）'},
  {kind: 'note', text: '未成立的部分：用量与用法未能核验。'}
];

/* 一条形态完整的证据单元。字段名与 `_card_payload`（后端）逐一对应，
   前端因此拿到的是与真实响应同形的对象，而不是为测试裁剪过的假数据。 */
const unit = (over) => Object.assign({
  unit_id: 'unit:1',
  risk_category: 'POPULATION',
  category_label: '人群禁忌',
  predicate: 'CONTRAINDICATED_IN',
  subject_name: '附子',
  subject_id: 'H:附子',
  object_id: 'P:孕妇',
  object_name: '孕妇',
  conditions: '孕妇',
  severity: 'CAUTION',
  severity_label: '慎用',
  polarity_verdict: 'NEGATIVE',
  polarity_basis: '教材原文',
  polarity_ambiguous: false,
  dosage_raw: null,
  page_label: '第 231 页',
  page_start: 231,
  page_end: 231,
  evidence_quote: '附子煎服，3～15g。',
  context: '附子煎服，3～15g。',
  source_kind: 'ASSERTION',
  assertion_id: 'a-p231',
  chunk_ids: ['c-1'],
  validation_status: 'VERIFIED',
  is_clearly_parsed: true,
  conflicts: [],
  undetermined_reason: '',
  is_citable: true,
  flags: []
}, over || {});

const DOSE_QUOTE = '原文依据：附子煎服，3～15g。（第 231 页）';
const FLAG = '原文说明：后续小句不是另一处独立剂量';

/* 四档状态各一份真实形态的载荷。grounded 的正文与证据卡讲的是**同一批
   证据**（这正是清单 §5⑨ 要断言「不再重复」的地方）。 */
const grounded = {
  high_risk: {is_high_risk: true},
  answer_status: 'grounded',
  decision_text: '不能直接判断为可以使用。教材记载：孕妇应慎用附子。',
  answer_blocks: [{kind: 'decision', text: '不能直接判断为可以使用。'}, {kind: 'quote', text: DOSE_QUOTE}],
  verification_note: '以上结论均通过程序核验。',
  sub_questions: [{question: '孕妇能否使用附子', status: 'grounded'}],
  evidence_units: [unit({flags: [FLAG]})],
  other_conditions: [unit({unit_id: 'unit:2', object_name: '阴虚阳亢者', evidence_quote: '阴虚阳亢者忌用。'})],
  rejections: [{claim_id: 'claim-1', notes: ['原文未给出剂量'], hard_failures: []}]
};
const grounded_plain = Object.assign({}, grounded, {other_conditions: [], rejections: []});
const partial = Object.assign({}, grounded, {
  answer_status: 'partial_grounded',
  sub_questions: [{question: '孕妇能否使用附子', status: 'conflict', reason: '「禁用」与「慎用」之间存在差异'}]
});
const source_only = Object.assign({}, grounded, {
  answer_status: 'source_only',
  sub_questions: [{question: '哺乳期能否使用附子', status: 'source_only', reason: '只有教材原文，没有可核验的结论'}],
  evidence_units: [],
  other_conditions: [unit({unit_id: 'unit:2', object_name: '孕妇'})]
});
const abstain = Object.assign({}, source_only, {
  answer_status: 'abstain',
  sub_questions: [{question: '哺乳期能否使用附子', status: 'abstain', reason: '没有与本题指定条件一致的可核对原文'}]
});
const unavailable = {
  high_risk: {is_high_risk: true},
  answer_status: 'verification_unavailable',
  answer_blocks: [],
  verification_note: '',
  sub_questions: [],
  evidence_units: [],
  other_conditions: [],
  rejections: []
};

const snapshot = (obj) => JSON.stringify(obj);
const grounded_before = snapshot(grounded);
const hr_grounded = fns.renderHighRiskBlock(grounded);
const hr_grounded_plain = fns.renderHighRiskBlock(grounded_plain);
const hr_partial = fns.renderHighRiskBlock(partial);
const hr_source_only = fns.renderHighRiskBlock(source_only);
const hr_abstain = fns.renderHighRiskBlock(abstain);
const hr_unavailable = fns.renderHighRiskBlock(unavailable);
const answer_grounded = fns.renderAnswerBody(grounded, nodes);

/* 同一批证据在两套渲染里各出现几次：「附子煎服，3～15g。」既在正文引文里，
   也在证据卡的原文里。grounded 之下应当只剩正文那一处。 */
const count = (html, needle) => html.split(needle).length - 1;

process.stdout.write(JSON.stringify({
  paired_bold: fns.normalizeAnswerText('**防风**可以祛风。'),
  lone_bold: fns.normalizeAnswerText('2 ** 3 = 8'),
  newline: fns.normalizeAnswerText('第一行\n第二行'),
  bold_and_highlight: fns.highlightEntities(
      fns.normalizeAnswerText('**防风**可以祛风。'), nodes),
  escaped_script: fns.highlightEntities('<script>alert(1)</script>防风', nodes),
  escaped_fallback: fns.renderAnswerBody({answer: '<img src=x onerror="alert(1)">'}, []),
  blocks_html: fns.renderAnswerBody(
      {answer_blocks: blocks, verification_note: '以上结论均通过程序核验。'}, nodes),
  no_note_html: fns.renderAnswerBody(
      {answer_blocks: [{kind: 'text', text: '证据不足。'}], verification_note: ''}, nodes),
  fallback_html: fns.renderAnswerBody({answer: '防风 的功效是祛风。'}, nodes),
  fallback_note_html: fns.renderAnswerBody(
      {answer: '防风 的功效是祛风。', verification_note: '不该显示的说明'}, nodes),

  /* ---- 行首 Markdown 标记：只做展示清理 ---- */
  md_h1: fns.stripMarkdownLite('# 人参（Rénshēn）中药学教材证据汇总'),
  md_h2: fns.normalizeAnswerText('## 一、来源与基原'),
  md_indented: fns.normalizeAnswerText('   ### 三、剂量与用法'),
  md_after_blank_line: fns.normalizeAnswerText('人参证据汇总\n\n# 人参（Rénshēn）中药学教材证据汇总'),
  md_after_single_newline: fns.normalizeAnswerText('人参证据汇总\n# 人参（Rénshēn）中药学教材证据汇总'),
  md_rule_dash: fns.normalizeAnswerText('上\n\n---\n\n下'),
  md_rule_star: fns.normalizeAnswerText('上\n\n***\n\n下'),
  md_rule_under: fns.normalizeAnswerText('上\n\n___\n\n下'),
  md_rule_indented: fns.normalizeAnswerText('上\n\n  ---   \n\n下'),
  md_rule_inline: fns.normalizeAnswerText('用量 3---5g 与 a***b 都保留'),
  md_bullet: fns.normalizeAnswerText('- 出处：人参\n- 出处：附子'),
  md_hash_inline: fns.normalizeAnswerText('编号#123 与 #1 均保留'),
  md_hash_no_space: fns.normalizeAnswerText('#人参'),
  md_hash_seven: fns.normalizeAnswerText('####### 七级不是标题'),
  md_heading_with_bold: fns.normalizeAnswerText('## 一、来源与基原\n**防风**可以祛风。'),
  md_paragraphs: fns.normalizeAnswerText('第一段。\n\n第二段。'),
  md_extra_blank_lines: fns.normalizeAnswerText('第一段。\n\n\n\n第二段。'),
  md_null: fns.normalizeAnswerText(null),
  md_script: fns.highlightEntities(
      fns.normalizeAnswerText('# <script>alert(1)</script>人参'), nodes),
  md_answer_body: fns.renderAnswerBody({answer: '# 人参\n\n---\n\n正文。'}, nodes),

  /* ---- 各状态的高风险区块 ---- */
  hr_grounded: hr_grounded,
  hr_grounded_plain: hr_grounded_plain,
  hr_partial: hr_partial,
  hr_source_only: hr_source_only,
  hr_abstain: hr_abstain,
  hr_unavailable: hr_unavailable,
  hr_ordinary: fns.renderHighRiskBlock({answer: '麻黄的功效是发汗解表。'}),
  hr_not_high_risk: fns.renderHighRiskBlock(
      {high_risk: {is_high_risk: false}, answer_status: 'grounded', evidence_units: [unit()]}),
  hr_cards_mode: fns.HR_CARDS_MODE,
  /* 展示去重不得改动证据数据本身：渲染前后载荷逐字节一致。 */
  payload_unchanged: grounded_before === snapshot(grounded),
  /* 同一批证据在「正文 + 高风险区块」里的出现次数。 */
  dose_in_grounded_page: count(answer_grounded + hr_grounded, '3～15g'),
  dose_in_partial_page: count(answer_grounded + hr_partial, '3～15g'),
  provenance_html: fns.renderProvenance(
      {sources: [{page_start: 231, page_end: 231, evidence_quote: '附子煎服，3～15g。'}],
       intent: 'dosage_query'},
      {nodes: [{label: '附子'}], edges: []})
}));
"""


@pytest.fixture(scope="module")
def frontend_rendered() -> dict:
    """把前端纯函数抽出来在 Node 里真实执行一次，返回各用例的输出。"""
    source = FRONTEND.read_text(encoding="utf-8")
    try:
        start = source.index(_SNIPPET_START)
        end = source.index(_SNIPPET_END, start)
    except ValueError as exc:  # 函数被移动或改名
        raise AssertionError(
            "frontend/index.html 中未找到待检查的纯函数区段"
            "（escapeHtml … hideWelcome）：若函数被移动或改名，"
            "请同步更新本用例的锚点，而不是删掉这条检查"
        ) from exc

    text = source[start:end]
    # 区段能被 Node 执行的前提是它不碰 DOM。若哪天有函数把 `document` 带进来，
    # 这里立刻指向真正的原因，而不是让 driver 在 Node 里报一句看不懂的错。
    assert "document." not in text, "抽取区段里出现了 DOM 访问，锚点需要调整"

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        snippet = Path(tmp) / "snippet.js"
        driver = Path(tmp) / "driver.js"
        snippet.write_text(text, encoding="utf-8")
        driver.write_text(_DRIVER, encoding="utf-8")
        proc = subprocess.run(
            [_node_binary(), str(driver), str(snippet)],
            capture_output=True,
            timeout=60,
        )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    return json.loads(proc.stdout.decode("utf-8"))


def test_paired_bold_is_removed_but_lone_stars_are_kept(frontend_rendered):
    """⑫ 成对 `**` 只做展示处理：成对去掉，落单的原样保留。"""
    assert frontend_rendered["paired_bold"] == "防风可以祛风。"
    # 落单的星号不是加粗标记，不能被吃掉
    assert frontend_rendered["lone_bold"] == "2 ** 3 = 8"
    # 换行不受影响
    assert frontend_rendered["newline"] == "第一行\n第二行"


def test_bold_text_is_still_entity_highlighted(frontend_rendered):
    """⑫ 去掉 `**` 之后实体高亮照常生效，且不残留星号。"""
    html = frontend_rendered["bold_and_highlight"]

    assert '<mark class="entity-mark">防风</mark>' in html
    assert "**" not in html


def test_model_text_is_still_escaped(frontend_rendered):
    """⑬ `<script>` 等仍被转义：展示处理不得引入 XSS。"""
    html = frontend_rendered["escaped_script"]

    assert "&lt;script&gt;" in html
    assert "<script>" not in html
    # 属性注入同样被转义
    fallback = frontend_rendered["escaped_fallback"]
    assert "<img" not in fallback
    assert "&lt;img" in fallback


def test_answer_blocks_are_rendered_in_order_with_own_classes(frontend_rendered):
    """⑪ 分块各有独立类名；核验说明单独成块并位于末尾。"""
    html = frontend_rendered["blocks_html"]

    decision = html.index('class="answer-decision"')
    quote = html.index('class="answer-evidence-quote"')
    note = html.index('class="answer-note"')
    verification = html.index('class="answer-verification-note"')
    assert decision < quote < note < verification, html
    assert "<p class=\"answer-verification-note\"" not in html, "核验说明应是独立 div"
    assert "以上结论均通过程序核验。" in html
    # 原文依据在引文里保留了标签与页码
    assert "原文依据：" in html and "第 231 页" in html


def test_verification_note_is_hidden_when_absent(frontend_rendered):
    """⑩ 没有核验说明时不得渲染出空白说明块。"""
    assert "answer-verification-note" not in frontend_rendered["no_note_html"]


def test_ordinary_answer_keeps_the_original_rendering(frontend_rendered):
    """⑮ 普通回答（没有分块）走原来的整体渲染：实体高亮、换行都在。"""
    html = frontend_rendered["fallback_html"]

    assert '<mark class="entity-mark">防风</mark>' in html
    assert "answer-decision" not in html
    # 普通回答的 verification_note 字段不存在时，绝不会凭空渲染出核验说明
    assert "answer-verification-note" not in frontend_rendered["fallback_note_html"]


def test_frontend_styles_and_wording_are_in_place():
    """⑪⑭ 灰色样式类与面向读者的文案确实在页面里。

    这里只检查**存在性**（样式本身由上面的执行用例检查行为），
    因为视觉效果无法在无头环境里可靠断言，不能假装已经验过。
    """
    source = FRONTEND.read_text(encoding="utf-8")

    for css_class in (
        ".answer-decision",
        ".answer-conclusion",
        ".answer-evidence-quote",
        ".answer-note",
        ".answer-verification-note",
        ".hr-card-quote-label",
    ):
        assert css_class in source, css_class

    # 核验说明用既有 muted 变量，字号略小、字重正常
    note_rule = source.split(".answer-verification-note {", 1)[1].split("}", 1)[0]
    assert "var(--muted)" in note_rule
    assert "font-weight: 400" in note_rule

    # 证据卡的原文统一带标签
    assert '<span class="hr-card-quote-label">原文依据：</span>' in source

    # ⑭ 普通用户文案：不再要求人工核对，改为系统自己的处置说明
    assert "待核对" not in source
    assert "人工核对" not in source
    # ⑭ 说明行由后端定稿：前端只显示 `flags`，不对 `conflicts` 各自套模板。
    # 套模板会把解析层说明（如「后续小句不是另一处独立剂量」）说成
    # 「证据存在差异」，与同一张卡的「已核验」徽标自相矛盾。
    assert "unit.flags" in source
    assert "证据存在差异：${escapeHtml(conflict)}" not in source
    assert "系统暂不作单一结论" not in source


# ------------------------------------------------------ 行首 Markdown 标记
def test_heading_markers_are_not_shown(frontend_rendered):
    """① `# 人参（Rénshēn）中药学教材证据汇总` 只显示标题正文。
    ② `## 一、来源与基原` 同样：文字保留，`##` 不显示。
    """
    r = frontend_rendered

    assert r["md_h1"] == "人参（Rénshēn）中药学教材证据汇总"
    assert r["md_h2"] == "一、来源与基原"
    assert r["md_indented"] == "三、剂量与用法"
    assert "#" not in r["md_h1"]
    assert "#" not in r["md_h2"]


def test_heading_cleanup_never_swallows_the_line_break(frontend_rendered):
    """① 标题与上一段必须仍是两段。

    这是这类清理最常见的错：前导空白用 `\\s` 会连行尾换行一起吃，
    「…汇总\\n\\n# 人参」会变成「…汇总人参」，标题被并进上一段末尾。
    """
    r = frontend_rendered

    assert r["md_after_blank_line"] == "人参证据汇总\n\n人参（Rénshēn）中药学教材证据汇总"
    assert r["md_after_single_newline"] == "人参证据汇总\n人参（Rénshēn）中药学教材证据汇总"


def test_standalone_rule_lines_are_removed(frontend_rendered):
    """③ 独占一行的 `---` / `***` / `___`（含缩进与行尾空格）整体删除。"""
    r = frontend_rendered

    for key in ("md_rule_dash", "md_rule_star", "md_rule_under", "md_rule_indented"):
        assert r[key] == "上\n\n下", (key, r[key])
    # 行内的短横线与星号是内容本身，不是分隔线
    assert r["md_rule_inline"] == "用量 3---5g 与 a***b 都保留"


def test_normal_bullets_are_kept(frontend_rendered):
    """④ 正常列表里的 `-` 保留：清理只针对**整行**都是分隔线的情况。"""
    assert frontend_rendered["md_bullet"] == "- 出处：人参\n- 出处：附子"


def test_hashes_inside_text_are_never_removed(frontend_rendered):
    """⑤ 「编号#123」「#1」以及 `#` 后没有空格的 `#人参` 都不是标题。"""
    r = frontend_rendered

    assert r["md_hash_inline"] == "编号#123 与 #1 均保留"
    assert r["md_hash_no_space"] == "#人参"
    assert r["md_hash_seven"] == "####### 七级不是标题"


def test_cleanup_composes_with_bold_and_keeps_paragraphs(frontend_rendered):
    """⑥ 标题清理与成对 `**` 清理叠加；空行只做合并，段落不被压成一段。"""
    r = frontend_rendered

    assert r["md_heading_with_bold"] == "一、来源与基原\n防风可以祛风。"
    assert r["md_paragraphs"] == "第一段。\n\n第二段。"
    assert r["md_extra_blank_lines"] == "第一段。\n\n第二段。"
    assert r["md_null"] == ""


def test_markdown_cleanup_does_not_open_an_injection_path(frontend_rendered):
    """⑦ 清理在先、转义在后：标题里的 `<script>` 仍被转义，不产生注入。"""
    html = frontend_rendered["md_script"]

    assert "&lt;script&gt;" in html
    assert "<script>" not in html
    # 整段回答也走同一条路：标记消失了，正文还在
    body = frontend_rendered["md_answer_body"]
    assert "人参" in body and "正文。" in body
    assert "#" not in body and "---" not in body


# ------------------------------------------------- 各状态下的证据卡与徽标
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)


def visible(html: str) -> str:
    """去掉 HTML 注释之后的标记串。

    注释不会渲染给用户，但它确实在渲染函数的输出里（卡片模板里带一段
    解释性注释）。「页面上不出现『已核验』」这类断言必须针对**渲染结果**，
    否则会被注释里的说明文字误伤——那会把一条正确的实现判成失败。
    """
    return _HTML_COMMENT_RE.sub("", html)


def test_grounded_page_shows_no_duplicate_evidence_cards(frontend_rendered):
    """⑧ grounded：徽标在、主证据卡不渲染、说明行与折叠区都还在。

    说明行为什么必须留：真正的「证据存在差异」也走 `flags` 通道，
    如果跟着卡片一起消失，用户会以为程序核验对这些证据没有任何保留意见。
    """
    html = frontend_rendered["hr_grounded"]

    assert frontend_rendered["hr_cards_mode"]["grounded"] == "none"
    assert "hr-badge hr-ok" in html
    assert "高风险 · 已核验" in html
    # 主证据卡（evidence_units）不再渲染，也没有为它套一个折叠区
    assert "hr-structured" not in html
    assert frontend_rendered["hr_grounded_plain"].count('class="hr-card"') == 0
    # 「其他使用注意」本来就与正文不重复，仍以折叠卡保留（1 条）
    assert html.count('class="hr-card"') == 1
    assert 'class="hr-other"' in html
    # 说明行仍在页面上
    assert DOSE_FLAG in html
    assert 'class="hr-flag"' in html
    # 同一批证据（3～15g）在「正文 + 高风险区块」里只剩正文那一处
    assert frontend_rendered["dose_in_grounded_page"] == 1


def test_grounded_rendering_does_not_mutate_the_evidence_payload(frontend_rendered):
    """⑨ 去重是**纯展示**：渲染前后 evidence_units 等证据数据逐字节不变。"""
    assert frontend_rendered["payload_unchanged"] is True


def test_grounded_response_keeps_the_full_evidence_payload(client, wire):
    """⑨ 前端不渲染卡片 ≠ 接口不给证据：字段一个不少，内容不变。

    接口契约、同源证据核验与评测读的都是这些字段；本用例把它们逐个钉住，
    以免将来有人把「前端不显示」顺手做成「后端不返回」。
    """
    wire(rows=rows_with_other_population())

    code, data = ask(client, POPULATION_QUESTION, "auto")

    assert code == 200, data
    assert data["answer_status"] == "grounded", data["coverage_notes"]
    for field in (
        "evidence_units",
        "claims",
        "sources",
        "subgraph",
        "answer_blocks",
        "evidence_method",
        "answer_status",
        "abstained",
    ):
        assert field in data, field
    units = data["evidence_units"]
    assert units, "证据单元仍在响应里，只是前端不再重复展开"
    assert units[0]["evidence_quote"] == "孕妇慎用。"
    assert units[0]["page_start"] == 231
    assert isinstance(units[0]["flags"], list)
    # 卡片与折叠区共用同一形状：没有字段为「少显示几张卡」而被删改
    assert set(units[0]) == set(data["other_conditions"][0])
    assert data["claims"], "程序核验通过的结论仍在"
    assert data["sources"], "顶层来源仍在"
    assert data["subgraph"]["nodes"], "子图仍在"
    assert data["abstained"] is False
    # 程序生成的正文本身就不含行首标记：前端清理是给模型文本兜底，不改变它
    assert not [line for line in data["answer"].splitlines() if line.strip().startswith("#")]
    assert not [line for line in data["answer"].splitlines() if line.strip() in {"---", "***", "___"}]


def test_partial_grounded_keeps_conflicts_and_folds_the_cards(frontend_rendered):
    """⑩ partial_grounded：仍显示「部分核验」与冲突原因，卡片默认折叠。

    折叠区必须**默认收着**：程度冲突要先读正文原因，卡片是备查的条件与来源，
    不是第一屏内容。所以这里断言的是完整标签串（多了 `open` 就不匹配）。
    """
    html = frontend_rendered["hr_partial"]

    assert frontend_rendered["hr_cards_mode"]["partial_grounded"] == "folded"
    assert "hr-badge hr-warn" in html
    assert "高风险 · 部分核验" in html
    assert "「禁用」与「慎用」之间存在差异" in html
    assert '<p class="hr-note">' in html
    assert '<details class="hr-structured">' in html
    folded = html.split('<details class="hr-structured">', 1)[1].split("</details>", 1)[0]
    assert "查看结构化证据" in folded
    assert "原文依据：" in folded  # 卡片里的原文仍在折叠区里可查
    assert DOSE_FLAG in folded
    # 卡片与正文各出现一次：partial 不隐藏证据，只是折叠
    assert frontend_rendered["dose_in_partial_page"] == 2


def test_source_only_and_abstain_never_read_as_verified(frontend_rendered):
    """⑪ source_only / abstain / 核验不可用都不得显示成「已核验」。"""
    r = frontend_rendered
    source_only = visible(r["hr_source_only"])
    abstain = visible(r["hr_abstain"])
    unavailable = visible(r["hr_unavailable"])

    assert "hr-badge hr-warn" in source_only
    assert "高风险 · 仅有原文，未形成结论" in source_only
    assert "只有教材原文，没有可核验的结论" in source_only
    assert "已核验" not in source_only

    assert "hr-badge hr-bad" in abstain
    assert "高风险 · 证据不足，不予结论" in abstain
    assert "没有与本题指定条件一致的可核对原文" in abstain
    assert "已核验" not in abstain
    # 「未通过程序核验」是另一回事：被驳回的候选结论要把原因露出来，
    # 不能因为「别显示通过核验」就把失败记录一起藏掉（那才叫掩盖）。
    assert "以上结论均通过程序核验" not in abstain
    assert "本回答中 1 条候选结论未通过程序核验" in abstain

    assert "hr-badge hr-bad" in unavailable
    assert "高风险 · 未通过核验（核验层不可用）" in unavailable
    assert "已核验" not in unavailable


def test_risk_sections_survive_alongside_the_hidden_cards(frontend_rendered):
    """§3 「其他使用注意」「未通过核验的候选结论」不因隐藏卡片而消失。"""
    html = frontend_rendered["hr_grounded"]

    assert "其他使用注意（与本题指定人群不同）· 1 条" in html
    assert "阴虚阳亢者忌用。" in html
    assert "本回答中 1 条候选结论未通过程序核验" in html
    assert "原文未给出剂量" in html
    # 被驳回的原因与原证据卡一样带「原文依据」标签，不混进结论
    assert "class=\"hr-card-quote-label\"" not in html.split('class="hr-other"', 1)[0]


def test_provenance_entry_still_renders_without_the_cards(frontend_rendered):
    """⑤ grounded 不列卡之后，检索溯源入口照旧可用。"""
    html = frontend_rendered["provenance_html"]

    assert '<details class="trace-block">' in html
    assert "检索溯源" in html
    assert "第 231 页" in html
    assert "附子煎服，3～15g。" in html


def test_ordinary_answers_and_graph_features_are_untouched(frontend_rendered):
    """⑫ 普通回答不产生高风险区块；图谱动画、统计面板与检索模式选择器未受影响。

    这几处是队友的展示资产，本轮只做回答区的收口，必须留下可核对的原地锚点。
    """
    r = frontend_rendered

    assert r["hr_ordinary"] == ""
    assert r["hr_not_high_risk"] == ""

    source = FRONTEND.read_text(encoding="utf-8")
    for anchor in (
        'id="demo-graph"',
        ".demo-orbit",
        "getElementById('demo-graph')",
        "d3.forceSimulation",
        "function initGraph()",
        'id="retrieval-method"',
        "加载图谱统计...",
    ):
        assert anchor in source, anchor


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
