"""图检索查询不应被「功效」「性味」等槽位词带跑。"""

from backend.api import (
    _build_fulltext_query,
    _chunks_mention_anchors,
    _is_slot_query_term,
)


def test_slot_words_are_not_search_anchors():
    assert _is_slot_query_term("功效")
    assert _is_slot_query_term("性味")
    assert not _is_slot_query_term("人参")


def test_fulltext_query_uses_herb_name_not_slot_bigrams():
    query = _build_fulltext_query("人参的性味、归经和功效是什么？", ["人参", "功效", "性味"])
    assert "人参" in query
    assert "功效" not in query
    assert "性味" not in query


def test_fulltext_query_without_anchors_still_uses_question_bigrams():
    query = _build_fulltext_query("什么是中药", [])
    assert query


def test_chunks_are_filtered_to_asked_herb():
    chunks = [
        {"id": "a", "text": "大黄苦寒，泻下攻积。"},
        {"id": "b", "text": "人参甘温，大补元气。"},
    ]
    kept = _chunks_mention_anchors(chunks, ["人参"])
    assert [chunk["id"] for chunk in kept] == ["b"]
