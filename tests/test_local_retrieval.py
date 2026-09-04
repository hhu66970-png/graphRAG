from graphragexpr.local_retrieval import rank_chunks


def test_definition_question_prefers_definition_chunk():
    chunks = [
        {
            "index": 0,
            "text": "中药材是中药饮片的原料。中药饮片可用于中医临床或制成中成药。",
        },
        {
            "index": 1,
            "text": "云南出产三七、茯苓、诃子、儿茶和石斛等道地药材。",
        },
        {
            "index": 2,
            "text": "温度和湿度会影响中药的贮藏质量。",
        },
    ]

    ranked = rank_chunks(
        "中药、中药材、中药饮片和中成药分别是什么？有什么区别？",
        chunks,
        exact_terms=["中药", "中药材", "中药饮片", "中成药"],
        top_k=2,
    )

    assert ranked[0]["index"] == 0
    assert "中药饮片" in ranked[0]["matched_terms"]
