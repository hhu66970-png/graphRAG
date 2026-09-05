from graphragexpr.retrieval.comparison import reduce_section_summaries


def test_section_map_reduce_keeps_both_entities_and_source_spans():
    rows = [
        {
            "entity_id": "HERB:renshen",
            "entity_name": "人参",
            "chunk_id": "c1",
            "text": "人参能大补元气。另有无关句。",
            "page_start": 378,
            "page_end": 378,
            "section_path": ["补虚药", "人参"],
        },
        {
            "entity_id": "HERB:huangqi",
            "entity_name": "黄芪",
            "chunk_id": "c2",
            "text": "黄芪能补气升阳。另有无关句。",
            "page_start": 382,
            "page_end": 382,
            "section_path": ["补虚药", "黄芪"],
        },
    ]

    summaries = reduce_section_summaries(
        "比较人参和黄芪的功效",
        rows,
        per_entity=2,
        total_limit=4,
    )

    assert {item["subject_id"] for item in summaries} == {
        "HERB:renshen",
        "HERB:huangqi",
    }
    assert all(
        item["quote"] in rows[index]["text"] for index, item in enumerate(summaries)
    )
    assert {item["chunk_ids"][0] for item in summaries} == {"c1", "c2"}
