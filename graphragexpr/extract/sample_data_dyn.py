"""Compatibility wrapper for the teacher repository's dynamic data loader."""

from corpus import records_from_environment


def get_chunk_records():
    """兼容老师仓库接口，返回带元数据的文本块。"""
    return records_from_environment()


def get_chunks():
    """兼容老师仓库接口，只返回块文本列表。"""
    return [record.text for record in get_chunk_records()]


if __name__ == "__main__":
    records = get_chunk_records()
    print(f"已生成 {len(records)} 个文本块")
    for record in records[:3]:
        print(
            f"\nChunk {record.index} | PDF 页 {record.page_start}-{record.page_end} | "
            f"章节：{record.section or '未识别'}\n{record.text[:240]}"
        )
