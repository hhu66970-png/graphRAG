"""测试嵌入器提供方切换（含哈希演示向量）。"""

from external_embedder import ExternalEmbedder


def test_explicit_hash_provider_is_available_for_tests(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    monkeypatch.setenv("EMBED_DIM", "32")

    embedder = ExternalEmbedder(dimension=32)
    documents = embedder.embed_documents(["中药", "中药饮片"])

    assert embedder.provider == "hash"
    assert len(documents) == 2
    assert all(len(vector) == 32 for vector in documents)
