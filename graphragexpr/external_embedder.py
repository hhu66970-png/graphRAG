"""Embedding providers for local, external API, and explicit demo hash modes."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class ExternalEmbedder:
    """Generate matching query/document vectors with the configured provider."""

    def __init__(self, dimension: int | None = None):
        """初始化该对象，保存后续调用所需的配置或依赖。"""
        self.dimension = dimension or int(os.getenv("EMBED_DIM", "512"))
        self.provider = (os.getenv("EMBED_PROVIDER") or "hash").strip().lower()
        self.endpoint = (os.getenv("EMBED_ENDPOINT") or "").strip()
        self.model_name = (os.getenv("EMBED_MODEL") or "").strip()
        self.token = (os.getenv("EMBED_TOKEN") or "").strip()
        self.query_prefix = os.getenv("EMBED_QUERY_PREFIX")

        if self.provider == "local":
            if not self.model_name:
                raise RuntimeError("EMBED_PROVIDER=local 时必须配置 EMBED_MODEL")
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "本地向量模式需要 sentence-transformers，请重新安装 requirements.txt"
                ) from exc

            cache_dir = os.getenv("EMBED_CACHE_DIR") or None
            if cache_dir:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
            device = (os.getenv("EMBED_DEVICE") or "cpu").strip()
            local_files_only = (os.getenv("EMBED_LOCAL_FILES_ONLY") or "false").lower() in {
                "1", "true", "yes"
            }
            self.local_model = SentenceTransformer(
                self.model_name,
                device=device,
                cache_folder=cache_dir,
                local_files_only=local_files_only,
            )
            if hasattr(self.local_model, "get_embedding_dimension"):
                actual_dimension = self.local_model.get_embedding_dimension()
            else:
                actual_dimension = self.local_model.get_sentence_embedding_dimension()
            if actual_dimension != self.dimension:
                raise RuntimeError(
                    f"向量维度不一致：配置为 {self.dimension}，模型实际为 {actual_dimension}"
                )
            if self.query_prefix is None and "bge" in self.model_name.lower():
                self.query_prefix = "为这个句子生成表示以用于检索相关文章："
            self.query_prefix = self.query_prefix or ""
            print(
                f"[ok] 使用本地向量模型: {self.model_name} "
                f"({self.dimension} 维, {device})"
            )
        elif self.provider == "external":
            missing = [
                name
                for name, value in {
                    "EMBED_ENDPOINT": self.endpoint,
                    "EMBED_MODEL": self.model_name,
                    "EMBED_TOKEN": self.token,
                }.items()
                if not value
            ]
            if missing:
                raise RuntimeError(f"外部向量接口缺少配置：{', '.join(missing)}")
            self.client = OpenAI(api_key=self.token, base_url=self.endpoint)
            self.query_prefix = self.query_prefix or ""
            print(f"[ok] 使用外部向量模型: {self.model_name}")
        elif self.provider == "hash":
            from custom_embedder import SimpleHashEmbedder

            self.fallback = SimpleHashEmbedder(dimension=self.dimension)
            self.query_prefix = ""
            print("[warn] 使用显式 hash 演示向量；该模式不具备语义检索能力")
        else:
            raise RuntimeError(
                f"未知 EMBED_PROVIDER={self.provider!r}，应为 local、external 或 hash"
            )

    def _validate(self, embedding: Iterable[float]) -> list[float]:
        """校验嵌入向量维度与提供方配置。"""
        vector = np.asarray(list(embedding), dtype=np.float32)
        if vector.shape != (self.dimension,):
            raise RuntimeError(
                f"向量维度不一致：期望 {self.dimension}，实际 {vector.size}"
            )
        norm = np.linalg.norm(vector)
        if norm:
            vector = vector / norm
        return vector.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Encode a search query."""
        query = f"{self.query_prefix}{text}"
        if self.provider == "local":
            vector = self.local_model.encode(
                query,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return self._validate(vector)
        if self.provider == "external":
            response = self.client.embeddings.create(model=self.model_name, input=query)
            return self._validate(response.data[0].embedding)
        return self.fallback.embed_query(text)

    def embed_document(self, text: str) -> list[float]:
        """Encode one corpus document without a retrieval-query instruction."""
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        """Encode corpus documents in a batch when the provider supports it."""
        documents = list(texts)
        if self.provider == "local":
            vectors = self.local_model.encode(
                documents,
                batch_size=int(os.getenv("EMBED_BATCH_SIZE", "16")),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=len(documents) > 8,
            )
            return [self._validate(vector) for vector in vectors]
        if self.provider == "external":
            response = self.client.embeddings.create(
                model=self.model_name,
                input=documents,
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            return [self._validate(item.embedding) for item in ordered]
        return [self.fallback.embed_query(text) for text in documents]
