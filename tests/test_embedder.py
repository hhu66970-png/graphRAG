# tests/test_embedder.py
"""
Tests for custom embedder
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "graphragexpr"))

from custom_embedder import SimpleHashEmbedder


class TestSimpleHashEmbedder:
    """Test suite for SimpleHashEmbedder"""

    def test_embedder_initialization(self):
        """Test embedder can be initialized with different dimensions"""
        embedder = SimpleHashEmbedder(dimension=512)
        assert embedder.dimension == 512

        embedder = SimpleHashEmbedder(dimension=1536)
        assert embedder.dimension == 1536

    def test_embed_query_returns_correct_dimension(self):
        """Test that embedding has correct dimension"""
        embedder = SimpleHashEmbedder(dimension=1536)
        embedding = embedder.embed_query("test text")

        assert len(embedding) == 1536
        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_query_is_normalized(self):
        """Test that embedding vector is normalized"""
        embedder = SimpleHashEmbedder(dimension=1536)
        embedding = embedder.embed_query("test text")

        # Calculate L2 norm
        norm = np.linalg.norm(embedding)

        # Should be approximately 1.0 (allowing for floating point precision)
        assert abs(norm - 1.0) < 1e-6

    def test_embed_query_is_deterministic(self):
        """Test that same text produces same embedding"""
        embedder = SimpleHashEmbedder(dimension=1536)

        embedding1 = embedder.embed_query("test text")
        embedding2 = embedder.embed_query("test text")

        assert embedding1 == embedding2

    def test_embed_query_different_text(self):
        """Test that different text produces different embeddings"""
        embedder = SimpleHashEmbedder(dimension=1536)

        embedding1 = embedder.embed_query("人参")
        embedding2 = embedder.embed_query("黄芪")

        assert embedding1 != embedding2

    def test_embed_query_chinese_text(self):
        """Test embedder works with Chinese text"""
        embedder = SimpleHashEmbedder(dimension=1536)

        text = "人参具有大补元气的功效"
        embedding = embedder.embed_query(text)

        assert len(embedding) == 1536
        assert abs(np.linalg.norm(embedding) - 1.0) < 1e-6

    def test_embed_query_empty_string(self):
        """Test embedder handles empty string"""
        embedder = SimpleHashEmbedder(dimension=1536)

        embedding = embedder.embed_query("")

        assert len(embedding) == 1536
        assert abs(np.linalg.norm(embedding) - 1.0) < 1e-6
