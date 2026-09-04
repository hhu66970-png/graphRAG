# custom_embedder.py
"""
Custom embedder that creates simple embeddings without external API calls
For demonstration purposes - uses basic text hashing for embeddings
"""
import hashlib
import numpy as np
from neo4j_graphrag.embeddings.base import Embedder


class SimpleHashEmbedder(Embedder):
    """
    Simple embedder that creates deterministic embeddings from text hashing.
    This is for demonstration purposes when OpenAI embeddings are not available.
    """

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension

    def embed_query(self, text: str) -> list[float]:
        """Generate a simple embedding from text using hashing"""
        # Create a deterministic hash-based embedding
        # Use multiple hash functions to fill the embedding vector
        embedding = []

        for i in range(self.dimension // 16):
            # Create hash with different seeds
            hash_input = f"{text}_{i}".encode('utf-8')
            hash_obj = hashlib.sha256(hash_input)
            hash_bytes = hash_obj.digest()

            # Convert bytes to float values between -1 and 1
            for j in range(min(16, self.dimension - len(embedding))):
                if j < len(hash_bytes):
                    # Normalize to [-1, 1] range
                    val = (hash_bytes[j] / 255.0) * 2 - 1
                    embedding.append(val)

        # Fill remaining dimensions if needed
        while len(embedding) < self.dimension:
            embedding.append(0.0)

        # Normalize the vector
        embedding = np.array(embedding)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()
