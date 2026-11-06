# tests/test_retrieval.py
"""
Tests for RAG retrieval components
"""
import pytest
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent / "graphragexpr"))

from custom_embedder import SimpleHashEmbedder

pytestmark = pytest.mark.integration


class TestRetrieval:
    """Test suite for RAG retrieval"""

    def test_vector_index_creation(self, neo4j_driver, clean_database, neo4j_test_config):
        """Test creating vector index"""
        from neo4j_graphrag.indexes import create_vector_index

        token = uuid4().hex
        label = f"ITestRetrievalVector_{token}"
        index_name = f"itest_retrieval_vector_{token}"
        database = neo4j_test_config["database"]
        embedding = SimpleHashEmbedder(dimension=128).embed_query("test")

        try:
            with neo4j_driver.session(database=database) as session:
                # These identifiers contain only the fixed prefix and a generated UUID.
                session.run(
                    f"CREATE (c:`{label}` {{text: 'test', embedding: $embedding}})",
                    embedding=embedding,
                ).consume()
            create_vector_index(
                neo4j_driver,
                name=index_name,
                label=label,
                embedding_property="embedding",
                dimensions=128,
                similarity_fn="cosine",
                fail_if_exists=True,
                neo4j_database=database,
            )
            with neo4j_driver.session(database=database) as session:
                session.run("CALL db.awaitIndex($name, 30)", name=index_name).consume()
                index = session.run(
                    "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, options "
                    "WHERE name = $name RETURN type, entityType, labelsOrTypes, properties, options",
                    name=index_name,
                ).single()
                assert index is not None
                assert index["type"] == "VECTOR"
                assert index["entityType"] == "NODE"
                assert index["labelsOrTypes"] == [label]
                assert index["properties"] == ["embedding"]
                config = index["options"]["indexConfig"]
                assert config["vector.dimensions"] == 128
                assert config["vector.similarity_function"].lower() == "cosine"
        finally:
            with neo4j_driver.session(database=database) as session:
                session.run(f"DROP INDEX `{index_name}` IF EXISTS").consume()

    def test_cypher_reduce_function(self, neo4j_driver, clean_database, neo4j_test_config):
        """Test the reduce function used in retrieval query"""
        with neo4j_driver.session(database=neo4j_test_config["database"]) as session:
            # Test reduce for text joining
            result = session.run("""
                WITH ['text1', 'text2', 'text3'] AS texts
                WITH reduce(s = '', text IN texts |
                    s + CASE WHEN s = '' THEN text ELSE ', ' + text END
                ) AS joined
                RETURN joined
            """)
            joined = result.single()["joined"]
            assert joined == "text1, text2, text3"

    def test_graph_traversal_query(self, neo4j_driver, clean_database, neo4j_test_config):
        """Test graph traversal for RAG retrieval"""
        embedder = SimpleHashEmbedder(dimension=128)
        embedding = embedder.embed_query("test text")

        with neo4j_driver.session(database=neo4j_test_config["database"]) as session:
            # Create knowledge graph structure
            session.run("""
                CREATE (c:Chunk {text: '人参具有大补元气的功效', embedding: $embedding})
                CREATE (drug:中药 {name: '人参'})
                CREATE (company:功效 {name: '大补元气'})
                CREATE (drug)-[:FROM_CHUNK]->(c)
                CREATE (drug)-[:具有功效]->(company)
            """, embedding=embedding)

            # Test 1-2 hop traversal from chunk
            result = session.run("""
                MATCH (chunk:Chunk)<-[:FROM_CHUNK]-(entity)-[rel*1..2]-(neighbor)
                RETURN DISTINCT neighbor.name as name
            """)

            entities = {record["name"] for record in result if record["name"]}
            assert "大补元气" in entities

    def test_retrieval_query_format(self, neo4j_driver, clean_database, neo4j_test_config):
        """Test the complete retrieval query format"""
        embedder = SimpleHashEmbedder(dimension=128)
        embedding = embedder.embed_query("人参")

        with neo4j_driver.session(database=neo4j_test_config["database"]) as session:
            # Setup test data
            session.run("""
                CREATE (c:Chunk {text: '人参具有大补元气的功效', embedding: $embedding})
                CREATE (drug:中药 {name: '人参'})
                CREATE (mechanism:功效 {name: '大补元气'})
                CREATE (drug)-[:FROM_CHUNK]->(c)
                CREATE (drug)-[:具有功效]->(mechanism)
            """, embedding=embedding)

            # Test retrieval query (simplified version)
            result = session.run("""
                MATCH (chunk:Chunk)<-[:FROM_CHUNK]-(entity)-[rel*1..2]-(neighbor)
                WITH chunk, rel
                UNWIND rel AS r
                WITH collect(DISTINCT chunk.text) AS texts,
                     collect(DISTINCT
                       coalesce(startNode(r).name, '') + ' -[' + type(r) + ']-> ' +
                       coalesce(endNode(r).name, '')
                     ) AS triples
                WITH reduce(s = '', text IN texts |
                        s + CASE WHEN s = '' THEN text ELSE '\\n' + text END
                     ) AS joined_texts,
                     reduce(s = '', triple IN triples |
                        s + CASE WHEN s = '' THEN triple ELSE '\\n' + triple END
                     ) AS joined_triples
                RETURN joined_texts, joined_triples
            """)

            record = result.single()
            assert "人参具有大补元气的功效" in record["joined_texts"]
            assert "具有功效" in record["joined_triples"]
