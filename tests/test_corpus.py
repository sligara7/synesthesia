"""
Unit tests for CorpusService
"""

import pytest
import tempfile
import os
import sys
sys.path.insert(0, 'src')

from structural_rorschach.corpus import Corpus, CorpusService, create_corpus, load_corpus, save_corpus
from structural_rorschach.signature import StructuralSignature


@pytest.fixture
def corpus_service():
    """Create a CorpusService instance."""
    return CorpusService()


@pytest.fixture
def sample_signature():
    """Create a sample signature."""
    return StructuralSignature(
        source_domain="text",
        source_id="doc_1",
        source_name="Document 1",
        num_nodes=10,
        num_edges=15,
        density=0.333,
        clustering_coefficient=0.5,
        avg_path_length=2.5,
        is_dag=True,
        num_communities=2,
        motif_vector={"hub_spoke": 0.3, "chain_3": 0.4},
        spectral_signature=[0.5, 0.3, 0.2],
        degree_distribution=[1, 2, 3, 2, 1, 1]
    )


@pytest.fixture
def sample_signature_2():
    """Create another sample signature."""
    return StructuralSignature(
        source_domain="text",
        source_id="doc_2",
        source_name="Document 2",
        num_nodes=12,
        num_edges=18,
        density=0.273,
        clustering_coefficient=0.45,
        avg_path_length=2.8,
        is_dag=True,
        num_communities=3,
        motif_vector={"hub_spoke": 0.35, "chain_3": 0.35},
        spectral_signature=[0.48, 0.32, 0.18],
        degree_distribution=[2, 2, 3, 3, 1, 1]
    )


class TestCorpus:
    """Tests for Corpus dataclass."""

    def test_corpus_creation(self):
        """Test creating a Corpus."""
        corpus = Corpus(name="test", domain="text", description="Test corpus")
        assert corpus.name == "test"
        assert corpus.domain == "text"
        assert len(corpus) == 0

    def test_corpus_to_dict(self, sample_signature):
        """Test serializing corpus to dict."""
        corpus = Corpus(name="test", domain="text")
        corpus.signatures.append(sample_signature)
        corpus._index[sample_signature.source_id] = 0

        data = corpus.to_dict()
        assert data["name"] == "test"
        assert data["domain"] == "text"
        assert len(data["signatures"]) == 1

    def test_corpus_from_dict(self, sample_signature):
        """Test deserializing corpus from dict."""
        data = {
            "name": "test",
            "domain": "text",
            "description": "Test corpus",
            "signatures": [sample_signature.to_dict()],
            "metadata": {}
        }
        corpus = Corpus.from_dict(data)
        assert corpus.name == "test"
        assert len(corpus) == 1
        assert corpus.signatures[0].source_id == sample_signature.source_id


class TestCorpusService:
    """Tests for CorpusService."""

    def test_create_corpus(self, corpus_service):
        """Test creating a corpus via service."""
        corpus = corpus_service.create_corpus("music_corpus", "music", "Music signatures")
        assert corpus.name == "music_corpus"
        assert corpus.domain == "music"
        assert corpus_service.get_corpus("music_corpus") is corpus

    def test_add_signature(self, corpus_service, sample_signature):
        """Test adding signature to corpus."""
        corpus = corpus_service.create_corpus("test", "text")
        result = corpus_service.add_signature(corpus, sample_signature)
        assert result is True
        assert len(corpus) == 1
        assert sample_signature.source_id in corpus._index

    def test_add_duplicate_signature(self, corpus_service, sample_signature):
        """Test that duplicate signatures are rejected."""
        corpus = corpus_service.create_corpus("test", "text")
        corpus_service.add_signature(corpus, sample_signature)
        result = corpus_service.add_signature(corpus, sample_signature)
        assert result is False
        assert len(corpus) == 1

    def test_save_and_load_corpus(self, corpus_service, sample_signature):
        """Test saving and loading corpus."""
        corpus = corpus_service.create_corpus("test", "text")
        corpus_service.add_signature(corpus, sample_signature)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            # Save
            result = corpus_service.save_corpus(corpus, temp_path)
            assert result is True
            assert os.path.exists(temp_path)

            # Load in new service
            new_service = CorpusService()
            loaded = new_service.load_corpus(temp_path)
            assert loaded is not None
            assert loaded.name == "test"
            assert len(loaded) == 1
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_corpus(self, corpus_service):
        """Test loading a corpus that doesn't exist."""
        result = corpus_service.load_corpus("/nonexistent/path/corpus.json")
        assert result is None

    def test_build_index(self, corpus_service, sample_signature, sample_signature_2):
        """Test rebuilding corpus index."""
        corpus = corpus_service.create_corpus("test", "text")
        corpus_service.add_signature(corpus, sample_signature)
        corpus_service.add_signature(corpus, sample_signature_2)

        # Clear and rebuild index
        corpus._index.clear()
        assert len(corpus._index) == 0

        corpus_service.build_index(corpus)
        assert len(corpus._index) == 2
        assert sample_signature.source_id in corpus._index
        assert sample_signature_2.source_id in corpus._index

    def test_query_corpus(self, corpus_service, sample_signature, sample_signature_2):
        """Test querying corpus for similar signatures."""
        corpus = corpus_service.create_corpus("test", "text")
        corpus_service.add_signature(corpus, sample_signature)
        corpus_service.add_signature(corpus, sample_signature_2)

        # Query with sample_signature_2
        results = corpus_service.query(corpus, sample_signature_2, top_k=5)
        assert len(results) == 1  # Only sample_signature (not self)
        assert results[0][0].source_id == sample_signature.source_id

    def test_query_with_threshold(self, corpus_service, sample_signature, sample_signature_2):
        """Test querying with high threshold."""
        corpus = corpus_service.create_corpus("test", "text")
        corpus_service.add_signature(corpus, sample_signature)
        corpus_service.add_signature(corpus, sample_signature_2)

        # Query with very high threshold
        results = corpus_service.query(corpus, sample_signature_2, threshold=0.99)
        # Should return no results (threshold too high for different signatures)
        assert len(results) == 0

    def test_list_corpora(self, corpus_service):
        """Test listing all loaded corpora."""
        corpus_service.create_corpus("corpus_1", "text")
        corpus_service.create_corpus("corpus_2", "music")

        names = corpus_service.list_corpora()
        assert "corpus_1" in names
        assert "corpus_2" in names


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_create_corpus_function(self):
        """Test create_corpus convenience function."""
        corpus = create_corpus("test", "image", "Image corpus")
        assert corpus.name == "test"
        assert corpus.domain == "image"

    def test_save_and_load_functions(self, sample_signature):
        """Test save_corpus and load_corpus convenience functions."""
        corpus = create_corpus("test", "text")
        corpus.signatures.append(sample_signature)
        corpus._index[sample_signature.source_id] = 0

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            assert save_corpus(corpus, temp_path) is True
            loaded = load_corpus(temp_path)
            assert loaded is not None
            assert loaded.name == "test"
        finally:
            os.unlink(temp_path)
