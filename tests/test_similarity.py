"""
Unit tests for SimilarityService
"""

import pytest
import sys
sys.path.insert(0, 'src')

from structural_rorschach.similarity import SimilarityService, SimilarityResult
from structural_rorschach.signature import StructuralSignature


@pytest.fixture
def similarity_service():
    """Create a SimilarityService instance."""
    return SimilarityService()


@pytest.fixture
def sample_signature_1():
    """Create a sample signature for testing."""
    return StructuralSignature(
        source_domain="text",
        source_id="test_1",
        source_name="Test Document 1",
        num_nodes=10,
        num_edges=15,
        density=0.333,
        clustering_coefficient=0.5,
        avg_path_length=2.5,
        is_dag=True,
        num_communities=2,
        motif_vector={
            "hub_spoke": 0.3,
            "chain_3": 0.4,
            "triangle": 0.2,
            "fork": 0.1
        },
        spectral_signature=[0.5, 0.3, 0.2, 0.1, 0.05],
        degree_distribution=[1, 2, 3, 2, 1, 1]
    )


@pytest.fixture
def sample_signature_2():
    """Create another sample signature for comparison."""
    return StructuralSignature(
        source_domain="music",
        source_id="test_2",
        source_name="Test Music Piece",
        num_nodes=12,
        num_edges=18,
        density=0.273,
        clustering_coefficient=0.45,
        avg_path_length=2.8,
        is_dag=True,
        num_communities=3,
        motif_vector={
            "hub_spoke": 0.35,
            "chain_3": 0.35,
            "triangle": 0.25,
            "fork": 0.05
        },
        spectral_signature=[0.48, 0.32, 0.18, 0.12, 0.06],
        degree_distribution=[2, 2, 3, 3, 1, 1]
    )


@pytest.fixture
def identical_signature():
    """Create an identical signature to sample_signature_1."""
    return StructuralSignature(
        source_domain="text",
        source_id="test_1_copy",
        source_name="Test Document 1 Copy",
        num_nodes=10,
        num_edges=15,
        density=0.333,
        clustering_coefficient=0.5,
        avg_path_length=2.5,
        is_dag=True,
        num_communities=2,
        motif_vector={
            "hub_spoke": 0.3,
            "chain_3": 0.4,
            "triangle": 0.2,
            "fork": 0.1
        },
        spectral_signature=[0.5, 0.3, 0.2, 0.1, 0.05],
        degree_distribution=[1, 2, 3, 2, 1, 1]
    )


class TestSimilarityService:
    """Tests for SimilarityService."""

    def test_compute_similarity_returns_result(
        self, similarity_service, sample_signature_1, sample_signature_2
    ):
        """Test that compute_similarity returns a SimilarityResult."""
        result = similarity_service.compute_similarity(
            sample_signature_1, sample_signature_2
        )
        assert isinstance(result, SimilarityResult)

    def test_compute_similarity_score_range(
        self, similarity_service, sample_signature_1, sample_signature_2
    ):
        """Test that similarity scores are in [0, 1] range."""
        result = similarity_service.compute_similarity(
            sample_signature_1, sample_signature_2
        )
        assert 0.0 <= result.overall_score <= 1.0
        assert 0.0 <= result.motif_similarity <= 1.0
        assert 0.0 <= result.spectral_similarity <= 1.0
        assert 0.0 <= result.scale_similarity <= 1.0

    def test_identical_signatures_high_similarity(
        self, similarity_service, sample_signature_1, identical_signature
    ):
        """Test that identical signatures have very high similarity."""
        result = similarity_service.compute_similarity(
            sample_signature_1, identical_signature
        )
        assert result.overall_score > 0.95
        assert result.motif_similarity > 0.99
        assert result.spectral_similarity > 0.99

    def test_compute_motif_similarity(self, similarity_service):
        """Test motif vector similarity computation."""
        vec1 = {"hub_spoke": 0.5, "chain": 0.3, "triangle": 0.2}
        vec2 = {"hub_spoke": 0.5, "chain": 0.3, "triangle": 0.2}

        score, matching = similarity_service.compute_motif_similarity(vec1, vec2)
        assert score > 0.99  # Identical vectors
        assert len(matching) == 3

    def test_compute_motif_similarity_different_keys(self, similarity_service):
        """Test motif similarity with different keys."""
        vec1 = {"hub_spoke": 0.5, "chain": 0.5}
        vec2 = {"hub_spoke": 0.5, "fork": 0.5}

        score, matching = similarity_service.compute_motif_similarity(vec1, vec2)
        assert 0.0 < score < 1.0
        assert "hub_spoke" in matching

    def test_compute_spectral_similarity(self, similarity_service):
        """Test spectral signature similarity."""
        spec1 = [0.5, 0.3, 0.2]
        spec2 = [0.5, 0.3, 0.2]

        score = similarity_service.compute_spectral_similarity(spec1, spec2)
        assert score > 0.99

    def test_compute_spectral_similarity_different_lengths(self, similarity_service):
        """Test spectral similarity with different length spectra."""
        spec1 = [0.5, 0.3, 0.2, 0.1]
        spec2 = [0.5, 0.3]

        score = similarity_service.compute_spectral_similarity(spec1, spec2)
        assert 0.0 <= score <= 1.0

    def test_compute_scale_similarity(
        self, similarity_service, sample_signature_1, sample_signature_2
    ):
        """Test scale similarity computation."""
        score = similarity_service.compute_scale_similarity(
            sample_signature_1, sample_signature_2
        )
        assert 0.0 <= score <= 1.0

    def test_matching_motifs_returned(
        self, similarity_service, sample_signature_1, sample_signature_2
    ):
        """Test that matching motifs are returned in result."""
        result = similarity_service.compute_similarity(
            sample_signature_1, sample_signature_2
        )
        assert isinstance(result.matching_motifs, list)


class TestSimilarityResult:
    """Tests for SimilarityResult dataclass."""

    def test_similarity_result_creation(self):
        """Test creating a SimilarityResult."""
        result = SimilarityResult(
            overall_score=0.75,
            motif_similarity=0.8,
            spectral_similarity=0.7,
            scale_similarity=0.75,
            matching_motifs=["hub_spoke", "chain"],
            details={"test": "value"}
        )
        assert result.overall_score == 0.75
        assert len(result.matching_motifs) == 2
