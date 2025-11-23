"""
Unit tests for ServiceContainer and Protocol-based DI
"""

import pytest
import sys
sys.path.insert(0, 'src')

from structural_rorschach import (
    ServiceContainer,
    create_service_container,
    create_test_container,
    get_container,
    reset_container,
    set_container,
)
from structural_rorschach.protocols import (
    CanFindResonances,
    CanComputeSimilarity,
    CanAdaptToGraph,
    FullCorpusService,
    FullSimilarityService,
    FullInterpretationService,
    CanRankResonances,
)
from structural_rorschach.similarity import SimilarityService, SimilarityResult


class TestServiceContainer:
    """Tests for ServiceContainer creation and wiring."""

    def test_create_service_container(self):
        """Test that create_service_container returns a ServiceContainer."""
        container = create_service_container()
        assert isinstance(container, ServiceContainer)

    def test_container_has_all_services(self):
        """Test that container has all required services."""
        container = create_service_container()
        assert container.similarity_service is not None
        assert container.corpus_service is not None
        assert container.resonance_service is not None
        assert container.interpretation_service is not None
        assert container.adapters is not None
        assert container.signature_extractor is not None

    def test_adapters_registered(self):
        """Test that all domain adapters are registered."""
        container = create_service_container()
        assert "text" in container.adapters
        assert "image" in container.adapters
        assert "music" in container.adapters
        assert "code" in container.adapters


class TestProtocolCompliance:
    """Tests that services implement their Protocol interfaces."""

    def test_similarity_service_protocol(self):
        """Test SimilarityService implements FullSimilarityService."""
        container = create_service_container()
        assert isinstance(container.similarity_service, FullSimilarityService)

    def test_corpus_service_protocol(self):
        """Test CorpusService implements FullCorpusService."""
        container = create_service_container()
        assert isinstance(container.corpus_service, FullCorpusService)

    def test_resonance_service_protocols(self):
        """Test ResonanceService implements resonance protocols."""
        container = create_service_container()
        assert isinstance(container.resonance_service, CanFindResonances)
        assert isinstance(container.resonance_service, CanRankResonances)

    def test_interpretation_service_protocol(self):
        """Test InterpretationService implements FullInterpretationService."""
        container = create_service_container()
        assert isinstance(container.interpretation_service, FullInterpretationService)

    def test_adapters_protocol(self):
        """Test all adapters implement CanAdaptToGraph."""
        container = create_service_container()
        for name, adapter in container.adapters.items():
            assert isinstance(adapter, CanAdaptToGraph), f"{name} adapter should implement CanAdaptToGraph"


class TestDependencyInjection:
    """Tests for dependency injection behavior."""

    def test_custom_similarity_service(self):
        """Test injecting a custom similarity service."""
        class MockSimilarity:
            def compute_similarity(self, sig1, sig2):
                return SimilarityResult(
                    overall_score=0.99,
                    motif_similarity=1.0,
                    spectral_similarity=0.98,
                    scale_similarity=1.0,
                    matching_motifs=["test"],
                    details={"mock": True}
                )

            def compute_motif_similarity(self, vec1, vec2):
                return (1.0, ["all"])

            def compute_spectral_similarity(self, spec1, spec2):
                return 1.0

            def compute_scale_similarity(self, sig1, sig2):
                return 1.0

        container = create_service_container(custom_similarity=MockSimilarity())
        assert type(container.similarity_service).__name__ == "MockSimilarity"

    def test_custom_service_propagates(self):
        """Test that custom service propagates to dependent services."""
        class MockSimilarity:
            def compute_similarity(self, sig1, sig2):
                return SimilarityResult(
                    overall_score=0.5,
                    motif_similarity=0.5,
                    spectral_similarity=0.5,
                    scale_similarity=0.5,
                    matching_motifs=[],
                    details={}
                )

            def compute_motif_similarity(self, vec1, vec2):
                return (0.5, [])

            def compute_spectral_similarity(self, spec1, spec2):
                return 0.5

            def compute_scale_similarity(self, sig1, sig2):
                return 0.5

        container = create_service_container(custom_similarity=MockSimilarity())

        # CorpusService should use the injected similarity
        assert type(container.corpus_service.similarity_service).__name__ == "MockSimilarity"

        # ResonanceService should use the injected similarity
        assert type(container.resonance_service.similarity_service).__name__ == "MockSimilarity"

    def test_custom_adapters(self):
        """Test injecting custom adapters."""
        class MockAdapter:
            @property
            def domain(self):
                return "mock"

            def adapt(self, data, **kwargs):
                return {"nodes": [], "edges": []}

        custom_adapters = {"mock": MockAdapter()}
        container = create_service_container(custom_adapters=custom_adapters)

        assert "mock" in container.adapters
        assert container.adapters["mock"].domain == "mock"


class TestGlobalContainer:
    """Tests for global container management."""

    def setup_method(self):
        """Reset global container before each test."""
        reset_container()

    def test_get_container_creates_default(self):
        """Test get_container creates default container on first call."""
        container = get_container()
        assert isinstance(container, ServiceContainer)

    def test_get_container_returns_same_instance(self):
        """Test get_container returns same instance on subsequent calls."""
        container1 = get_container()
        container2 = get_container()
        assert container1 is container2

    def test_reset_container(self):
        """Test reset_container clears the global container."""
        container1 = get_container()
        reset_container()
        container2 = get_container()
        assert container1 is not container2

    def test_set_container(self):
        """Test set_container sets a custom container."""
        custom = create_service_container()
        set_container(custom)
        assert get_container() is custom


class TestCreateTestContainer:
    """Tests for test container creation."""

    def test_create_test_container_basic(self):
        """Test creating a test container."""
        container = create_test_container()
        assert isinstance(container, ServiceContainer)

    def test_create_test_container_with_mocks(self):
        """Test creating test container with mock services."""
        class MockSimilarity:
            def compute_similarity(self, sig1, sig2):
                return SimilarityResult(
                    overall_score=1.0,
                    motif_similarity=1.0,
                    spectral_similarity=1.0,
                    scale_similarity=1.0,
                    matching_motifs=[],
                    details={}
                )

            def compute_motif_similarity(self, vec1, vec2):
                return (1.0, [])

            def compute_spectral_similarity(self, spec1, spec2):
                return 1.0

            def compute_scale_similarity(self, sig1, sig2):
                return 1.0

        container = create_test_container(mock_similarity=MockSimilarity())
        assert type(container.similarity_service).__name__ == "MockSimilarity"
