"""
Service Container - Protocol-based Dependency Injection

Provides a central factory that wires all services together using
Protocol-based dependency injection for maximum flexibility and testability.

Key benefits:
- Services can be swapped without changing dependent code
- Mock implementations for testing
- Clear dependency graph
- Topologically sorted instantiation
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any

from .protocols import (
    CanExtractSignatures,
    CanExtractMotifs,
    CanComputeSimilarity,
    CanManageCorpus,
    CanQueryCorpus,
    CanFindResonances,
    CanExplainResonance,
    CanAdaptToGraph,
    FullSimilarityService,
    FullCorpusService,
    FullInterpretationService,
)


@dataclass
class ServiceContainer:
    """
    Central service registry using protocol-based dependency injection.

    This container holds all service instances and handles wiring them together.
    Services are instantiated in dependency order (topological sort).

    Usage:
        container = create_service_container()
        signature = container.signature_extractor.extract(graph, "text", "my_text")
        resonances = container.resonance_service.find_resonances(signature, corpus)
    """

    # Foundation services (zero dependencies)
    motif_detector_factory: Any  # Factory function to create MotifDetector instances
    interpretation_service: FullInterpretationService

    # Domain adapters (zero dependencies)
    adapters: Dict[str, CanAdaptToGraph]

    # Tier 2: Depends on foundation
    similarity_service: FullSimilarityService

    # Tier 3: Depends on similarity
    corpus_service: FullCorpusService

    # Tier 4: Depends on corpus
    resonance_service: CanFindResonances

    # Signature extractor (depends on motif detector)
    signature_extractor: CanExtractSignatures


def create_service_container(
    custom_adapters: Optional[Dict[str, CanAdaptToGraph]] = None,
    custom_similarity: Optional[FullSimilarityService] = None,
    custom_corpus: Optional[FullCorpusService] = None,
    custom_resonance: Optional[CanFindResonances] = None,
    custom_interpretation: Optional[FullInterpretationService] = None,
) -> ServiceContainer:
    """
    Factory function that instantiates all services in dependency order.

    Any service can be overridden with a custom implementation that
    satisfies the same Protocol interface.

    Args:
        custom_adapters: Override domain adapters
        custom_similarity: Override similarity service
        custom_corpus: Override corpus service
        custom_resonance: Override resonance service
        custom_interpretation: Override interpretation service

    Returns:
        ServiceContainer with all services wired and ready

    Example:
        # Default container
        container = create_service_container()

        # With custom similarity algorithm
        my_sim = MySimilarityService()  # Implements FullSimilarityService protocol
        container = create_service_container(custom_similarity=my_sim)
    """
    from .motifs import MotifDetector
    from .extractor import SignatureExtractor
    from .similarity import SimilarityService
    from .corpus import CorpusService
    from .resonance import ResonanceService
    from .interpretation import InterpretationService
    from .adapters import (
        TextAdapter,
        ImageAdapter,
        MusicAdapter,
        CodeAdapter,
    )

    # ============================================
    # Step 1: Create foundation services (zero dependencies)
    # ============================================

    # MotifDetector requires a graph at construction, so we provide a factory
    def motif_detector_factory(graph):
        return MotifDetector(graph)

    # Interpretation service (zero dependencies - uses lookup tables)
    interpretation_service = custom_interpretation or InterpretationService()

    # ============================================
    # Step 2: Create domain adapters (zero dependencies)
    # ============================================

    if custom_adapters is not None:
        adapters = custom_adapters
    else:
        adapters = {
            "text": TextAdapter(),
            "image": ImageAdapter(),
            "music": MusicAdapter(),
            "code": CodeAdapter(),
        }

    # ============================================
    # Step 3: Create similarity service (zero external deps)
    # ============================================

    similarity_service = custom_similarity or SimilarityService()

    # ============================================
    # Step 4: Create corpus service (depends on similarity)
    # ============================================

    if custom_corpus is not None:
        corpus_service = custom_corpus
    else:
        corpus_service = CorpusService(similarity_service=similarity_service)

    # ============================================
    # Step 5: Create resonance service (depends on similarity, corpus)
    # ============================================

    if custom_resonance is not None:
        resonance_service = custom_resonance
    else:
        resonance_service = ResonanceService(
            similarity_service=similarity_service,
            corpus_service=corpus_service
        )

    # ============================================
    # Step 6: Create signature extractor
    # ============================================

    signature_extractor = SignatureExtractor()

    # ============================================
    # Step 7: Return assembled container
    # ============================================

    return ServiceContainer(
        motif_detector_factory=motif_detector_factory,
        interpretation_service=interpretation_service,
        adapters=adapters,
        similarity_service=similarity_service,
        corpus_service=corpus_service,
        resonance_service=resonance_service,
        signature_extractor=signature_extractor,
    )


def create_test_container(
    mock_similarity: Optional[FullSimilarityService] = None,
    mock_corpus: Optional[FullCorpusService] = None,
    mock_resonance: Optional[CanFindResonances] = None,
) -> ServiceContainer:
    """
    Create a container with mock services for testing.

    This allows unit tests to inject mock implementations
    that satisfy the Protocol interfaces without needing
    real implementations.

    Example:
        class MockSimilarity:
            def compute_similarity(self, sig1, sig2):
                return SimilarityResult(overall_score=0.75, ...)

        container = create_test_container(mock_similarity=MockSimilarity())
    """
    return create_service_container(
        custom_similarity=mock_similarity,
        custom_corpus=mock_corpus,
        custom_resonance=mock_resonance,
    )


# ============================================
# Convenience: Global container instance
# ============================================

_default_container: Optional[ServiceContainer] = None


def get_container() -> ServiceContainer:
    """
    Get the default service container (lazy initialization).

    Returns:
        The default ServiceContainer instance
    """
    global _default_container
    if _default_container is None:
        _default_container = create_service_container()
    return _default_container


def reset_container() -> None:
    """
    Reset the default container (useful for testing).
    """
    global _default_container
    _default_container = None


def set_container(container: ServiceContainer) -> None:
    """
    Set a custom container as the default (useful for testing).
    """
    global _default_container
    _default_container = container
