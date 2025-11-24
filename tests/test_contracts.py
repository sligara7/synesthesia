"""
Contract Tests for Synesthesia - Protocol Compliance Verification

Verifies that all services implement their declared protocols correctly.
This ensures the Protocol-based DI system maintains type safety.
"""

import pytest
import sys
sys.path.insert(0, 'src')

from structural_rorschach import (
    create_service_container,
    ServiceContainer,
)
from structural_rorschach.protocols import (
    # Data protocols
    SignatureData,
    ResonanceData,
    SimilarityData,
    # Adapter protocols
    CanAdaptToGraph,
    ProvidesGraphValidation,
    # Signature protocols
    CanExtractSignatures,
    CanExtractMotifs,
    # Similarity protocols
    CanComputeSimilarity,
    CanComputeMotifSimilarity,
    CanComputeSpectralSimilarity,
    FullSimilarityService,
    # Corpus protocols
    CanManageCorpus,
    CanQueryCorpus,
    FullCorpusService,
    # Resonance protocols
    CanFindResonances,
    CanRankResonances,
    # Interpretation protocols
    CanExplainResonance,
    CanExplainMotifs,
    CanGenerateReports,
    FullInterpretationService,
)


@pytest.fixture
def container():
    """Create a fresh service container."""
    return create_service_container()


class TestSimilarityServiceContracts:
    """Verify SimilarityService implements all required protocols."""

    def test_implements_full_similarity_service(self, container):
        """SimilarityService must implement FullSimilarityService protocol."""
        assert isinstance(container.similarity_service, FullSimilarityService)

    def test_implements_can_compute_motif_similarity(self, container):
        """SimilarityService must implement CanComputeMotifSimilarity."""
        assert isinstance(container.similarity_service, CanComputeMotifSimilarity)

    def test_implements_can_compute_spectral_similarity(self, container):
        """SimilarityService must implement CanComputeSpectralSimilarity."""
        assert isinstance(container.similarity_service, CanComputeSpectralSimilarity)

    def test_has_compute_similarity_method(self, container):
        """SimilarityService must have compute_similarity method."""
        assert hasattr(container.similarity_service, 'compute_similarity')
        assert callable(container.similarity_service.compute_similarity)

    def test_has_compute_motif_similarity_method(self, container):
        """SimilarityService must have compute_motif_similarity method."""
        assert hasattr(container.similarity_service, 'compute_motif_similarity')
        assert callable(container.similarity_service.compute_motif_similarity)

    def test_has_compute_spectral_similarity_method(self, container):
        """SimilarityService must have compute_spectral_similarity method."""
        assert hasattr(container.similarity_service, 'compute_spectral_similarity')
        assert callable(container.similarity_service.compute_spectral_similarity)


class TestCorpusServiceContracts:
    """Verify CorpusService implements all required protocols."""

    def test_implements_full_corpus_service(self, container):
        """CorpusService must implement FullCorpusService protocol."""
        assert isinstance(container.corpus_service, FullCorpusService)

    def test_implements_can_query_corpus(self, container):
        """CorpusService must implement CanQueryCorpus."""
        assert isinstance(container.corpus_service, CanQueryCorpus)

    def test_has_create_corpus_method(self, container):
        """CorpusService must have create_corpus method."""
        assert hasattr(container.corpus_service, 'create_corpus')
        assert callable(container.corpus_service.create_corpus)

    def test_has_add_signature_method(self, container):
        """CorpusService must have add_signature method."""
        assert hasattr(container.corpus_service, 'add_signature')
        assert callable(container.corpus_service.add_signature)

    def test_has_query_method(self, container):
        """CorpusService must have query method."""
        assert hasattr(container.corpus_service, 'query')
        assert callable(container.corpus_service.query)

    def test_has_save_corpus_method(self, container):
        """CorpusService must have save_corpus method."""
        assert hasattr(container.corpus_service, 'save_corpus')
        assert callable(container.corpus_service.save_corpus)

    def test_has_load_corpus_method(self, container):
        """CorpusService must have load_corpus method."""
        assert hasattr(container.corpus_service, 'load_corpus')
        assert callable(container.corpus_service.load_corpus)


class TestResonanceServiceContracts:
    """Verify ResonanceService implements all required protocols."""

    def test_implements_can_find_resonances(self, container):
        """ResonanceService must implement CanFindResonances."""
        assert isinstance(container.resonance_service, CanFindResonances)

    def test_implements_can_rank_resonances(self, container):
        """ResonanceService must implement CanRankResonances."""
        assert isinstance(container.resonance_service, CanRankResonances)

    def test_has_find_resonances_method(self, container):
        """ResonanceService must have find_resonances method."""
        assert hasattr(container.resonance_service, 'find_resonances')
        assert callable(container.resonance_service.find_resonances)

    def test_has_rank_resonances_method(self, container):
        """ResonanceService must have rank_resonances method."""
        assert hasattr(container.resonance_service, 'rank_resonances')
        assert callable(container.resonance_service.rank_resonances)


class TestInterpretationServiceContracts:
    """Verify InterpretationService implements all required protocols."""

    def test_implements_full_interpretation_service(self, container):
        """InterpretationService must implement FullInterpretationService."""
        assert isinstance(container.interpretation_service, FullInterpretationService)

    def test_implements_can_explain_resonance(self, container):
        """InterpretationService must implement CanExplainResonance."""
        assert isinstance(container.interpretation_service, CanExplainResonance)

    def test_implements_can_explain_motifs(self, container):
        """InterpretationService must implement CanExplainMotifs."""
        assert isinstance(container.interpretation_service, CanExplainMotifs)

    def test_implements_can_generate_reports(self, container):
        """InterpretationService must implement CanGenerateReports."""
        assert isinstance(container.interpretation_service, CanGenerateReports)

    def test_has_explain_resonance_method(self, container):
        """InterpretationService must have explain_resonance method."""
        assert hasattr(container.interpretation_service, 'explain_resonance')
        assert callable(container.interpretation_service.explain_resonance)

    def test_has_generate_comparison_report_method(self, container):
        """InterpretationService must have generate_comparison_report method."""
        assert hasattr(container.interpretation_service, 'generate_comparison_report')
        assert callable(container.interpretation_service.generate_comparison_report)


class TestSignatureExtractorContracts:
    """Verify SignatureExtractor implements required methods."""

    def test_has_extract_from_networkx_method(self, container):
        """SignatureExtractor must have extract_from_networkx method."""
        assert hasattr(container.signature_extractor, 'extract_from_networkx')
        assert callable(container.signature_extractor.extract_from_networkx)


class TestAdapterContracts:
    """Verify all adapters implement required protocols."""

    def test_text_adapter_implements_can_adapt(self, container):
        """TextAdapter must implement CanAdaptToGraph."""
        assert "text" in container.adapters
        assert isinstance(container.adapters["text"], CanAdaptToGraph)

    def test_image_adapter_implements_can_adapt(self, container):
        """ImageAdapter must implement CanAdaptToGraph."""
        assert "image" in container.adapters
        assert isinstance(container.adapters["image"], CanAdaptToGraph)

    def test_music_adapter_implements_can_adapt(self, container):
        """MusicAdapter must implement CanAdaptToGraph."""
        assert "music" in container.adapters
        assert isinstance(container.adapters["music"], CanAdaptToGraph)

    def test_code_adapter_implements_can_adapt(self, container):
        """CodeAdapter must implement CanAdaptToGraph."""
        assert "code" in container.adapters
        assert isinstance(container.adapters["code"], CanAdaptToGraph)

    def test_all_adapters_have_adapt_method(self, container):
        """All adapters must have adapt method."""
        for domain, adapter in container.adapters.items():
            assert hasattr(adapter, 'adapt'), f"{domain} adapter missing adapt()"
            assert callable(adapter.adapt), f"{domain} adapter adapt not callable"


class TestServiceContainerContracts:
    """Verify ServiceContainer provides all expected services."""

    def test_container_has_similarity_service(self, container):
        """Container must provide similarity_service."""
        assert hasattr(container, 'similarity_service')
        assert container.similarity_service is not None

    def test_container_has_corpus_service(self, container):
        """Container must provide corpus_service."""
        assert hasattr(container, 'corpus_service')
        assert container.corpus_service is not None

    def test_container_has_resonance_service(self, container):
        """Container must provide resonance_service."""
        assert hasattr(container, 'resonance_service')
        assert container.resonance_service is not None

    def test_container_has_interpretation_service(self, container):
        """Container must provide interpretation_service."""
        assert hasattr(container, 'interpretation_service')
        assert container.interpretation_service is not None

    def test_container_has_signature_extractor(self, container):
        """Container must provide signature_extractor."""
        assert hasattr(container, 'signature_extractor')
        assert container.signature_extractor is not None

    def test_container_has_adapters(self, container):
        """Container must provide adapters dict."""
        assert hasattr(container, 'adapters')
        assert isinstance(container.adapters, dict)
        assert len(container.adapters) >= 4


class TestDataTypeContracts:
    """Verify data types have required attributes."""

    def test_structural_signature_has_required_fields(self):
        """StructuralSignature must have all required fields."""
        from structural_rorschach import StructuralSignature
        sig = StructuralSignature(
            source_domain="test",
            source_id="test_id",
            source_name="Test"
        )
        # Verify required fields exist
        assert hasattr(sig, 'source_domain')
        assert hasattr(sig, 'source_id')
        assert hasattr(sig, 'source_name')
        assert hasattr(sig, 'num_nodes')
        assert hasattr(sig, 'num_edges')
        assert hasattr(sig, 'motif_vector')
        assert hasattr(sig, 'spectral_signature')
        # Verify serialization
        assert hasattr(sig, 'to_dict')
        assert hasattr(sig, 'from_dict')

    def test_resonance_has_required_fields(self):
        """Resonance must have all required fields."""
        from structural_rorschach import Resonance
        res = Resonance(
            query_domain="test",
            query_id="q1",
            query_name="Query",
            match_domain="test",
            match_id="m1",
            match_name="Match",
            overall_score=0.5,
            motif_similarity=0.5,
            spectral_similarity=0.5,
            scale_similarity=0.5,
            matching_motifs=[],
            shared_properties={},
            explanation="Test"
        )
        # Verify required fields exist
        assert hasattr(res, 'query_id')
        assert hasattr(res, 'match_id')
        assert hasattr(res, 'overall_score')
        assert hasattr(res, 'matching_motifs')
        assert hasattr(res, 'explanation')
        # Verify serialization
        assert hasattr(res, 'to_dict')
