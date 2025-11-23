"""
Unit tests for InterpretationService
"""

import pytest
import sys
sys.path.insert(0, 'src')

from structural_rorschach.interpretation import (
    InterpretationService,
    MOTIF_INTERPRETATIONS,
    explain_resonance,
    generate_comparison_report,
)
from structural_rorschach.signature import StructuralSignature, Resonance


@pytest.fixture
def interpretation_service():
    """Create an InterpretationService instance."""
    return InterpretationService()


@pytest.fixture
def sample_resonance():
    """Create a sample Resonance for testing."""
    return Resonance(
        query_domain="image",
        query_id="img_1",
        query_name="Test Image",
        match_domain="music",
        match_id="music_1",
        match_name="Test Music",
        overall_score=0.75,
        motif_similarity=0.8,
        spectral_similarity=0.7,
        scale_similarity=0.75,
        matching_motifs=["hub_spoke", "chain_3", "triangle"],
        shared_properties={"test": "value"},
        explanation="Test explanation"
    )


@pytest.fixture
def sample_signature_1():
    """Create a sample signature."""
    return StructuralSignature(
        source_domain="text",
        source_id="text_1",
        source_name="Test Document",
        num_nodes=10,
        num_edges=15,
        density=0.333,
        clustering_coefficient=0.5,
        avg_path_length=2.5,
        is_dag=True,
        num_communities=2,
        motif_vector={"hub_spoke": 0.3, "chain_3": 0.4, "triangle": 0.2},
        spectral_signature=[0.5, 0.3, 0.2, 0.1],
        degree_distribution=[1, 2, 3, 2, 1, 1]
    )


@pytest.fixture
def sample_signature_2():
    """Create another sample signature."""
    return StructuralSignature(
        source_domain="music",
        source_id="music_1",
        source_name="Test Music Piece",
        num_nodes=12,
        num_edges=18,
        density=0.273,
        clustering_coefficient=0.45,
        avg_path_length=2.8,
        is_dag=False,
        num_communities=3,
        motif_vector={"hub_spoke": 0.35, "chain_3": 0.35, "fork": 0.3},
        spectral_signature=[0.48, 0.32, 0.18, 0.12],
        degree_distribution=[2, 2, 3, 3, 1, 1]
    )


class TestInterpretationService:
    """Tests for InterpretationService."""

    def test_explain_resonance_brief(self, interpretation_service, sample_resonance):
        """Test brief resonance explanation."""
        explanation = interpretation_service.explain_resonance(
            sample_resonance, detail_level="brief"
        )
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert "75%" in explanation or "75" in explanation
        assert sample_resonance.query_name in explanation

    def test_explain_resonance_medium(self, interpretation_service, sample_resonance):
        """Test medium resonance explanation."""
        explanation = interpretation_service.explain_resonance(
            sample_resonance, detail_level="medium"
        )
        assert isinstance(explanation, str)
        assert "Structural Resonance" in explanation
        assert "Shared structural patterns" in explanation

    def test_explain_resonance_detailed(self, interpretation_service, sample_resonance):
        """Test detailed resonance explanation."""
        explanation = interpretation_service.explain_resonance(
            sample_resonance, detail_level="detailed"
        )
        assert isinstance(explanation, str)
        assert "STRUCTURAL RESONANCE ANALYSIS" in explanation
        assert "QUERY STRUCTURE" in explanation
        assert "MATCHING STRUCTURE" in explanation
        assert "SIMILARITY COMPONENTS" in explanation

    def test_explain_cross_domain_motif(self, interpretation_service):
        """Test cross-domain motif explanation."""
        explanation = interpretation_service.explain_cross_domain_motif(
            "hub_spoke", "image", "music"
        )
        assert isinstance(explanation, str)
        assert "hub_spoke" in explanation
        assert "image" in explanation
        assert "music" in explanation

    def test_get_motif_interpretation_known(self, interpretation_service):
        """Test getting interpretation for known motif/domain."""
        interp = interpretation_service.get_motif_interpretation("hub_spoke", "image")
        assert isinstance(interp, str)
        assert "focal point" in interp.lower() or "radial" in interp.lower()

    def test_get_motif_interpretation_unknown_domain(self, interpretation_service):
        """Test getting interpretation for unknown domain."""
        interp = interpretation_service.get_motif_interpretation("hub_spoke", "unknown_domain")
        assert "hub_spoke" in interp
        assert "not available" in interp

    def test_get_motif_interpretation_unknown_motif(self, interpretation_service):
        """Test getting interpretation for unknown motif."""
        interp = interpretation_service.get_motif_interpretation("unknown_motif", "image")
        assert "unknown_motif" in interp

    def test_generate_comparison_report(
        self, interpretation_service, sample_signature_1, sample_signature_2
    ):
        """Test generating comparison report."""
        report = interpretation_service.generate_comparison_report(
            sample_signature_1, sample_signature_2
        )
        assert isinstance(report, str)
        assert "STRUCTURAL COMPARISON REPORT" in report
        assert "STRUCTURE A" in report
        assert "STRUCTURE B" in report
        assert "MOTIF COMPARISON" in report
        assert "SPECTRAL COMPARISON" in report

    def test_generate_comparison_report_contains_names(
        self, interpretation_service, sample_signature_1, sample_signature_2
    ):
        """Test that comparison report contains structure names."""
        report = interpretation_service.generate_comparison_report(
            sample_signature_1, sample_signature_2
        )
        assert sample_signature_1.source_name in report
        assert sample_signature_2.source_name in report


class TestMotifInterpretations:
    """Tests for MOTIF_INTERPRETATIONS dictionary."""

    def test_all_domains_present(self):
        """Test that all expected domains are in interpretations."""
        expected_domains = ["image", "music", "text", "code"]
        for domain in expected_domains:
            assert domain in MOTIF_INTERPRETATIONS

    def test_common_motifs_in_all_domains(self):
        """Test that common motifs exist across all domains."""
        common_motifs = ["hub_spoke", "chain_3", "triangle"]
        for domain in MOTIF_INTERPRETATIONS:
            for motif in common_motifs:
                assert motif in MOTIF_INTERPRETATIONS[domain], \
                    f"Missing {motif} in {domain}"


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_explain_resonance_function(self, sample_resonance):
        """Test explain_resonance convenience function."""
        explanation = explain_resonance(sample_resonance)
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_generate_comparison_report_function(
        self, sample_signature_1, sample_signature_2
    ):
        """Test generate_comparison_report convenience function."""
        report = generate_comparison_report(sample_signature_1, sample_signature_2)
        assert isinstance(report, str)
        assert "COMPARISON REPORT" in report
