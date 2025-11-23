"""
Interpretation Service - Generate human-readable explanations for resonances

Implements F-IE-01 through F-IE-03:
- F-IE-01: Explain Resonance
- F-IE-02: Explain Cross-Domain Motif
- F-IE-03: Generate Comparison Report
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

from .signature import StructuralSignature, Resonance


# Domain-specific interpretations for motif types
MOTIF_INTERPRETATIONS = {
    "image": {
        "hub_spoke": "focal point with radial features (like flower petals or wheel spokes)",
        "star_3": "central element with radiating connections",
        "chain_3": "edge, contour, or boundary line",
        "chain": "continuous edge or visual flow",
        "triangle": "stable triangular region or corner junction",
        "fork": "branching structure or diverging paths",
        "funnel": "converging lines or focus point",
        "bridge": "transition zone connecting different regions",
        "cycle": "enclosed area or symmetrical pattern"
    },
    "music": {
        "hub_spoke": "tonic note with harmonic extensions (like a chord)",
        "star_3": "central pitch with surrounding ornaments",
        "chain_3": "melodic phrase or sequential notes",
        "chain": "melodic line or temporal sequence",
        "triangle": "triad chord (three notes forming a stable harmony)",
        "fork": "voice split or arpeggiated chord",
        "funnel": "resolution or cadence (multiple voices converging)",
        "bridge": "modulation or key change",
        "cycle": "ostinato, refrain, or repeating pattern"
    },
    "text": {
        "hub_spoke": "central theme with supporting points",
        "star_3": "main idea with related concepts",
        "chain_3": "narrative sequence or logical flow",
        "chain": "sentence structure or argument progression",
        "triangle": "circular reference or interconnected concepts",
        "fork": "enumeration or branching ideas",
        "funnel": "conclusion or synthesis of ideas",
        "bridge": "transition or plot turn",
        "cycle": "repetition, refrain, or cyclical narrative"
    },
    "code": {
        "hub_spoke": "central module with many dependents",
        "star_3": "core function with helper functions",
        "chain_3": "sequential processing pipeline",
        "chain": "call chain or data transformation",
        "triangle": "circular dependency (often problematic)",
        "fork": "branching logic or polymorphic dispatch",
        "funnel": "aggregation or data collection",
        "bridge": "adapter or interface layer",
        "cycle": "recursive pattern or feedback loop"
    }
}


class InterpretationService:
    """
    Generate human-readable explanations for structural resonances.

    This service bridges the gap between mathematical similarity scores
    and meaningful human understanding of cross-domain patterns.
    """

    def __init__(self):
        # Note: MotifDetector requires a graph, but we only need the
        # interpretation lookup which is done via MOTIF_INTERPRETATIONS dict
        pass

    def explain_resonance(
        self,
        resonance: Resonance,
        detail_level: str = "medium"
    ) -> str:
        """
        Generate human-readable explanation for a resonance.

        F-IE-01: Explain Resonance

        Args:
            resonance: The resonance to explain
            detail_level: "brief", "medium", or "detailed"

        Returns:
            Human-readable explanation string
        """
        score_pct = resonance.overall_score * 100

        if detail_level == "brief":
            return self._brief_explanation(resonance, score_pct)
        elif detail_level == "detailed":
            return self._detailed_explanation(resonance, score_pct)
        else:
            return self._medium_explanation(resonance, score_pct)

    def _brief_explanation(self, resonance: Resonance, score_pct: float) -> str:
        """Generate brief one-line explanation."""
        if resonance.matching_motifs:
            motif_str = ", ".join(resonance.matching_motifs[:2])
            return (
                f"{resonance.query_name} ({resonance.query_domain}) resonates with "
                f"{resonance.match_name} ({resonance.match_domain}) at {score_pct:.0f}% "
                f"through shared {motif_str} patterns."
            )
        return (
            f"{resonance.query_name} ({resonance.query_domain}) resonates with "
            f"{resonance.match_name} ({resonance.match_domain}) at {score_pct:.0f}% similarity."
        )

    def _medium_explanation(self, resonance: Resonance, score_pct: float) -> str:
        """Generate medium-length explanation with motif details."""
        lines = [
            f"Structural Resonance: {score_pct:.1f}% similarity",
            f"",
            f"Query: {resonance.query_name} ({resonance.query_domain})",
            f"Match: {resonance.match_name} ({resonance.match_domain})",
            f""
        ]

        if resonance.matching_motifs:
            lines.append("Shared structural patterns:")
            for motif in resonance.matching_motifs[:5]:
                query_interp = self.get_motif_interpretation(
                    motif, resonance.query_domain
                )
                match_interp = self.get_motif_interpretation(
                    motif, resonance.match_domain
                )
                lines.append(f"  - {motif}: {query_interp} <-> {match_interp}")

        lines.extend([
            f"",
            f"Similarity breakdown:",
            f"  - Motif patterns: {resonance.motif_similarity*100:.0f}%",
            f"  - Spectral shape: {resonance.spectral_similarity*100:.0f}%",
            f"  - Scale metrics: {resonance.scale_similarity*100:.0f}%"
        ])

        return "\n".join(lines)

    def _detailed_explanation(self, resonance: Resonance, score_pct: float) -> str:
        """Generate detailed explanation with full context."""
        lines = [
            "=" * 60,
            "STRUCTURAL RESONANCE ANALYSIS",
            "=" * 60,
            "",
            f"Overall Similarity: {score_pct:.1f}%",
            "",
            "QUERY STRUCTURE",
            f"  Name: {resonance.query_name}",
            f"  Domain: {resonance.query_domain}",
            f"  ID: {resonance.query_id}",
            "",
            "MATCHING STRUCTURE",
            f"  Name: {resonance.match_name}",
            f"  Domain: {resonance.match_domain}",
            f"  ID: {resonance.match_id}",
            "",
            "SIMILARITY COMPONENTS",
            f"  Motif Similarity: {resonance.motif_similarity*100:.1f}%",
            f"    Weight: 40% of overall score",
            f"    Measures: Cosine similarity of structural pattern frequencies",
            "",
            f"  Spectral Similarity: {resonance.spectral_similarity*100:.1f}%",
            f"    Weight: 30% of overall score",
            f"    Measures: Graph eigenvalue distribution similarity",
            "",
            f"  Scale Similarity: {resonance.scale_similarity*100:.1f}%",
            f"    Weight: 30% of overall score",
            f"    Measures: Node count, edge count, density",
            ""
        ]

        if resonance.matching_motifs:
            lines.extend([
                "MATCHING STRUCTURAL PATTERNS",
                ""
            ])
            for motif in resonance.matching_motifs:
                lines.append(f"  Pattern: {motif}")
                lines.append(f"    In {resonance.query_domain}: {self.get_motif_interpretation(motif, resonance.query_domain)}")
                lines.append(f"    In {resonance.match_domain}: {self.get_motif_interpretation(motif, resonance.match_domain)}")
                lines.append("")

        lines.extend([
            "INTERPRETATION",
            f"  These two structures share the same 'shape' despite coming from",
            f"  different domains. The structural patterns that appear in the",
            f"  {resonance.query_domain} domain have analogous patterns in the",
            f"  {resonance.match_domain} domain, suggesting a deep structural similarity.",
            "",
            "=" * 60
        ])

        return "\n".join(lines)

    def explain_cross_domain_motif(
        self,
        motif_type: str,
        domain1: str,
        domain2: str
    ) -> str:
        """
        Explain the meaning of a motif across two domains.

        F-IE-02: Explain Cross-Domain Motif

        Args:
            motif_type: Type of motif (e.g., "hub_spoke", "chain")
            domain1: First domain
            domain2: Second domain

        Returns:
            Explanation of how the motif manifests in both domains
        """
        interp1 = self.get_motif_interpretation(motif_type, domain1)
        interp2 = self.get_motif_interpretation(motif_type, domain2)

        return (
            f"The '{motif_type}' pattern appears differently in each domain:\n\n"
            f"In {domain1}: {interp1}\n\n"
            f"In {domain2}: {interp2}\n\n"
            f"When we find this pattern in both domains, it suggests a structural "
            f"'resonance' - the same underlying shape expressed through different media."
        )

    def get_motif_interpretation(self, motif_type: str, domain: str) -> str:
        """
        Get human-readable interpretation of a motif type for a domain.

        Args:
            motif_type: Type of motif
            domain: Domain context

        Returns:
            Interpretation string
        """
        domain_interps = MOTIF_INTERPRETATIONS.get(domain, {})
        return domain_interps.get(
            motif_type,
            f"{motif_type} pattern (domain-specific interpretation not available)"
        )

    def generate_comparison_report(
        self,
        sig1: StructuralSignature,
        sig2: StructuralSignature
    ) -> str:
        """
        Generate a full structural comparison report.

        F-IE-03: Generate Comparison Report

        Args:
            sig1: First signature
            sig2: Second signature

        Returns:
            Comprehensive comparison report
        """
        lines = [
            "=" * 70,
            "STRUCTURAL COMPARISON REPORT",
            "=" * 70,
            "",
            "STRUCTURE A",
            f"  Name: {sig1.source_name}",
            f"  Domain: {sig1.source_domain}",
            f"  Nodes: {sig1.num_nodes}, Edges: {sig1.num_edges}",
            f"  Density: {sig1.density:.4f}",
            f"  Clustering Coefficient: {sig1.clustering_coefficient:.4f}",
            f"  Communities: {sig1.num_communities}",
            f"  Average Path Length: {sig1.avg_path_length:.2f}",
            f"  Is DAG: {sig1.is_dag}",
            "",
            "STRUCTURE B",
            f"  Name: {sig2.source_name}",
            f"  Domain: {sig2.source_domain}",
            f"  Nodes: {sig2.num_nodes}, Edges: {sig2.num_edges}",
            f"  Density: {sig2.density:.4f}",
            f"  Clustering Coefficient: {sig2.clustering_coefficient:.4f}",
            f"  Communities: {sig2.num_communities}",
            f"  Average Path Length: {sig2.avg_path_length:.2f}",
            f"  Is DAG: {sig2.is_dag}",
            "",
            "MOTIF COMPARISON",
        ]

        # Compare motif vectors
        all_motifs = set(sig1.motif_vector.keys()) | set(sig2.motif_vector.keys())
        for motif in sorted(all_motifs):
            v1 = sig1.motif_vector.get(motif, 0.0)
            v2 = sig2.motif_vector.get(motif, 0.0)
            diff = abs(v1 - v2)
            status = "SIMILAR" if diff < 0.1 else "DIFFERENT"
            lines.append(f"  {motif}: A={v1:.3f}, B={v2:.3f} [{status}]")

        lines.extend([
            "",
            "SPECTRAL COMPARISON",
            f"  Structure A eigenvalues: {sig1.spectral_signature[:5]}...",
            f"  Structure B eigenvalues: {sig2.spectral_signature[:5]}...",
            "",
            "=" * 70
        ])

        return "\n".join(lines)


# Convenience functions
def explain_resonance(resonance: Resonance, detail_level: str = "medium") -> str:
    """Generate explanation for a resonance."""
    service = InterpretationService()
    return service.explain_resonance(resonance, detail_level)


def generate_comparison_report(
    sig1: StructuralSignature,
    sig2: StructuralSignature
) -> str:
    """Generate comparison report between two signatures."""
    service = InterpretationService()
    return service.generate_comparison_report(sig1, sig2)
