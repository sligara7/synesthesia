"""
Similarity Computation Service - Compute structural similarity between signatures

Implements F-SC-01 through F-SC-04:
- F-SC-01: Compute Overall Similarity (weighted combination)
- F-SC-02: Compute Motif Similarity (cosine similarity)
- F-SC-03: Compute Spectral Similarity (eigenvalue comparison)
- F-SC-04: Compute Scale Similarity (node/edge/density)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import math

from .signature import StructuralSignature


@dataclass
class SimilarityResult:
    """Result of similarity computation between two signatures."""
    overall_score: float
    motif_similarity: float
    spectral_similarity: float
    scale_similarity: float
    matching_motifs: List[str]
    details: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "overall_score": self.overall_score,
            "motif_similarity": self.motif_similarity,
            "spectral_similarity": self.spectral_similarity,
            "scale_similarity": self.scale_similarity,
            "matching_motifs": self.matching_motifs,
            "details": self.details
        }


class SimilarityService:
    """
    Compute structural similarity between signatures.

    Default weights (can be customized):
    - Motif similarity: 40%
    - Spectral similarity: 30%
    - Scale similarity: 30%
    """

    def __init__(
        self,
        motif_weight: float = 0.4,
        spectral_weight: float = 0.3,
        scale_weight: float = 0.3
    ):
        self.motif_weight = motif_weight
        self.spectral_weight = spectral_weight
        self.scale_weight = scale_weight

    def compute_similarity(
        self,
        sig1: StructuralSignature,
        sig2: StructuralSignature
    ) -> SimilarityResult:
        """
        Compute overall structural similarity between two signatures.

        F-SC-01: Combines motif, spectral, and scale similarity with configurable weights.

        Args:
            sig1: First structural signature
            sig2: Second structural signature

        Returns:
            SimilarityResult with overall and component scores
        """
        # Compute component similarities
        motif_sim, matching_motifs = self.compute_motif_similarity(
            sig1.motif_vector, sig2.motif_vector
        )
        spectral_sim = self.compute_spectral_similarity(
            sig1.spectral_signature, sig2.spectral_signature
        )
        scale_sim = self.compute_scale_similarity(sig1, sig2)

        # Weighted combination
        overall = (
            self.motif_weight * motif_sim +
            self.spectral_weight * spectral_sim +
            self.scale_weight * scale_sim
        )

        return SimilarityResult(
            overall_score=overall,
            motif_similarity=motif_sim,
            spectral_similarity=spectral_sim,
            scale_similarity=scale_sim,
            matching_motifs=matching_motifs,
            details={
                "motif_weight": self.motif_weight,
                "spectral_weight": self.spectral_weight,
                "scale_weight": self.scale_weight,
                "sig1_domain": sig1.source_domain,
                "sig2_domain": sig2.source_domain
            }
        )

    def compute_motif_similarity(
        self,
        vec1: Dict[str, float],
        vec2: Dict[str, float],
        threshold: float = 0.1
    ) -> tuple[float, List[str]]:
        """
        Compute cosine similarity between motif vectors.

        F-SC-02: Motif similarity using cosine distance.

        Args:
            vec1: First motif vector {motif_type: frequency}
            vec2: Second motif vector
            threshold: Minimum frequency to consider a motif "matching"

        Returns:
            Tuple of (similarity score, list of matching motif types)
        """
        if not vec1 or not vec2:
            return 0.0, []

        # Get all motif types
        all_motifs = set(vec1.keys()) | set(vec2.keys())

        # Compute dot product and magnitudes
        dot_product = 0.0
        mag1 = 0.0
        mag2 = 0.0
        matching_motifs = []

        for motif in all_motifs:
            v1 = vec1.get(motif, 0.0)
            v2 = vec2.get(motif, 0.0)

            dot_product += v1 * v2
            mag1 += v1 * v1
            mag2 += v2 * v2

            # Track matching motifs (both above threshold)
            if v1 >= threshold and v2 >= threshold:
                matching_motifs.append(motif)

        mag1 = math.sqrt(mag1)
        mag2 = math.sqrt(mag2)

        if mag1 == 0 or mag2 == 0:
            return 0.0, []

        similarity = dot_product / (mag1 * mag2)
        return similarity, matching_motifs

    def compute_spectral_similarity(
        self,
        spec1: List[float],
        spec2: List[float]
    ) -> float:
        """
        Compute similarity between spectral signatures.

        F-SC-03: Compare eigenvalue distributions.

        Args:
            spec1: First spectral signature (eigenvalues)
            spec2: Second spectral signature

        Returns:
            Similarity score in [0, 1]
        """
        if not spec1 or not spec2:
            return 0.0

        # Pad shorter list with zeros
        max_len = max(len(spec1), len(spec2))
        s1 = list(spec1) + [0.0] * (max_len - len(spec1))
        s2 = list(spec2) + [0.0] * (max_len - len(spec2))

        # Normalize to sum to 1 (treat as distribution)
        sum1 = sum(abs(x) for x in s1) or 1.0
        sum2 = sum(abs(x) for x in s2) or 1.0

        s1 = [x / sum1 for x in s1]
        s2 = [x / sum2 for x in s2]

        # Compute similarity as 1 - normalized L1 distance
        l1_distance = sum(abs(a - b) for a, b in zip(s1, s2))

        # L1 distance is in [0, 2] for normalized distributions
        similarity = 1.0 - (l1_distance / 2.0)
        return max(0.0, min(1.0, similarity))

    def compute_scale_similarity(
        self,
        sig1: StructuralSignature,
        sig2: StructuralSignature
    ) -> float:
        """
        Compute similarity based on graph scale metrics.

        F-SC-04: Compare node count, edge count, and density.

        Args:
            sig1: First signature
            sig2: Second signature

        Returns:
            Similarity score in [0, 1]
        """
        # Node count similarity (log scale to handle large differences)
        n1, n2 = sig1.num_nodes, sig2.num_nodes
        if n1 > 0 and n2 > 0:
            node_sim = 1.0 - abs(math.log(n1) - math.log(n2)) / max(math.log(n1), math.log(n2), 1)
            node_sim = max(0.0, node_sim)
        else:
            node_sim = 1.0 if n1 == n2 else 0.0

        # Edge count similarity (log scale)
        e1, e2 = sig1.num_edges, sig2.num_edges
        if e1 > 0 and e2 > 0:
            edge_sim = 1.0 - abs(math.log(e1) - math.log(e2)) / max(math.log(e1), math.log(e2), 1)
            edge_sim = max(0.0, edge_sim)
        else:
            edge_sim = 1.0 if e1 == e2 else 0.0

        # Density similarity (linear)
        d1, d2 = sig1.density, sig2.density
        density_sim = 1.0 - abs(d1 - d2)

        # Combine with equal weights
        return (node_sim + edge_sim + density_sim) / 3.0


# Convenience function for simple similarity computation
def compute_similarity(
    sig1: StructuralSignature,
    sig2: StructuralSignature,
    weights: Optional[Dict[str, float]] = None
) -> SimilarityResult:
    """
    Compute structural similarity between two signatures.

    Args:
        sig1: First signature
        sig2: Second signature
        weights: Optional dict with 'motif', 'spectral', 'scale' weights

    Returns:
        SimilarityResult with all scores
    """
    if weights:
        service = SimilarityService(
            motif_weight=weights.get('motif', 0.4),
            spectral_weight=weights.get('spectral', 0.3),
            scale_weight=weights.get('scale', 0.3)
        )
    else:
        service = SimilarityService()

    return service.compute_similarity(sig1, sig2)
