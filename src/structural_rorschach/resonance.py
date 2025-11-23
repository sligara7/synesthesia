"""
Resonance Finding Service - Find cross-domain structural matches

Implements F-RF-01 through F-RF-04:
- F-RF-01: Find Resonances (single corpus search)
- F-RF-02: Find Cross-Domain Resonances (multi-corpus search)
- F-RF-04: Rank Resonances
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .signature import StructuralSignature, Resonance
from .similarity import SimilarityService, SimilarityResult
from .corpus import Corpus, CorpusService


@dataclass
class ResonanceMatch:
    """A resonance match with full context."""
    query: StructuralSignature
    match: StructuralSignature
    similarity: SimilarityResult
    resonance: Resonance


class ResonanceService:
    """
    Find cross-domain structural matches (resonances).

    This is the core service that enables "structural Rorschach" -
    showing an image and finding what it "sounds like" based on topology.
    """

    def __init__(
        self,
        similarity_service: Optional[SimilarityService] = None,
        corpus_service: Optional[CorpusService] = None
    ):
        self.similarity_service = similarity_service or SimilarityService()
        self.corpus_service = corpus_service or CorpusService(self.similarity_service)

    def find_resonances(
        self,
        query: StructuralSignature,
        corpus: Corpus,
        top_k: int = 10,
        threshold: float = 0.3
    ) -> List[Resonance]:
        """
        Find structurally similar items in a corpus.

        F-RF-01: Find Resonances

        Args:
            query: Query signature (the "inkblot")
            corpus: Target corpus to search
            top_k: Maximum number of results
            threshold: Minimum similarity score

        Returns:
            List of Resonance objects, ranked by similarity
        """
        # Query corpus for similar signatures
        matches = self.corpus_service.query(
            corpus, query, top_k=top_k, threshold=threshold
        )

        # Convert to Resonance objects
        resonances = []
        for match_sig, sim_result in matches:
            resonance = Resonance(
                query_domain=query.source_domain,
                query_id=query.source_id,
                query_name=query.source_name,
                match_domain=match_sig.source_domain,
                match_id=match_sig.source_id,
                match_name=match_sig.source_name,
                overall_score=sim_result.overall_score,
                motif_similarity=sim_result.motif_similarity,
                spectral_similarity=sim_result.spectral_similarity,
                scale_similarity=sim_result.scale_similarity,
                matching_motifs=sim_result.matching_motifs,
                shared_properties=sim_result.details,
                explanation=self._generate_brief_explanation(
                    query, match_sig, sim_result
                )
            )
            resonances.append(resonance)

        return resonances

    def find_cross_domain_resonances(
        self,
        query: StructuralSignature,
        domain_corpora: Dict[str, Corpus],
        top_k_per_domain: int = 5,
        threshold: float = 0.3,
        exclude_same_domain: bool = True
    ) -> Dict[str, List[Resonance]]:
        """
        Find resonances across multiple domain corpora.

        F-RF-02: Find Cross-Domain Resonances

        Args:
            query: Query signature
            domain_corpora: Dict mapping domain name to Corpus
            top_k_per_domain: Max results per domain
            threshold: Minimum similarity score
            exclude_same_domain: Skip query's own domain

        Returns:
            Dict mapping domain name to list of Resonance objects
        """
        results = {}

        for domain_name, corpus in domain_corpora.items():
            # Optionally skip same domain
            if exclude_same_domain and corpus.domain == query.source_domain:
                continue

            resonances = self.find_resonances(
                query, corpus,
                top_k=top_k_per_domain,
                threshold=threshold
            )

            if resonances:
                results[domain_name] = resonances

        return results

    def rank_resonances(
        self,
        resonances: List[Resonance],
        sort_by: str = "overall_score",
        reverse: bool = True
    ) -> List[Resonance]:
        """
        Rank resonances by a specified criterion.

        F-RF-04: Rank Resonances

        Args:
            resonances: List of resonances to rank
            sort_by: Field to sort by (overall_score, motif_similarity, etc.)
            reverse: True for descending order

        Returns:
            Sorted list of resonances
        """
        key_func = lambda r: getattr(r, sort_by, r.overall_score)
        return sorted(resonances, key=key_func, reverse=reverse)

    def _generate_brief_explanation(
        self,
        query: StructuralSignature,
        match: StructuralSignature,
        sim_result: SimilarityResult
    ) -> str:
        """Generate a brief explanation for a resonance."""
        if sim_result.matching_motifs:
            motifs = ", ".join(sim_result.matching_motifs[:2])
            return (
                f"Structural resonance through shared {motifs} patterns. "
                f"The {query.source_domain} structure has similar topology "
                f"to this {match.source_domain} structure."
            )
        return (
            f"Structural similarity based on graph topology "
            f"({sim_result.overall_score:.0%} overall match)."
        )

    def find_best_resonance(
        self,
        query: StructuralSignature,
        corpus: Corpus,
        threshold: float = 0.3
    ) -> Optional[Resonance]:
        """
        Find the single best resonance match.

        Convenience method for getting the top match.

        Args:
            query: Query signature
            corpus: Target corpus
            threshold: Minimum similarity score

        Returns:
            Best Resonance or None if no matches above threshold
        """
        resonances = self.find_resonances(
            query, corpus, top_k=1, threshold=threshold
        )
        return resonances[0] if resonances else None


# Convenience functions
def find_resonances(
    query: StructuralSignature,
    corpus: Corpus,
    top_k: int = 10,
    threshold: float = 0.3
) -> List[Resonance]:
    """Find resonances in a corpus."""
    service = ResonanceService()
    return service.find_resonances(query, corpus, top_k, threshold)


def find_cross_domain_resonances(
    query: StructuralSignature,
    domain_corpora: Dict[str, Corpus],
    top_k_per_domain: int = 5,
    threshold: float = 0.3
) -> Dict[str, List[Resonance]]:
    """Find resonances across multiple domains."""
    service = ResonanceService()
    return service.find_cross_domain_resonances(
        query, domain_corpora, top_k_per_domain, threshold
    )
