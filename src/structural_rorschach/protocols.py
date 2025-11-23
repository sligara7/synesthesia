"""
Protocol Definitions for Synesthesia Services

These protocols define the interfaces between services, enabling:
- Mix-and-match implementations (swap services seamlessly)
- Structural typing (no inheritance required)
- Easy testing with mock implementations
- Clear service boundaries and contracts

Key Principle: As long as your service implements the Protocol methods,
it will work with all other Synesthesia services.
"""

from typing import Protocol, Dict, List, Optional, Any, Tuple, runtime_checkable
from dataclasses import dataclass


# ============================================================
# Data Types (used across protocols)
# ============================================================

@dataclass
class SignatureData:
    """Lightweight signature data for protocol communication."""
    source_domain: str
    source_id: str
    source_name: str
    num_nodes: int
    num_edges: int
    motif_vector: Dict[str, float]
    spectral_signature: List[float]


@dataclass
class ResonanceData:
    """Resonance result for protocol communication."""
    query_id: str
    match_id: str
    overall_score: float
    matching_motifs: List[str]
    explanation: str


@dataclass
class SimilarityData:
    """Similarity result for protocol communication."""
    overall_score: float
    motif_similarity: float
    spectral_similarity: float
    scale_similarity: float
    matching_motifs: List[str]


# ============================================================
# Domain Adapter Protocols (F-DA-*)
# ============================================================

@runtime_checkable
class CanAdaptToGraph(Protocol):
    """Protocol for domain adapters that convert data to graphs."""

    @property
    def domain(self) -> str:
        """Return the domain name this adapter handles."""
        ...

    def adapt(self, data: Any, **kwargs) -> Dict:
        """Convert domain-specific data to graph JSON."""
        ...


@runtime_checkable
class ProvidesGraphValidation(Protocol):
    """Protocol for validating graph format."""

    def validate(self, graph_json: Dict) -> Tuple[bool, List[str]]:
        """
        Validate graph JSON against schema.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        ...


# ============================================================
# Signature Extraction Protocols (F-SE-*)
# ============================================================

@runtime_checkable
class CanExtractSignatures(Protocol):
    """Protocol for signature extraction from graphs."""

    def extract(
        self,
        graph: Any,
        domain: str,
        name: str,
        source_id: Optional[str] = None
    ) -> Any:
        """
        Extract structural signature from a graph.

        Args:
            graph: NetworkX graph or graph dict
            domain: Source domain (text, image, music, code)
            name: Human-readable name
            source_id: Optional unique identifier

        Returns:
            StructuralSignature instance
        """
        ...


@runtime_checkable
class CanExtractMotifs(Protocol):
    """Protocol for motif detection in graphs."""

    def detect_motifs(self, graph: Any) -> Dict[str, List[Any]]:
        """
        Detect structural motifs in a graph.

        Returns:
            Dict mapping motif type to list of matches
        """
        ...

    def get_motif_vector(self, graph: Any) -> Dict[str, float]:
        """
        Get normalized motif frequency vector.

        Returns:
            Dict mapping motif type to normalized frequency
        """
        ...

    def get_interpretation(self, motif_type: str, domain: str) -> str:
        """Get human-readable interpretation of motif for domain."""
        ...


# ============================================================
# Similarity Protocols (F-SC-*)
# ============================================================

@runtime_checkable
class CanComputeSimilarity(Protocol):
    """Protocol for computing structural similarity."""

    def compute(self, sig1: Any, sig2: Any) -> SimilarityData:
        """
        Compute structural similarity between two signatures.

        Args:
            sig1: First StructuralSignature
            sig2: Second StructuralSignature

        Returns:
            SimilarityData with overall and component scores
        """
        ...


@runtime_checkable
class CanComputeMotifSimilarity(Protocol):
    """Protocol for motif vector similarity."""

    def compute_motif_similarity(
        self,
        vec1: Dict[str, float],
        vec2: Dict[str, float]
    ) -> Tuple[float, List[str]]:
        """
        Compute cosine similarity between motif vectors.

        Returns:
            Tuple of (similarity_score, matching_motifs)
        """
        ...


@runtime_checkable
class CanComputeSpectralSimilarity(Protocol):
    """Protocol for spectral signature similarity."""

    def compute_spectral_similarity(
        self,
        spec1: List[float],
        spec2: List[float]
    ) -> float:
        """Compute similarity between spectral signatures."""
        ...


# ============================================================
# Corpus Management Protocols (F-CM-*)
# ============================================================

@runtime_checkable
class CanManageCorpus(Protocol):
    """Protocol for corpus CRUD operations."""

    def create(self, name: str, domain: str, description: str = "") -> Any:
        """Create a new corpus."""
        ...

    def add(self, corpus: Any, signature: Any) -> bool:
        """Add signature to corpus."""
        ...

    def load(self, path: str) -> Optional[Any]:
        """Load corpus from path."""
        ...

    def save(self, corpus: Any, path: str) -> bool:
        """Save corpus to path."""
        ...


@runtime_checkable
class CanQueryCorpus(Protocol):
    """Protocol for corpus search operations."""

    def query(
        self,
        corpus: Any,
        query_signature: Any,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[Any, Any]]:
        """
        Find similar signatures in corpus.

        Returns:
            List of (signature, similarity_result) tuples
        """
        ...

    def build_index(self, corpus: Any) -> None:
        """Build/rebuild search index for corpus."""
        ...


# ============================================================
# Resonance Finding Protocols (F-RF-*)
# ============================================================

@runtime_checkable
class CanFindResonances(Protocol):
    """Protocol for finding cross-domain resonances."""

    def find_resonances(
        self,
        query: Any,
        corpus: Any,
        top_k: int = 10,
        threshold: float = 0.3
    ) -> List[Any]:
        """
        Find resonances in a corpus.

        Args:
            query: Query signature
            corpus: Target corpus
            top_k: Maximum results
            threshold: Minimum similarity

        Returns:
            List of Resonance objects
        """
        ...

    def find_cross_domain_resonances(
        self,
        query: Any,
        domain_corpora: Dict[str, Any],
        top_k_per_domain: int = 5,
        threshold: float = 0.3
    ) -> Dict[str, List[Any]]:
        """
        Find resonances across multiple domain corpora.

        Returns:
            Dict mapping domain to list of Resonance objects
        """
        ...


@runtime_checkable
class CanRankResonances(Protocol):
    """Protocol for ranking resonances."""

    def rank_resonances(
        self,
        resonances: List[Any],
        sort_by: str = "overall_score",
        reverse: bool = True
    ) -> List[Any]:
        """Rank resonances by specified criterion."""
        ...


# ============================================================
# Interpretation Protocols (F-IE-*)
# ============================================================

@runtime_checkable
class CanExplainResonance(Protocol):
    """Protocol for generating resonance explanations."""

    def explain_resonance(
        self,
        resonance: Any,
        detail_level: str = "medium"
    ) -> str:
        """
        Generate human-readable explanation for resonance.

        Args:
            resonance: Resonance to explain
            detail_level: "brief", "medium", or "detailed"

        Returns:
            Human-readable explanation string
        """
        ...


@runtime_checkable
class CanExplainMotifs(Protocol):
    """Protocol for explaining motif meanings across domains."""

    def explain_cross_domain_motif(
        self,
        motif_type: str,
        domain1: str,
        domain2: str
    ) -> str:
        """Explain motif meaning across two domains."""
        ...

    def get_motif_interpretation(self, motif_type: str, domain: str) -> str:
        """Get interpretation of motif for a domain."""
        ...


@runtime_checkable
class CanGenerateReports(Protocol):
    """Protocol for generating comparison reports."""

    def generate_comparison_report(self, sig1: Any, sig2: Any) -> str:
        """Generate full structural comparison report."""
        ...


# ============================================================
# Composite Protocols (Multiple Capabilities)
# ============================================================

@runtime_checkable
class FullSimilarityService(Protocol):
    """Full similarity service combining all similarity capabilities."""

    def compute_similarity(self, sig1: Any, sig2: Any) -> Any:
        """Compute overall similarity."""
        ...

    def compute_motif_similarity(
        self,
        vec1: Dict[str, float],
        vec2: Dict[str, float]
    ) -> Tuple[float, List[str]]:
        """Compute motif similarity."""
        ...

    def compute_spectral_similarity(
        self,
        spec1: List[float],
        spec2: List[float]
    ) -> float:
        """Compute spectral similarity."""
        ...

    def compute_scale_similarity(self, sig1: Any, sig2: Any) -> float:
        """Compute scale similarity."""
        ...


@runtime_checkable
class FullCorpusService(Protocol):
    """Full corpus service combining management and query capabilities."""

    def create_corpus(self, name: str, domain: str, description: str = "") -> Any:
        ...

    def add_signature(self, corpus: Any, signature: Any) -> bool:
        ...

    def load_corpus(self, path: str) -> Optional[Any]:
        ...

    def save_corpus(self, corpus: Any, path: str) -> bool:
        ...

    def build_index(self, corpus: Any) -> None:
        ...

    def query(
        self,
        corpus: Any,
        query_signature: Any,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[Any, Any]]:
        ...


@runtime_checkable
class FullInterpretationService(Protocol):
    """Full interpretation service combining all explanation capabilities."""

    def explain_resonance(self, resonance: Any, detail_level: str = "medium") -> str:
        ...

    def explain_cross_domain_motif(
        self,
        motif_type: str,
        domain1: str,
        domain2: str
    ) -> str:
        ...

    def get_motif_interpretation(self, motif_type: str, domain: str) -> str:
        ...

    def generate_comparison_report(self, sig1: Any, sig2: Any) -> str:
        ...
