"""
Structural Signature - Domain-agnostic graph representation

The StructuralSignature serves as the "intermediary" that bridges domains,
enabling comparison between graphs from completely different sources
(images, music, text) based purely on their topology.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json


@dataclass
class StructuralSignature:
    """
    Domain-agnostic structural representation of a graph.

    This is the "common language" that allows us to compare:
    - An image region adjacency graph
    - A word transition graph
    - A musical note sequence graph
    - Any other graph structure

    By stripping away domain-specific semantics and keeping only
    topology, we can find structural "resonances" across domains.
    """

    # === Identity ===
    source_domain: str          # "image", "music", "text", "code", etc.
    source_id: str              # Original graph identifier
    source_name: str            # Human-readable name
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # === Scale Metrics ===
    num_nodes: int = 0
    num_edges: int = 0
    density: float = 0.0        # edges / possible_edges
    is_directed: bool = True

    # === Degree Patterns ===
    # Normalized histogram of node degrees (10 bins)
    degree_distribution: List[float] = field(default_factory=list)
    degree_entropy: float = 0.0       # Shannon entropy of degree dist
    avg_degree: float = 0.0
    max_degree: int = 0
    hub_ratio: float = 0.0            # Fraction of nodes that are hubs (degree > 2*avg)

    # === Clustering Patterns ===
    clustering_coefficient: float = 0.0   # Global clustering
    num_communities: int = 0              # Detected community count
    modularity: float = 0.0               # How separable into clusters
    # Normalized sizes of top communities
    community_sizes: List[float] = field(default_factory=list)

    # === Flow Patterns ===
    avg_path_length: float = 0.0
    diameter: int = 0                 # Longest shortest path
    is_dag: bool = False              # Is it a DAG?
    num_sources: int = 0              # Nodes with no incoming edges
    num_sinks: int = 0                # Nodes with no outgoing edges

    # === Motif Fingerprint ===
    # Normalized counts of structural patterns
    # This is the key "fingerprint" for cross-domain matching
    motif_vector: Dict[str, float] = field(default_factory=dict)
    # Expected keys:
    # - "star_3": 3-node star (hub + 2 leaves)
    # - "chain_3": 3-node chain (a→b→c)
    # - "triangle": closed triangle (a→b→c→a or undirected)
    # - "fork": branching point (a→b, a→c)
    # - "funnel": convergence point (a→c, b→c)
    # - "cycle_4": 4-node cycle
    # - "clique_4": 4-node fully connected

    # === Centrality Patterns ===
    centrality_gini: float = 0.0          # Inequality of node importance
    betweenness_concentration: float = 0.0 # How concentrated is betweenness?
    num_articulation_points: int = 0       # Critical bridge nodes

    # === Spectral Properties ===
    # First k eigenvalues of graph Laplacian (for spectral similarity)
    spectral_signature: List[float] = field(default_factory=list)

    # === Raw Metadata ===
    # Store any domain-specific info that might be useful
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "source_domain": self.source_domain,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "created_at": self.created_at,
            "scale": {
                "num_nodes": self.num_nodes,
                "num_edges": self.num_edges,
                "density": self.density,
                "is_directed": self.is_directed,
            },
            "degree_patterns": {
                "distribution": self.degree_distribution,
                "entropy": self.degree_entropy,
                "avg_degree": self.avg_degree,
                "max_degree": self.max_degree,
                "hub_ratio": self.hub_ratio,
            },
            "clustering": {
                "coefficient": self.clustering_coefficient,
                "num_communities": self.num_communities,
                "modularity": self.modularity,
                "community_sizes": self.community_sizes,
            },
            "flow": {
                "avg_path_length": self.avg_path_length,
                "diameter": self.diameter,
                "is_dag": self.is_dag,
                "num_sources": self.num_sources,
                "num_sinks": self.num_sinks,
            },
            "motif_vector": self.motif_vector,
            "centrality": {
                "gini": self.centrality_gini,
                "betweenness_concentration": self.betweenness_concentration,
                "num_articulation_points": self.num_articulation_points,
            },
            "spectral_signature": self.spectral_signature,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StructuralSignature":
        """Create from dictionary"""
        return cls(
            source_domain=data["source_domain"],
            source_id=data["source_id"],
            source_name=data["source_name"],
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            num_nodes=data["scale"]["num_nodes"],
            num_edges=data["scale"]["num_edges"],
            density=data["scale"]["density"],
            is_directed=data["scale"]["is_directed"],
            degree_distribution=data["degree_patterns"]["distribution"],
            degree_entropy=data["degree_patterns"]["entropy"],
            avg_degree=data["degree_patterns"]["avg_degree"],
            max_degree=data["degree_patterns"]["max_degree"],
            hub_ratio=data["degree_patterns"]["hub_ratio"],
            clustering_coefficient=data["clustering"]["coefficient"],
            num_communities=data["clustering"]["num_communities"],
            modularity=data["clustering"]["modularity"],
            community_sizes=data["clustering"]["community_sizes"],
            avg_path_length=data["flow"]["avg_path_length"],
            diameter=data["flow"]["diameter"],
            is_dag=data["flow"]["is_dag"],
            num_sources=data["flow"]["num_sources"],
            num_sinks=data["flow"]["num_sinks"],
            motif_vector=data["motif_vector"],
            centrality_gini=data["centrality"]["gini"],
            betweenness_concentration=data["centrality"]["betweenness_concentration"],
            num_articulation_points=data["centrality"]["num_articulation_points"],
            spectral_signature=data.get("spectral_signature", []),
            metadata=data.get("metadata", {}),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "StructuralSignature":
        """Deserialize from JSON string"""
        return cls.from_dict(json.loads(json_str))

    def summary(self) -> str:
        """Human-readable summary"""
        return f"""Structural Signature: {self.source_name}
  Domain: {self.source_domain}
  Scale: {self.num_nodes} nodes, {self.num_edges} edges (density: {self.density:.4f})
  Clustering: {self.clustering_coefficient:.3f} (communities: {self.num_communities})
  Flow: avg path {self.avg_path_length:.2f}, diameter {self.diameter}
  Hubs: {self.hub_ratio:.1%} of nodes are hubs
  Top motifs: {self._top_motifs(3)}"""

    def _top_motifs(self, n: int = 3) -> str:
        """Get top n motifs by frequency"""
        if not self.motif_vector:
            return "none detected"
        sorted_motifs = sorted(
            self.motif_vector.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
        return ", ".join(f"{k}={v:.2f}" for k, v in sorted_motifs)


@dataclass
class Resonance:
    """
    A cross-domain structural match.

    Represents a "resonance" between two graphs from different domains
    that share structural similarities.
    """

    # The query (the "inkblot")
    query_domain: str
    query_id: str
    query_name: str

    # The match
    match_domain: str
    match_id: str
    match_name: str

    # Similarity metrics
    overall_score: float          # Combined similarity [0, 1]
    motif_similarity: float       # Motif vector cosine similarity
    spectral_similarity: float    # Spectral signature similarity
    scale_similarity: float       # Size/density similarity

    # Explanation
    matching_motifs: List[str]    # Which motifs contributed most
    shared_properties: Dict[str, float]  # Properties with similar values
    explanation: str              # Human-readable explanation

    # Metadata
    computed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "query": {
                "domain": self.query_domain,
                "id": self.query_id,
                "name": self.query_name,
            },
            "match": {
                "domain": self.match_domain,
                "id": self.match_id,
                "name": self.match_name,
            },
            "similarity": {
                "overall": self.overall_score,
                "motif": self.motif_similarity,
                "spectral": self.spectral_similarity,
                "scale": self.scale_similarity,
            },
            "matching_motifs": self.matching_motifs,
            "shared_properties": self.shared_properties,
            "explanation": self.explanation,
            "computed_at": self.computed_at,
        }

    def summary(self) -> str:
        """Human-readable summary"""
        return f"""Resonance Found!
  Query: [{self.query_domain}] {self.query_name}
  Match: [{self.match_domain}] {self.match_name}
  Score: {self.overall_score:.2%}

  Why they resonate:
  {self.explanation}

  Matching motifs: {', '.join(self.matching_motifs)}"""


@dataclass
class MotifDefinition:
    """
    Definition of a structural motif.

    Motifs are the "vocabulary" of structural patterns that
    are meaningful across domains.
    """

    name: str                     # e.g., "hub_spoke", "chain", "triangle"
    description: str              # What this pattern means structurally
    node_count: int               # Number of nodes in the motif
    edge_pattern: List[Tuple[int, int]]  # Edges as (source_idx, target_idx)
    is_directed: bool             # Whether edge direction matters

    # Cross-domain interpretations
    interpretations: Dict[str, str] = field(default_factory=dict)
    # e.g., {
    #   "image": "Focal point with radial connections",
    #   "music": "Tonic note with harmonic extensions",
    #   "text": "Central theme with supporting details"
    # }

    def matches_subgraph(self, adjacency: List[Tuple[int, int]]) -> bool:
        """Check if this motif pattern matches the given subgraph"""
        # Simplified check - real implementation would use subgraph isomorphism
        if len(adjacency) != len(self.edge_pattern):
            return False
        # TODO: Implement proper isomorphism check
        return True


# Pre-defined motif vocabulary
MOTIF_VOCABULARY = {
    "star_3": MotifDefinition(
        name="star_3",
        description="Central hub connected to 2 peripheral nodes",
        node_count=3,
        edge_pattern=[(0, 1), (0, 2)],
        is_directed=False,
        interpretations={
            "image": "Focal point with radiating features",
            "music": "Root note with chord tones",
            "text": "Main topic with subtopics",
        }
    ),
    "chain_3": MotifDefinition(
        name="chain_3",
        description="Linear sequence of 3 nodes",
        node_count=3,
        edge_pattern=[(0, 1), (1, 2)],
        is_directed=True,
        interpretations={
            "image": "Edge or contour line",
            "music": "Melodic phrase",
            "text": "Narrative sequence",
        }
    ),
    "triangle": MotifDefinition(
        name="triangle",
        description="Fully connected 3-node cycle",
        node_count=3,
        edge_pattern=[(0, 1), (1, 2), (2, 0)],
        is_directed=False,
        interpretations={
            "image": "Enclosed stable region",
            "music": "Triad chord",
            "text": "Circular reference or theme",
        }
    ),
    "fork": MotifDefinition(
        name="fork",
        description="One node diverging to multiple targets",
        node_count=3,
        edge_pattern=[(0, 1), (0, 2)],
        is_directed=True,
        interpretations={
            "image": "Branching structure",
            "music": "Arpeggio or voice split",
            "text": "Enumeration or alternatives",
        }
    ),
    "funnel": MotifDefinition(
        name="funnel",
        description="Multiple nodes converging to one",
        node_count=3,
        edge_pattern=[(0, 2), (1, 2)],
        is_directed=True,
        interpretations={
            "image": "Convergence point",
            "music": "Resolution to tonic",
            "text": "Conclusion or synthesis",
        }
    ),
    "cycle_4": MotifDefinition(
        name="cycle_4",
        description="4-node closed loop",
        node_count=4,
        edge_pattern=[(0, 1), (1, 2), (2, 3), (3, 0)],
        is_directed=True,
        interpretations={
            "image": "Enclosed region, symmetry",
            "music": "Ostinato, repeated pattern",
            "text": "Refrain, callback",
        }
    ),
}
