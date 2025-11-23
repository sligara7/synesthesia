"""
Spectral Signature - Linear algebra-based graph compression and analysis

Uses SVD and Laplacian eigendecomposition to create compact, domain-agnostic
representations of graphs that can be compared efficiently regardless of size.

Key insight: The spectrum (eigenvalues/singular values) captures structural
properties without expensive graph algorithms:
- Community structure → Small Laplacian eigenvalues
- Connectivity → Fiedler value (λ₂)
- Hub structure → Singular value concentration
- Graph complexity → Spectral decay rate

Complexity: O(nk²) instead of O(n³) for full graph analysis
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

try:
    import numpy as np
except ImportError:
    raise ImportError("NumPy required: pip install numpy")

try:
    import networkx as nx
except ImportError:
    raise ImportError("NetworkX required: pip install networkx")

try:
    from scipy.sparse.linalg import svds, eigsh
    from scipy.sparse import csr_matrix
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class SpectralSignature:
    """
    Compressed graph representation using spectral decomposition.

    This is a lightweight alternative to StructuralSignature that scales
    to very large graphs by using linear algebra instead of graph algorithms.
    """

    # Identity
    source_domain: str
    source_id: str
    source_name: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Basic scale (cheap to compute)
    num_nodes: int = 0
    num_edges: int = 0
    density: float = 0.0

    # SVD of adjacency matrix
    singular_values: List[float] = field(default_factory=list)
    spectral_decay: float = 0.0          # sv[0] / sv[-1] - how fast values decay
    effective_rank: int = 0              # Number of significant singular values
    sv_entropy: float = 0.0              # Entropy of normalized singular values

    # Laplacian spectrum
    laplacian_eigenvalues: List[float] = field(default_factory=list)
    fiedler_value: float = 0.0           # λ₂ - algebraic connectivity
    spectral_gap: float = 0.0            # λ₂ / λ_max - community strength

    # Derived estimates (from spectrum, no graph traversal needed)
    estimated_communities: int = 0
    estimated_diameter: int = 0
    connectivity_score: float = 0.0

    # Compression stats
    compression_ratio: float = 0.0       # Original size / compressed size
    k_components: int = 0                # Number of spectral components used

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
            },
            "svd": {
                "singular_values": self.singular_values,
                "spectral_decay": self.spectral_decay,
                "effective_rank": self.effective_rank,
                "sv_entropy": self.sv_entropy,
            },
            "laplacian": {
                "eigenvalues": self.laplacian_eigenvalues,
                "fiedler_value": self.fiedler_value,
                "spectral_gap": self.spectral_gap,
            },
            "estimates": {
                "communities": self.estimated_communities,
                "diameter": self.estimated_diameter,
                "connectivity_score": self.connectivity_score,
            },
            "compression": {
                "ratio": self.compression_ratio,
                "k_components": self.k_components,
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpectralSignature":
        """Create from dictionary"""
        return cls(
            source_domain=data["source_domain"],
            source_id=data["source_id"],
            source_name=data["source_name"],
            created_at=data.get("created_at", ""),
            num_nodes=data["scale"]["num_nodes"],
            num_edges=data["scale"]["num_edges"],
            density=data["scale"]["density"],
            singular_values=data["svd"]["singular_values"],
            spectral_decay=data["svd"]["spectral_decay"],
            effective_rank=data["svd"]["effective_rank"],
            sv_entropy=data["svd"]["sv_entropy"],
            laplacian_eigenvalues=data["laplacian"]["eigenvalues"],
            fiedler_value=data["laplacian"]["fiedler_value"],
            spectral_gap=data["laplacian"]["spectral_gap"],
            estimated_communities=data["estimates"]["communities"],
            estimated_diameter=data["estimates"]["diameter"],
            connectivity_score=data["estimates"]["connectivity_score"],
            compression_ratio=data["compression"]["ratio"],
            k_components=data["compression"]["k_components"],
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Human-readable summary"""
        return f"""Spectral Signature: {self.source_name}
  Domain: {self.source_domain}
  Scale: {self.num_nodes} nodes, {self.num_edges} edges
  Compression: {self.compression_ratio:.1f}x ({self.k_components} components)

  Spectral Properties:
    Effective rank: {self.effective_rank} (of {self.num_nodes})
    Fiedler value (connectivity): {self.fiedler_value:.4f}
    Spectral gap (community strength): {self.spectral_gap:.4f}

  Structural Estimates:
    Communities: ~{self.estimated_communities}
    Diameter: ~{self.estimated_diameter}
    Connectivity score: {self.connectivity_score:.4f}"""


class SpectralExtractor:
    """
    Extract spectral signatures from graphs.

    Uses sparse linear algebra for efficiency on large graphs.
    Complexity: O(nk²) where k << n is the number of components.
    """

    def __init__(self, k_components: int = 50, energy_threshold: float = 0.95):
        """
        Initialize the extractor.

        Args:
            k_components: Number of spectral components to compute
            energy_threshold: Fraction of spectral energy to capture for effective rank
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy required for spectral extraction: pip install scipy")

        self.k_components = k_components
        self.energy_threshold = energy_threshold

    def extract_from_networkx(
        self,
        G: nx.Graph,
        domain: str,
        name: str,
        source_id: str
    ) -> SpectralSignature:
        """
        Extract spectral signature from a NetworkX graph.

        Args:
            G: NetworkX graph
            domain: Domain identifier
            name: Human-readable name
            source_id: Unique identifier

        Returns:
            SpectralSignature
        """
        n = G.number_of_nodes()
        m = G.number_of_edges()

        if n == 0:
            return SpectralSignature(
                source_domain=domain,
                source_id=source_id,
                source_name=name,
                num_nodes=0,
                num_edges=0,
            )

        # Compute k (number of components to extract)
        k = min(self.k_components, n - 1)
        if k < 2:
            k = min(2, n - 1)

        # Density
        if n > 1:
            max_edges = n * (n - 1) if G.is_directed() else n * (n - 1) / 2
            density = m / max_edges if max_edges > 0 else 0
        else:
            density = 0

        # === SVD of Adjacency Matrix ===
        singular_values, sv_entropy, effective_rank, spectral_decay = \
            self._compute_svd_features(G, k)

        # === Laplacian Spectrum ===
        laplacian_eigs, fiedler_value, spectral_gap = \
            self._compute_laplacian_features(G, k)

        # === Derived Estimates ===
        # Estimate number of communities from spectral gap
        # Small eigenvalues indicate separate communities
        if len(laplacian_eigs) > 1 and laplacian_eigs[-1] > 0:
            threshold = 0.1 * laplacian_eigs[-1]
            estimated_communities = max(1, sum(1 for e in laplacian_eigs if e < threshold))
        else:
            estimated_communities = 1

        # Estimate diameter from spectral gap
        if spectral_gap > 0 and spectral_gap < 1:
            estimated_diameter = int(np.ceil(np.log(n) / np.log(1 / spectral_gap)))
            estimated_diameter = min(estimated_diameter, n)  # Cap at n
        else:
            estimated_diameter = n

        # Connectivity score (normalized Fiedler value)
        connectivity_score = fiedler_value / np.sqrt(n) if n > 0 else 0

        # Compression ratio: original matrix size / spectral representation
        original_size = n * n  # Full adjacency matrix
        compressed_size = k * 2 + k  # k singular values + k eigenvalues + metadata
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1

        return SpectralSignature(
            source_domain=domain,
            source_id=source_id,
            source_name=name,
            num_nodes=n,
            num_edges=m,
            density=density,
            singular_values=singular_values,
            spectral_decay=spectral_decay,
            effective_rank=effective_rank,
            sv_entropy=sv_entropy,
            laplacian_eigenvalues=laplacian_eigs,
            fiedler_value=fiedler_value,
            spectral_gap=spectral_gap,
            estimated_communities=estimated_communities,
            estimated_diameter=estimated_diameter,
            connectivity_score=connectivity_score,
            compression_ratio=compression_ratio,
            k_components=k,
        )

    def _compute_svd_features(
        self,
        G: nx.Graph,
        k: int
    ) -> Tuple[List[float], float, int, float]:
        """Compute SVD-based features from adjacency matrix"""
        try:
            # Get sparse adjacency matrix
            A = nx.adjacency_matrix(G).astype(float)

            # Truncated SVD (only top-k singular values)
            # This is O(nk²) instead of O(n³)
            U, S, Vt = svds(A, k=k)

            # Sort in descending order
            idx = np.argsort(S)[::-1]
            S = S[idx]

            singular_values = S.tolist()

            # Spectral decay: ratio of largest to smallest
            spectral_decay = S[0] / S[-1] if S[-1] > 1e-10 else float('inf')

            # Effective rank: number of singular values capturing threshold of energy
            total_energy = np.sum(S ** 2)
            if total_energy > 0:
                cumulative = np.cumsum(S ** 2) / total_energy
                effective_rank = int(np.searchsorted(cumulative, self.energy_threshold) + 1)
            else:
                effective_rank = 0

            # Entropy of normalized singular values
            S_norm = S / np.sum(S) if np.sum(S) > 0 else S
            S_norm = S_norm[S_norm > 1e-10]  # Remove near-zeros
            sv_entropy = float(-np.sum(S_norm * np.log2(S_norm))) if len(S_norm) > 0 else 0

            return singular_values, sv_entropy, effective_rank, spectral_decay

        except Exception as e:
            print(f"Warning: SVD computation failed: {e}")
            return [], 0.0, 0, 0.0

    def _compute_laplacian_features(
        self,
        G: nx.Graph,
        k: int
    ) -> Tuple[List[float], float, float]:
        """Compute Laplacian spectrum features"""
        try:
            # Get Laplacian matrix (sparse)
            if G.is_directed():
                G_undirected = G.to_undirected()
            else:
                G_undirected = G

            L = nx.laplacian_matrix(G_undirected).astype(float)

            # Get smallest eigenvalues (they reveal community structure)
            # 'SM' = smallest magnitude
            eigenvalues, _ = eigsh(L, k=k, which='SM')
            eigenvalues = np.sort(np.real(eigenvalues))

            laplacian_eigs = eigenvalues.tolist()

            # Fiedler value: second smallest eigenvalue (first is always 0)
            fiedler_value = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0

            # Spectral gap: ratio of Fiedler to max eigenvalue
            max_eig = eigenvalues[-1] if len(eigenvalues) > 0 else 1
            spectral_gap = fiedler_value / max_eig if max_eig > 1e-10 else 0

            return laplacian_eigs, fiedler_value, spectral_gap

        except Exception as e:
            print(f"Warning: Laplacian computation failed: {e}")
            return [], 0.0, 0.0

    def extract_from_file(
        self,
        file_path: str,
        domain: str = "unknown",
        name: str = None
    ) -> SpectralSignature:
        """Extract spectral signature from a graph JSON file"""
        from pathlib import Path

        path = Path(file_path)
        with open(path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)

        if name is None:
            name = graph_data.get('metadata', {}).get('title', path.stem)

        G = self._json_to_networkx(graph_data)
        return self.extract_from_networkx(G, domain, name, str(path))

    def _json_to_networkx(self, graph_data: dict) -> nx.Graph:
        """Convert system_of_systems_graph.json to NetworkX"""
        graph_section = graph_data.get('graph', {})
        is_directed = graph_section.get('directed', True)

        G = nx.DiGraph() if is_directed else nx.Graph()

        for node in graph_section.get('nodes', []):
            node_id = node.get('id', node.get('name'))
            G.add_node(node_id)

        for link in graph_section.get('links', []):
            source = link.get('source')
            target = link.get('target')
            weight = link.get('weight', 1.0)
            G.add_edge(source, target, weight=weight)

        return G


def spectral_similarity(
    sig1: SpectralSignature,
    sig2: SpectralSignature,
    weights: Dict[str, float] = None
) -> float:
    """
    Compute similarity between two spectral signatures.

    This is the core cross-domain comparison function.
    Two graphs with similar spectra have similar structure.

    Args:
        sig1: First spectral signature
        sig2: Second spectral signature
        weights: Optional weights for different components

    Returns:
        Similarity score [0, 1]
    """
    if weights is None:
        weights = {
            "singular_values": 0.4,
            "laplacian": 0.3,
            "derived": 0.3,
        }

    # Normalize singular values for comparison
    sv1 = np.array(sig1.singular_values)
    sv2 = np.array(sig2.singular_values)

    if len(sv1) > 0 and sv1[0] > 0:
        sv1 = sv1 / sv1[0]
    if len(sv2) > 0 and sv2[0] > 0:
        sv2 = sv2 / sv2[0]

    # Pad to same length
    max_len = max(len(sv1), len(sv2))
    sv1 = np.pad(sv1, (0, max_len - len(sv1)))
    sv2 = np.pad(sv2, (0, max_len - len(sv2)))

    # Singular value similarity (cosine)
    sv_sim = 0.0
    if np.linalg.norm(sv1) > 0 and np.linalg.norm(sv2) > 0:
        sv_sim = np.dot(sv1, sv2) / (np.linalg.norm(sv1) * np.linalg.norm(sv2))

    # Laplacian eigenvalue similarity
    le1 = np.array(sig1.laplacian_eigenvalues)
    le2 = np.array(sig2.laplacian_eigenvalues)

    # Normalize by max
    if len(le1) > 0 and le1[-1] > 0:
        le1 = le1 / le1[-1]
    if len(le2) > 0 and le2[-1] > 0:
        le2 = le2 / le2[-1]

    max_len = max(len(le1), len(le2))
    le1 = np.pad(le1, (0, max_len - len(le1)))
    le2 = np.pad(le2, (0, max_len - len(le2)))

    lap_sim = 0.0
    if np.linalg.norm(le1) > 0 and np.linalg.norm(le2) > 0:
        lap_sim = np.dot(le1, le2) / (np.linalg.norm(le1) * np.linalg.norm(le2))

    # Derived metrics similarity
    derived_metrics = [
        (sig1.spectral_gap, sig2.spectral_gap),
        (sig1.connectivity_score, sig2.connectivity_score),
        (sig1.sv_entropy, sig2.sv_entropy),
    ]

    derived_sims = []
    for v1, v2 in derived_metrics:
        max_val = max(abs(v1), abs(v2))
        if max_val > 0:
            derived_sims.append(1 - abs(v1 - v2) / max_val)
        else:
            derived_sims.append(1.0)

    derived_sim = np.mean(derived_sims) if derived_sims else 0.0

    # Weighted combination
    total = (
        weights["singular_values"] * sv_sim +
        weights["laplacian"] * lap_sim +
        weights["derived"] * derived_sim
    )

    return float(total)
