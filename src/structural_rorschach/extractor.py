"""
Signature Extractor - Extract StructuralSignature from any graph

This module converts domain-specific graphs (image, music, text) into
domain-agnostic structural signatures that can be compared across domains.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter

try:
    import networkx as nx
except ImportError:
    raise ImportError("NetworkX required: pip install networkx")

try:
    import numpy as np
except ImportError:
    raise ImportError("NumPy required: pip install numpy")

from .signature import StructuralSignature, MOTIF_VOCABULARY


class SignatureExtractor:
    """
    Extract domain-agnostic structural signatures from graphs.

    Supports:
    - NetworkX graphs directly
    - system_of_systems_graph.json format
    - Raw adjacency lists
    """

    def __init__(self, num_degree_bins: int = 10, num_spectral_values: int = 10):
        """
        Initialize the extractor.

        Args:
            num_degree_bins: Number of bins for degree distribution histogram
            num_spectral_values: Number of Laplacian eigenvalues to keep
        """
        self.num_degree_bins = num_degree_bins
        self.num_spectral_values = num_spectral_values

    def extract_from_file(
        self,
        file_path: str,
        domain: str = "unknown",
        name: Optional[str] = None
    ) -> StructuralSignature:
        """
        Extract signature from a graph JSON file.

        Args:
            file_path: Path to system_of_systems_graph.json format file
            domain: Domain identifier (image, music, text, etc.)
            name: Human-readable name (defaults to filename)

        Returns:
            StructuralSignature for the graph
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Graph file not found: {file_path}")

        with open(path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)

        if name is None:
            name = graph_data.get('metadata', {}).get('title', path.stem)

        # Detect domain from metadata if not specified
        if domain == "unknown":
            framework = graph_data.get('metadata', {}).get('framework_id', '')
            if 'image' in framework.lower():
                domain = 'image'
            elif 'music' in framework.lower() or 'midi' in framework.lower():
                domain = 'music'
            elif 'text' in framework.lower() or 'word' in framework.lower():
                domain = 'text'

        return self.extract_from_json(graph_data, domain, name, str(path))

    def extract_from_json(
        self,
        graph_data: dict,
        domain: str,
        name: str,
        source_id: str
    ) -> StructuralSignature:
        """
        Extract signature from graph JSON data.

        Args:
            graph_data: Dictionary in system_of_systems_graph.json format
            domain: Domain identifier
            name: Human-readable name
            source_id: Unique identifier for the source

        Returns:
            StructuralSignature
        """
        # Convert to NetworkX graph
        G = self._json_to_networkx(graph_data)

        return self.extract_from_networkx(G, domain, name, source_id, graph_data)

    def extract_from_networkx(
        self,
        G: nx.Graph,
        domain: str,
        name: str,
        source_id: str,
        original_data: Optional[dict] = None
    ) -> StructuralSignature:
        """
        Extract signature from a NetworkX graph.

        Args:
            G: NetworkX graph (DiGraph or Graph)
            domain: Domain identifier
            name: Human-readable name
            source_id: Unique identifier
            original_data: Optional original JSON data for metadata

        Returns:
            StructuralSignature
        """
        is_directed = G.is_directed()

        # Handle empty or trivial graphs
        if G.number_of_nodes() == 0:
            return StructuralSignature(
                source_domain=domain,
                source_id=source_id,
                source_name=name,
                num_nodes=0,
                num_edges=0,
                metadata={"warning": "Empty graph"}
            )

        # === Scale Metrics ===
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        # Density calculation
        if num_nodes > 1:
            if is_directed:
                max_edges = num_nodes * (num_nodes - 1)
            else:
                max_edges = num_nodes * (num_nodes - 1) / 2
            density = num_edges / max_edges if max_edges > 0 else 0
        else:
            density = 0

        # === Degree Patterns ===
        degrees = [d for n, d in G.degree()]
        degree_dist, degree_entropy, avg_degree, max_degree, hub_ratio = \
            self._compute_degree_metrics(degrees, num_nodes)

        # === Clustering Patterns ===
        clustering_coef, num_communities, modularity, community_sizes = \
            self._compute_clustering_metrics(G)

        # === Flow Patterns ===
        avg_path, diameter, is_dag, num_sources, num_sinks = \
            self._compute_flow_metrics(G, is_directed)

        # === Motif Vector ===
        motif_vector = self._compute_motif_vector(G)

        # === Centrality Patterns ===
        centrality_gini, betweenness_conc, num_articulation = \
            self._compute_centrality_metrics(G)

        # === Spectral Properties ===
        spectral_sig = self._compute_spectral_signature(G)

        # === Metadata ===
        metadata = {}
        if original_data:
            metadata = original_data.get('metadata', {})

        return StructuralSignature(
            source_domain=domain,
            source_id=source_id,
            source_name=name,
            num_nodes=num_nodes,
            num_edges=num_edges,
            density=density,
            is_directed=is_directed,
            degree_distribution=degree_dist,
            degree_entropy=degree_entropy,
            avg_degree=avg_degree,
            max_degree=max_degree,
            hub_ratio=hub_ratio,
            clustering_coefficient=clustering_coef,
            num_communities=num_communities,
            modularity=modularity,
            community_sizes=community_sizes,
            avg_path_length=avg_path,
            diameter=diameter,
            is_dag=is_dag,
            num_sources=num_sources,
            num_sinks=num_sinks,
            motif_vector=motif_vector,
            centrality_gini=centrality_gini,
            betweenness_concentration=betweenness_conc,
            num_articulation_points=num_articulation,
            spectral_signature=spectral_sig,
            metadata=metadata,
        )

    def _json_to_networkx(self, graph_data: dict) -> nx.Graph:
        """Convert system_of_systems_graph.json to NetworkX"""
        graph_section = graph_data.get('graph', {})

        # Determine if directed
        is_directed = graph_section.get('directed', True)

        if is_directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()

        # Add nodes
        for node in graph_section.get('nodes', []):
            node_id = node.get('id', node.get('name'))
            G.add_node(node_id)

        # Add edges (links)
        for link in graph_section.get('links', []):
            source = link.get('source')
            target = link.get('target')
            weight = link.get('weight', 1.0)
            G.add_edge(source, target, weight=weight)

        return G

    def _compute_degree_metrics(
        self,
        degrees: List[int],
        num_nodes: int
    ) -> Tuple[List[float], float, float, int, float]:
        """Compute degree-related metrics"""
        if not degrees:
            return [], 0.0, 0.0, 0, 0.0

        # Degree distribution (normalized histogram)
        max_deg = max(degrees)
        if max_deg == 0:
            degree_dist = [1.0] + [0.0] * (self.num_degree_bins - 1)
        else:
            bins = np.linspace(0, max_deg + 1, self.num_degree_bins + 1)
            hist, _ = np.histogram(degrees, bins=bins)
            degree_dist = (hist / num_nodes).tolist()

        # Degree entropy
        if num_nodes > 0:
            degree_probs = np.array(degrees) / sum(degrees) if sum(degrees) > 0 else np.zeros(len(degrees))
            degree_probs = degree_probs[degree_probs > 0]  # Remove zeros for log
            degree_entropy = -np.sum(degree_probs * np.log2(degree_probs)) if len(degree_probs) > 0 else 0.0
        else:
            degree_entropy = 0.0

        avg_degree = np.mean(degrees)
        max_degree = max(degrees)

        # Hub ratio: nodes with degree > 2 * average
        hub_threshold = 2 * avg_degree
        num_hubs = sum(1 for d in degrees if d > hub_threshold)
        hub_ratio = num_hubs / num_nodes if num_nodes > 0 else 0.0

        return degree_dist, float(degree_entropy), float(avg_degree), int(max_degree), float(hub_ratio)

    def _compute_clustering_metrics(
        self,
        G: nx.Graph
    ) -> Tuple[float, int, float, List[float]]:
        """Compute clustering-related metrics"""
        # Clustering coefficient
        try:
            if G.is_directed():
                clustering_coef = nx.average_clustering(G.to_undirected())
            else:
                clustering_coef = nx.average_clustering(G)
        except Exception:
            clustering_coef = 0.0

        # Community detection
        try:
            if G.is_directed():
                G_undirected = G.to_undirected()
            else:
                G_undirected = G

            if G_undirected.number_of_nodes() > 1:
                communities = list(nx.community.greedy_modularity_communities(G_undirected))
                num_communities = len(communities)
                modularity = nx.community.modularity(G_undirected, communities)

                # Normalized community sizes (top 5)
                sizes = sorted([len(c) for c in communities], reverse=True)[:5]
                total = G.number_of_nodes()
                community_sizes = [s / total for s in sizes]
            else:
                num_communities = 1
                modularity = 0.0
                community_sizes = [1.0]
        except Exception:
            num_communities = 1
            modularity = 0.0
            community_sizes = [1.0]

        return float(clustering_coef), num_communities, float(modularity), community_sizes

    def _compute_flow_metrics(
        self,
        G: nx.Graph,
        is_directed: bool
    ) -> Tuple[float, int, bool, int, int]:
        """Compute flow-related metrics"""
        # Average path length and diameter
        try:
            if is_directed:
                # Use largest strongly connected component
                if nx.is_strongly_connected(G):
                    avg_path = nx.average_shortest_path_length(G)
                    diameter = nx.diameter(G)
                else:
                    # Use largest SCC
                    largest_scc = max(nx.strongly_connected_components(G), key=len)
                    subgraph = G.subgraph(largest_scc)
                    if len(largest_scc) > 1:
                        avg_path = nx.average_shortest_path_length(subgraph)
                        diameter = nx.diameter(subgraph)
                    else:
                        avg_path = 0.0
                        diameter = 0
            else:
                if nx.is_connected(G):
                    avg_path = nx.average_shortest_path_length(G)
                    diameter = nx.diameter(G)
                else:
                    # Use largest component
                    largest_cc = max(nx.connected_components(G), key=len)
                    subgraph = G.subgraph(largest_cc)
                    if len(largest_cc) > 1:
                        avg_path = nx.average_shortest_path_length(subgraph)
                        diameter = nx.diameter(subgraph)
                    else:
                        avg_path = 0.0
                        diameter = 0
        except Exception:
            avg_path = 0.0
            diameter = 0

        # DAG check
        is_dag = nx.is_directed_acyclic_graph(G) if is_directed else False

        # Sources and sinks (for directed graphs)
        if is_directed:
            num_sources = sum(1 for n in G.nodes() if G.in_degree(n) == 0)
            num_sinks = sum(1 for n in G.nodes() if G.out_degree(n) == 0)
        else:
            num_sources = 0
            num_sinks = 0

        return float(avg_path), int(diameter), is_dag, num_sources, num_sinks

    def _compute_motif_vector(self, G: nx.Graph) -> Dict[str, float]:
        """
        Compute normalized motif frequencies.

        This is the key "fingerprint" for cross-domain matching.
        """
        motif_counts = Counter()
        num_nodes = G.number_of_nodes()

        if num_nodes < 3:
            return {name: 0.0 for name in MOTIF_VOCABULARY.keys()}

        # Count triangles
        if G.is_directed():
            G_undirected = G.to_undirected()
        else:
            G_undirected = G

        triangles = sum(nx.triangles(G_undirected).values()) // 3
        motif_counts['triangle'] = triangles

        # Count star patterns (nodes with degree >= 2 as centers)
        for node in G.nodes():
            degree = G.degree(node)
            if degree >= 2:
                # Each high-degree node contributes to star_3 count
                # C(degree, 2) = number of 3-stars centered at this node
                motif_counts['star_3'] += degree * (degree - 1) // 2

        # Count chains (approximate by looking at degree-2 nodes)
        for node in G.nodes():
            if G.is_directed():
                in_deg = G.in_degree(node)
                out_deg = G.out_degree(node)
                # Node in middle of chain: 1 in, 1 out
                if in_deg == 1 and out_deg == 1:
                    motif_counts['chain_3'] += 1
                # Fork: 1 in, 2+ out
                if in_deg <= 1 and out_deg >= 2:
                    motif_counts['fork'] += 1
                # Funnel: 2+ in, 1 out
                if in_deg >= 2 and out_deg <= 1:
                    motif_counts['funnel'] += 1
            else:
                # Undirected: degree-2 nodes are chain midpoints
                if G.degree(node) == 2:
                    motif_counts['chain_3'] += 1

        # Count 4-cycles (approximate)
        try:
            cycle_basis = nx.cycle_basis(G_undirected)
            motif_counts['cycle_4'] = sum(1 for c in cycle_basis if len(c) == 4)
        except Exception:
            motif_counts['cycle_4'] = 0

        # Normalize to [0, 1] range relative to graph size
        total_possible = max(1, num_nodes * (num_nodes - 1) * (num_nodes - 2) // 6)
        motif_vector = {}
        for name in MOTIF_VOCABULARY.keys():
            count = motif_counts.get(name, 0)
            motif_vector[name] = min(1.0, count / total_possible) if total_possible > 0 else 0.0

        return motif_vector

    def _compute_centrality_metrics(
        self,
        G: nx.Graph
    ) -> Tuple[float, float, int]:
        """Compute centrality-related metrics"""
        if G.number_of_nodes() < 2:
            return 0.0, 0.0, 0

        # Betweenness centrality
        try:
            betweenness = nx.betweenness_centrality(G)
            bc_values = list(betweenness.values())

            # Gini coefficient of centrality
            centrality_gini = self._gini_coefficient(bc_values)

            # Betweenness concentration: max / mean
            max_bc = max(bc_values) if bc_values else 0
            mean_bc = np.mean(bc_values) if bc_values else 0
            betweenness_conc = max_bc / mean_bc if mean_bc > 0 else 0
        except Exception:
            centrality_gini = 0.0
            betweenness_conc = 0.0

        # Articulation points (for undirected)
        try:
            if G.is_directed():
                G_undirected = G.to_undirected()
            else:
                G_undirected = G
            num_articulation = len(list(nx.articulation_points(G_undirected)))
        except Exception:
            num_articulation = 0

        return float(centrality_gini), float(betweenness_conc), num_articulation

    def _compute_spectral_signature(self, G: nx.Graph) -> List[float]:
        """Compute spectral signature (Laplacian eigenvalues)"""
        try:
            if G.is_directed():
                G_undirected = G.to_undirected()
            else:
                G_undirected = G

            if G_undirected.number_of_nodes() < 2:
                return []

            # Get Laplacian spectrum
            eigenvalues = nx.laplacian_spectrum(G_undirected)
            eigenvalues = sorted(eigenvalues)[:self.num_spectral_values]

            # Normalize by largest eigenvalue
            max_eig = max(abs(e) for e in eigenvalues) if eigenvalues else 1
            normalized = [float(e / max_eig) if max_eig > 0 else 0.0 for e in eigenvalues]

            return normalized
        except Exception:
            return []

    def _gini_coefficient(self, values: List[float]) -> float:
        """Compute Gini coefficient (inequality measure)"""
        if not values or len(values) < 2:
            return 0.0

        values = sorted(values)
        n = len(values)
        total = sum(values)

        if total == 0:
            return 0.0

        # Gini formula
        numerator = sum((2 * i - n - 1) * v for i, v in enumerate(values, 1))
        return numerator / (n * total)


def extract_signature(
    file_path: str,
    domain: str = "unknown",
    name: Optional[str] = None
) -> StructuralSignature:
    """
    Convenience function to extract signature from a file.

    Args:
        file_path: Path to graph JSON file
        domain: Domain identifier
        name: Human-readable name

    Returns:
        StructuralSignature
    """
    extractor = SignatureExtractor()
    return extractor.extract_from_file(file_path, domain, name)
