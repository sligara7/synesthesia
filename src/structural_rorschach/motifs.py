"""
Motif Detector - Identify structural patterns in graphs

Motifs are the "vocabulary" of structural patterns that are
meaningful across domains. This module detects and counts
these patterns in any graph.
"""

from typing import Dict, List, Set, Tuple, Optional
from collections import Counter
from dataclasses import dataclass

try:
    import networkx as nx
except ImportError:
    raise ImportError("NetworkX required: pip install networkx")


@dataclass
class MotifMatch:
    """A detected motif instance in a graph"""
    motif_type: str
    nodes: Tuple[str, ...]
    edges: List[Tuple[str, str]]
    central_node: Optional[str] = None  # For star-like patterns


class MotifDetector:
    """
    Detect and count structural motifs in graphs.

    Motifs serve as the "bridge language" between domains:
    - A star pattern in an image (focal point) can match
      a star pattern in music (tonic with extensions)
    """

    # Motif definitions with their structural meaning
    MOTIF_DEFINITIONS = {
        "hub_spoke": {
            "description": "Central node connected to multiple peripherals",
            "min_spokes": 3,
            "interpretations": {
                "image": "Focal point with radial features (flower, eye, sun)",
                "music": "Tonic/root with harmonic extensions",
                "text": "Central theme with supporting details",
            }
        },
        "chain": {
            "description": "Linear sequence of nodes",
            "min_length": 3,
            "interpretations": {
                "image": "Edge, contour, or flow line",
                "music": "Melodic phrase or scale passage",
                "text": "Narrative sequence or logical progression",
            }
        },
        "cluster": {
            "description": "Densely interconnected group of nodes",
            "min_density": 0.5,
            "interpretations": {
                "image": "Cohesive region or object",
                "music": "Chord voicing or tonal cluster",
                "text": "Topic cluster or paragraph",
            }
        },
        "bridge": {
            "description": "Node connecting otherwise separate communities",
            "interpretations": {
                "image": "Transition zone between regions",
                "music": "Pivot chord or modulation point",
                "text": "Transitional phrase or plot turn",
            }
        },
        "cycle": {
            "description": "Closed loop of nodes",
            "min_length": 3,
            "interpretations": {
                "image": "Enclosed shape, symmetry",
                "music": "Ostinato, repeated pattern, loop",
                "text": "Refrain, circular argument, callback",
            }
        },
        "star_3": {
            "description": "One hub with exactly 2 spokes",
            "interpretations": {
                "image": "Simple branching point",
                "music": "Dyad or simple interval",
                "text": "Binary choice or comparison",
            }
        },
        "triangle": {
            "description": "Three fully connected nodes",
            "interpretations": {
                "image": "Stable triangular region",
                "music": "Triad (major/minor chord)",
                "text": "Triangular relationship, trio",
            }
        },
        "fork": {
            "description": "One node diverging to multiple targets",
            "interpretations": {
                "image": "Branching, splitting",
                "music": "Voice splitting, arpeggio start",
                "text": "Enumeration, alternatives",
            }
        },
        "funnel": {
            "description": "Multiple nodes converging to one",
            "interpretations": {
                "image": "Convergence, focus",
                "music": "Resolution, cadence",
                "text": "Conclusion, synthesis",
            }
        },
    }

    def __init__(self, G: nx.Graph):
        """
        Initialize detector with a graph.

        Args:
            G: NetworkX graph (directed or undirected)
        """
        self.G = G
        self.is_directed = G.is_directed()

        # Convert to undirected for some analyses
        if self.is_directed:
            self.G_undirected = G.to_undirected()
        else:
            self.G_undirected = G

    def detect_all(self) -> Dict[str, List[MotifMatch]]:
        """
        Detect all motif types in the graph.

        Returns:
            Dictionary mapping motif type to list of matches
        """
        results = {}

        results['hub_spoke'] = self.detect_hub_spokes()
        results['chain'] = self.detect_chains()
        results['triangle'] = self.detect_triangles()
        results['bridge'] = self.detect_bridges()
        results['cycle'] = self.detect_cycles()
        results['fork'] = self.detect_forks()
        results['funnel'] = self.detect_funnels()

        return results

    def count_all(self) -> Dict[str, int]:
        """
        Count occurrences of each motif type.

        Returns:
            Dictionary mapping motif type to count
        """
        all_motifs = self.detect_all()
        return {motif_type: len(matches) for motif_type, matches in all_motifs.items()}

    def get_motif_vector(self, normalize: bool = True) -> Dict[str, float]:
        """
        Get motif frequency vector for similarity comparison.

        Args:
            normalize: If True, normalize to sum to 1.0

        Returns:
            Dictionary mapping motif type to frequency
        """
        counts = self.count_all()

        if normalize:
            total = sum(counts.values())
            if total > 0:
                return {k: v / total for k, v in counts.items()}

        return {k: float(v) for k, v in counts.items()}

    def detect_hub_spokes(self, min_spokes: int = 3) -> List[MotifMatch]:
        """
        Detect hub-spoke patterns (high-degree central nodes).

        A hub-spoke is a node connected to many peripheral nodes
        that are not well-connected to each other.
        """
        matches = []
        avg_degree = sum(d for _, d in self.G.degree()) / max(1, self.G.number_of_nodes())

        for node in self.G.nodes():
            degree = self.G.degree(node)

            # Hub threshold: significantly higher than average
            if degree >= max(min_spokes, 2 * avg_degree):
                neighbors = list(self.G.neighbors(node))

                # Check that neighbors aren't well-connected to each other
                # (distinguishes hub-spoke from clique)
                neighbor_edges = 0
                for i, n1 in enumerate(neighbors):
                    for n2 in neighbors[i+1:]:
                        if self.G_undirected.has_edge(n1, n2):
                            neighbor_edges += 1

                max_neighbor_edges = len(neighbors) * (len(neighbors) - 1) / 2
                neighbor_density = neighbor_edges / max_neighbor_edges if max_neighbor_edges > 0 else 0

                # Low neighbor connectivity = hub-spoke pattern
                if neighbor_density < 0.3:
                    matches.append(MotifMatch(
                        motif_type="hub_spoke",
                        nodes=tuple([node] + neighbors),
                        edges=[(node, n) for n in neighbors],
                        central_node=node
                    ))

        return matches

    def detect_chains(self, min_length: int = 3) -> List[MotifMatch]:
        """
        Detect chain patterns (linear sequences).

        For directed graphs: a→b→c→d
        For undirected graphs: degree-2 paths
        """
        matches = []

        if self.is_directed:
            # Find nodes that are part of chains (1 in, 1 out)
            chain_nodes = set()
            for node in self.G.nodes():
                in_deg = self.G.in_degree(node)
                out_deg = self.G.out_degree(node)
                if in_deg == 1 and out_deg == 1:
                    chain_nodes.add(node)

            # Trace chains
            visited = set()
            for start in self.G.nodes():
                if start in visited:
                    continue
                if self.G.in_degree(start) == 0 or self.G.in_degree(start) > 1:
                    # Potential chain start
                    chain = [start]
                    current = start
                    while True:
                        successors = list(self.G.successors(current))
                        if len(successors) != 1:
                            break
                        next_node = successors[0]
                        if next_node in chain:  # Avoid cycles
                            break
                        chain.append(next_node)
                        visited.add(next_node)
                        current = next_node
                        if self.G.out_degree(current) != 1:
                            break

                    if len(chain) >= min_length:
                        edges = [(chain[i], chain[i+1]) for i in range(len(chain)-1)]
                        matches.append(MotifMatch(
                            motif_type="chain",
                            nodes=tuple(chain),
                            edges=edges
                        ))
        else:
            # Undirected: find paths through degree-2 nodes
            chain_nodes = [n for n in self.G.nodes() if self.G.degree(n) == 2]
            # Count chains based on degree-2 sequences
            # Simplified: each degree-2 node represents one chain segment
            if len(chain_nodes) >= min_length - 2:
                matches.append(MotifMatch(
                    motif_type="chain",
                    nodes=tuple(chain_nodes[:min_length]),
                    edges=[]
                ))

        return matches

    def detect_triangles(self) -> List[MotifMatch]:
        """
        Detect triangle patterns (3 fully connected nodes).
        """
        matches = []
        triangles = nx.triangles(self.G_undirected)

        # Get actual triangles (not just counts)
        triangle_nodes = set()
        for node in self.G_undirected.nodes():
            if triangles[node] > 0:
                neighbors = list(self.G_undirected.neighbors(node))
                for i, n1 in enumerate(neighbors):
                    for n2 in neighbors[i+1:]:
                        if self.G_undirected.has_edge(n1, n2):
                            tri = tuple(sorted([node, n1, n2]))
                            if tri not in triangle_nodes:
                                triangle_nodes.add(tri)
                                matches.append(MotifMatch(
                                    motif_type="triangle",
                                    nodes=tri,
                                    edges=[(tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])]
                                ))

        return matches

    def detect_bridges(self) -> List[MotifMatch]:
        """
        Detect bridge nodes connecting separate communities.
        """
        matches = []

        try:
            # Find articulation points (nodes whose removal disconnects the graph)
            articulation_points = list(nx.articulation_points(self.G_undirected))

            for node in articulation_points:
                neighbors = list(self.G_undirected.neighbors(node))
                matches.append(MotifMatch(
                    motif_type="bridge",
                    nodes=(node,) + tuple(neighbors[:4]),  # Include some neighbors
                    edges=[(node, n) for n in neighbors[:4]],
                    central_node=node
                ))
        except Exception:
            pass

        return matches

    def detect_cycles(self, max_length: int = 6) -> List[MotifMatch]:
        """
        Detect cycle patterns.
        """
        matches = []

        try:
            cycle_basis = nx.cycle_basis(self.G_undirected)

            for cycle in cycle_basis:
                if 3 <= len(cycle) <= max_length:
                    edges = [(cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))]
                    matches.append(MotifMatch(
                        motif_type="cycle",
                        nodes=tuple(cycle),
                        edges=edges
                    ))
        except Exception:
            pass

        return matches

    def detect_forks(self) -> List[MotifMatch]:
        """
        Detect fork patterns (one node diverging to multiple).
        Only meaningful for directed graphs.
        """
        matches = []

        if not self.is_directed:
            return matches

        for node in self.G.nodes():
            in_deg = self.G.in_degree(node)
            out_deg = self.G.out_degree(node)

            # Fork: few inputs, many outputs
            if in_deg <= 1 and out_deg >= 2:
                successors = list(self.G.successors(node))
                matches.append(MotifMatch(
                    motif_type="fork",
                    nodes=(node,) + tuple(successors),
                    edges=[(node, s) for s in successors],
                    central_node=node
                ))

        return matches

    def detect_funnels(self) -> List[MotifMatch]:
        """
        Detect funnel patterns (multiple nodes converging to one).
        Only meaningful for directed graphs.
        """
        matches = []

        if not self.is_directed:
            return matches

        for node in self.G.nodes():
            in_deg = self.G.in_degree(node)
            out_deg = self.G.out_degree(node)

            # Funnel: many inputs, few outputs
            if in_deg >= 2 and out_deg <= 1:
                predecessors = list(self.G.predecessors(node))
                matches.append(MotifMatch(
                    motif_type="funnel",
                    nodes=tuple(predecessors) + (node,),
                    edges=[(p, node) for p in predecessors],
                    central_node=node
                ))

        return matches

    def get_interpretation(self, motif_type: str, domain: str) -> str:
        """
        Get the domain-specific interpretation of a motif.

        Args:
            motif_type: Type of motif (hub_spoke, chain, etc.)
            domain: Domain (image, music, text)

        Returns:
            Human-readable interpretation
        """
        if motif_type not in self.MOTIF_DEFINITIONS:
            return f"Unknown motif type: {motif_type}"

        interpretations = self.MOTIF_DEFINITIONS[motif_type].get("interpretations", {})
        return interpretations.get(domain, f"No interpretation for {domain}")

    def summary(self) -> str:
        """Get a human-readable summary of detected motifs"""
        counts = self.count_all()
        total = sum(counts.values())

        lines = [
            f"Motif Analysis Summary",
            f"=" * 40,
            f"Total motifs detected: {total}",
            f""
        ]

        for motif_type, count in sorted(counts.items(), key=lambda x: -x[1]):
            if count > 0:
                desc = self.MOTIF_DEFINITIONS.get(motif_type, {}).get("description", "")
                lines.append(f"  {motif_type}: {count} ({desc})")

        return "\n".join(lines)
