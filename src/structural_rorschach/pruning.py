"""
Graph Pruning - Remove noise and focus on essential structure

Different domains have different types of noise:
- Text: Rare words, stopwords, hapax legomena (words appearing once)
- Image: Background regions, small isolated areas
- Music: Transient notes, noise artifacts

Pruning strategies:
1. Frequency-based: Remove nodes/edges below threshold
2. Degree-based: Remove low-connectivity nodes
3. Weight-based: Remove weak edges
4. Core extraction: Keep only k-core (densely connected subgraph)
5. Community-based: Keep only nodes in significant communities
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import Counter
import json

try:
    import networkx as nx
except ImportError:
    raise ImportError("NetworkX required: pip install networkx")


@dataclass
class PruningStats:
    """Statistics about what was pruned"""
    original_nodes: int
    original_edges: int
    pruned_nodes: int
    pruned_edges: int
    nodes_removed: int
    edges_removed: int
    pruning_ratio: float  # % removed
    strategy: str
    parameters: Dict

    def summary(self) -> str:
        return f"""Pruning Results ({self.strategy}):
  Original: {self.original_nodes} nodes, {self.original_edges} edges
  Pruned:   {self.pruned_nodes} nodes, {self.pruned_edges} edges
  Removed:  {self.nodes_removed} nodes ({100*self.nodes_removed/max(1,self.original_nodes):.1f}%)
            {self.edges_removed} edges ({100*self.edges_removed/max(1,self.original_edges):.1f}%)
  Parameters: {self.parameters}"""


class GraphPruner:
    """
    Prune graphs to remove noise and focus on essential structure.

    Multiple strategies available, can be combined.
    """

    def __init__(self, G: nx.Graph):
        """
        Initialize pruner with a graph.

        Args:
            G: NetworkX graph to prune
        """
        self.original = G.copy()
        self.G = G.copy()
        self.pruning_history: List[PruningStats] = []

    def prune_by_node_frequency(
        self,
        min_frequency: int = 2,
        frequency_attr: str = None
    ) -> "GraphPruner":
        """
        Remove nodes that appear infrequently.

        For text graphs: removes words that appear only once (hapax legomena)
        For image graphs: removes tiny regions

        Args:
            min_frequency: Minimum frequency to keep
            frequency_attr: Node attribute containing frequency (if None, uses degree)

        Returns:
            self for chaining
        """
        original_nodes = self.G.number_of_nodes()
        original_edges = self.G.number_of_edges()

        nodes_to_remove = []
        for node in self.G.nodes():
            if frequency_attr:
                freq = self.G.nodes[node].get(frequency_attr, 0)
            else:
                # Use degree as proxy for frequency
                freq = self.G.degree(node)

            if freq < min_frequency:
                nodes_to_remove.append(node)

        self.G.remove_nodes_from(nodes_to_remove)

        self.pruning_history.append(PruningStats(
            original_nodes=original_nodes,
            original_edges=original_edges,
            pruned_nodes=self.G.number_of_nodes(),
            pruned_edges=self.G.number_of_edges(),
            nodes_removed=len(nodes_to_remove),
            edges_removed=original_edges - self.G.number_of_edges(),
            pruning_ratio=len(nodes_to_remove) / max(1, original_nodes),
            strategy="node_frequency",
            parameters={"min_frequency": min_frequency}
        ))

        return self

    def prune_by_degree(
        self,
        min_degree: int = 1,
        max_degree: int = None
    ) -> "GraphPruner":
        """
        Remove nodes with degree outside specified range.

        Low degree = isolated/peripheral nodes (noise)
        Very high degree = potential artifacts (e.g., "the" in text)

        Args:
            min_degree: Minimum degree to keep
            max_degree: Maximum degree to keep (None = no max)

        Returns:
            self for chaining
        """
        original_nodes = self.G.number_of_nodes()
        original_edges = self.G.number_of_edges()

        nodes_to_remove = []
        for node in self.G.nodes():
            degree = self.G.degree(node)
            if degree < min_degree:
                nodes_to_remove.append(node)
            elif max_degree is not None and degree > max_degree:
                nodes_to_remove.append(node)

        self.G.remove_nodes_from(nodes_to_remove)

        self.pruning_history.append(PruningStats(
            original_nodes=original_nodes,
            original_edges=original_edges,
            pruned_nodes=self.G.number_of_nodes(),
            pruned_edges=self.G.number_of_edges(),
            nodes_removed=len(nodes_to_remove),
            edges_removed=original_edges - self.G.number_of_edges(),
            pruning_ratio=len(nodes_to_remove) / max(1, original_nodes),
            strategy="degree",
            parameters={"min_degree": min_degree, "max_degree": max_degree}
        ))

        return self

    def prune_by_edge_weight(
        self,
        min_weight: float = 0.01,
        weight_attr: str = "weight"
    ) -> "GraphPruner":
        """
        Remove edges with weight below threshold.

        For text: removes rare word transitions
        For images: removes weak adjacencies (small boundaries)

        Args:
            min_weight: Minimum edge weight to keep
            weight_attr: Edge attribute containing weight

        Returns:
            self for chaining
        """
        original_nodes = self.G.number_of_nodes()
        original_edges = self.G.number_of_edges()

        edges_to_remove = []
        for u, v, data in self.G.edges(data=True):
            weight = data.get(weight_attr, 1.0)
            if weight < min_weight:
                edges_to_remove.append((u, v))

        self.G.remove_edges_from(edges_to_remove)

        # Remove isolated nodes created by edge removal
        isolated = list(nx.isolates(self.G))
        self.G.remove_nodes_from(isolated)

        self.pruning_history.append(PruningStats(
            original_nodes=original_nodes,
            original_edges=original_edges,
            pruned_nodes=self.G.number_of_nodes(),
            pruned_edges=self.G.number_of_edges(),
            nodes_removed=len(isolated),
            edges_removed=len(edges_to_remove),
            pruning_ratio=len(edges_to_remove) / max(1, original_edges),
            strategy="edge_weight",
            parameters={"min_weight": min_weight}
        ))

        return self

    def prune_to_k_core(self, k: int = 2) -> "GraphPruner":
        """
        Extract k-core: maximal subgraph where all nodes have degree >= k.

        This is a powerful way to find the "dense backbone" of a graph,
        removing peripheral noise while preserving core structure.

        Args:
            k: Core number (minimum degree in subgraph)

        Returns:
            self for chaining
        """
        original_nodes = self.G.number_of_nodes()
        original_edges = self.G.number_of_edges()

        # Remove self-loops first (k-core doesn't support them)
        self.G.remove_edges_from(nx.selfloop_edges(self.G))

        # Convert to undirected for k-core (standard definition)
        if self.G.is_directed():
            G_undirected = self.G.to_undirected()
            G_undirected.remove_edges_from(nx.selfloop_edges(G_undirected))
            core_nodes = set(nx.k_core(G_undirected, k=k).nodes())
            # Keep only core nodes in directed graph
            nodes_to_remove = [n for n in self.G.nodes() if n not in core_nodes]
            self.G.remove_nodes_from(nodes_to_remove)
        else:
            self.G.remove_edges_from(nx.selfloop_edges(self.G))
            self.G = nx.k_core(self.G, k=k)

        self.pruning_history.append(PruningStats(
            original_nodes=original_nodes,
            original_edges=original_edges,
            pruned_nodes=self.G.number_of_nodes(),
            pruned_edges=self.G.number_of_edges(),
            nodes_removed=original_nodes - self.G.number_of_nodes(),
            edges_removed=original_edges - self.G.number_of_edges(),
            pruning_ratio=(original_nodes - self.G.number_of_nodes()) / max(1, original_nodes),
            strategy="k_core",
            parameters={"k": k}
        ))

        return self

    def prune_to_largest_component(self) -> "GraphPruner":
        """
        Keep only the largest connected component.

        Removes isolated subgraphs that are disconnected from the main structure.

        Returns:
            self for chaining
        """
        original_nodes = self.G.number_of_nodes()
        original_edges = self.G.number_of_edges()

        if self.G.is_directed():
            # Use weakly connected components
            components = list(nx.weakly_connected_components(self.G))
        else:
            components = list(nx.connected_components(self.G))

        if components:
            largest = max(components, key=len)
            self.G = self.G.subgraph(largest).copy()

        self.pruning_history.append(PruningStats(
            original_nodes=original_nodes,
            original_edges=original_edges,
            pruned_nodes=self.G.number_of_nodes(),
            pruned_edges=self.G.number_of_edges(),
            nodes_removed=original_nodes - self.G.number_of_nodes(),
            edges_removed=original_edges - self.G.number_of_edges(),
            pruning_ratio=(original_nodes - self.G.number_of_nodes()) / max(1, original_nodes),
            strategy="largest_component",
            parameters={}
        ))

        return self

    def prune_by_percentile(
        self,
        keep_top_percent: float = 80.0,
        metric: str = "degree"
    ) -> "GraphPruner":
        """
        Keep only top percentile of nodes by specified metric.

        Args:
            keep_top_percent: Percentage of nodes to keep (0-100)
            metric: "degree", "betweenness", "pagerank"

        Returns:
            self for chaining
        """
        original_nodes = self.G.number_of_nodes()
        original_edges = self.G.number_of_edges()

        # Compute metric for all nodes
        if metric == "degree":
            scores = dict(self.G.degree())
        elif metric == "betweenness":
            scores = nx.betweenness_centrality(self.G)
        elif metric == "pagerank":
            scores = nx.pagerank(self.G)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Find threshold
        import numpy as np
        values = list(scores.values())
        threshold = np.percentile(values, 100 - keep_top_percent)

        # Remove nodes below threshold
        nodes_to_remove = [n for n, v in scores.items() if v < threshold]
        self.G.remove_nodes_from(nodes_to_remove)

        self.pruning_history.append(PruningStats(
            original_nodes=original_nodes,
            original_edges=original_edges,
            pruned_nodes=self.G.number_of_nodes(),
            pruned_edges=self.G.number_of_edges(),
            nodes_removed=len(nodes_to_remove),
            edges_removed=original_edges - self.G.number_of_edges(),
            pruning_ratio=len(nodes_to_remove) / max(1, original_nodes),
            strategy="percentile",
            parameters={"keep_top_percent": keep_top_percent, "metric": metric}
        ))

        return self

    def prune_stopwords(
        self,
        stopwords: Set[str] = None,
        name_attr: str = "name"
    ) -> "GraphPruner":
        """
        Remove common/stopword nodes (for text graphs).

        Default stopwords include common English words that don't
        contribute to structural meaning.

        Args:
            stopwords: Set of node names to remove (uses default if None)
            name_attr: Node attribute containing the name

        Returns:
            self for chaining
        """
        if stopwords is None:
            # Default English stopwords relevant to biblical text
            stopwords = {
                "the", "and", "of", "to", "in", "a", "that", "is", "was",
                "he", "for", "it", "with", "as", "his", "on", "be", "at",
                "by", "i", "this", "had", "not", "are", "but", "from",
                "or", "have", "an", "they", "which", "one", "you", "were",
                "her", "all", "she", "there", "would", "their", "we",
                "him", "been", "has", "when", "who", "will", "no", "more",
                "if", "out", "so", "said", "what", "up", "its", "about",
                "into", "than", "them", "can", "only", "other", "new",
                "some", "could", "time", "these", "two", "may", "then",
                "do", "first", "any", "my", "now", "such", "like", "our",
                "over", "man", "me", "even", "most", "made", "after",
                "also", "did", "many", "before", "must", "through", "back",
                "years", "where", "much", "your", "way", "well", "down",
                "should", "because", "each", "just", "those", "people",
                "how", "too", "little", "state", "good", "very", "make",
                "world", "still", "own", "see", "men", "work", "long",
                "get", "here", "between", "both", "life", "being", "under",
                "never", "day", "same", "another", "know", "while", "last",
                "might", "us", "great", "old", "year", "off", "come",
                "since", "against", "go", "came", "right", "used", "take",
                "unto", "thou", "thee", "thy", "ye", "hath", "shall",
                "upon", "saith", "therefore", "therefore", "thus"
            }

        original_nodes = self.G.number_of_nodes()
        original_edges = self.G.number_of_edges()

        nodes_to_remove = []
        for node in self.G.nodes():
            # Get node name
            if name_attr and name_attr in self.G.nodes[node]:
                name = self.G.nodes[node][name_attr]
            else:
                name = str(node)

            # Handle prefixed names (e.g., "word_the" -> "the")
            if name.startswith("word_"):
                name = name[5:]

            if name.lower() in stopwords:
                nodes_to_remove.append(node)

        self.G.remove_nodes_from(nodes_to_remove)

        self.pruning_history.append(PruningStats(
            original_nodes=original_nodes,
            original_edges=original_edges,
            pruned_nodes=self.G.number_of_nodes(),
            pruned_edges=self.G.number_of_edges(),
            nodes_removed=len(nodes_to_remove),
            edges_removed=original_edges - self.G.number_of_edges(),
            pruning_ratio=len(nodes_to_remove) / max(1, original_nodes),
            strategy="stopwords",
            parameters={"num_stopwords": len(stopwords)}
        ))

        return self

    def get_pruned_graph(self) -> nx.Graph:
        """Get the pruned graph"""
        return self.G

    def get_stats(self) -> List[PruningStats]:
        """Get pruning history"""
        return self.pruning_history

    def total_reduction(self) -> Tuple[float, float]:
        """Get total node and edge reduction ratios"""
        original_nodes = self.original.number_of_nodes()
        original_edges = self.original.number_of_edges()
        final_nodes = self.G.number_of_nodes()
        final_edges = self.G.number_of_edges()

        node_reduction = 1 - (final_nodes / max(1, original_nodes))
        edge_reduction = 1 - (final_edges / max(1, original_edges))

        return node_reduction, edge_reduction

    def summary(self) -> str:
        """Get summary of all pruning operations"""
        lines = ["=" * 50, "PRUNING SUMMARY", "=" * 50]

        for i, stats in enumerate(self.pruning_history, 1):
            lines.append(f"\nStep {i}: {stats.strategy}")
            lines.append(stats.summary())

        node_red, edge_red = self.total_reduction()
        lines.append(f"\n{'=' * 50}")
        lines.append(f"TOTAL REDUCTION")
        lines.append(f"  Nodes: {self.original.number_of_nodes()} -> {self.G.number_of_nodes()} ({100*node_red:.1f}% removed)")
        lines.append(f"  Edges: {self.original.number_of_edges()} -> {self.G.number_of_edges()} ({100*edge_red:.1f}% removed)")

        return "\n".join(lines)


def auto_prune(
    G: nx.Graph,
    domain: str = "text",
    aggression: str = "medium"
) -> Tuple[nx.Graph, str]:
    """
    Automatically prune a graph based on domain and aggression level.

    Args:
        G: Graph to prune
        domain: "text", "image", or "music"
        aggression: "light", "medium", or "aggressive"

    Returns:
        Tuple of (pruned graph, summary string)
    """
    pruner = GraphPruner(G)

    if domain == "text":
        if aggression == "light":
            pruner.prune_by_degree(min_degree=1)
            pruner.prune_to_largest_component()
        elif aggression == "medium":
            pruner.prune_stopwords()
            pruner.prune_by_degree(min_degree=2)
            pruner.prune_to_largest_component()
        else:  # aggressive
            pruner.prune_stopwords()
            pruner.prune_by_edge_weight(min_weight=0.01)
            pruner.prune_to_k_core(k=2)
            pruner.prune_to_largest_component()

    elif domain == "image":
        if aggression == "light":
            pruner.prune_by_edge_weight(min_weight=0.01)
        elif aggression == "medium":
            pruner.prune_by_edge_weight(min_weight=0.02)
            pruner.prune_by_percentile(keep_top_percent=80, metric="degree")
        else:  # aggressive
            pruner.prune_by_edge_weight(min_weight=0.05)
            pruner.prune_to_k_core(k=2)
            pruner.prune_to_largest_component()

    elif domain == "music":
        if aggression == "light":
            pruner.prune_by_degree(min_degree=1)
        elif aggression == "medium":
            pruner.prune_by_edge_weight(min_weight=0.01)
            pruner.prune_by_degree(min_degree=2)
        else:  # aggressive
            pruner.prune_by_edge_weight(min_weight=0.02)
            pruner.prune_to_k_core(k=2)

    return pruner.get_pruned_graph(), pruner.summary()
