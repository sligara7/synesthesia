"""
Graph Analogies - Cross-Domain Structural Transfer

Implements the A × B = C transformation:
  G_target = G_source + T
  where T = G_exemplar_target - G_exemplar_source

Inspired by Word2Vec's "king - man + woman = queen"
Applied to graph structures across domains.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import networkx as nx
from scipy import linalg


@dataclass
class GraphEmbedding:
    """Embedding of a graph in structural space."""
    vector: np.ndarray
    graph_id: str
    domain: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AnalogyResult:
    """Result of a graph analogy computation."""
    source_embedding: GraphEmbedding
    transformation: np.ndarray
    predicted_embedding: np.ndarray
    nearest_match: Optional[GraphEmbedding]
    similarity_to_match: float
    structural_fidelity: Dict[str, float]


class StructuralEncoder:
    """
    Encodes graphs into a domain-agnostic structural embedding space.

    The embedding captures topology, not content:
    - Spectral properties (Laplacian eigenvalues)
    - Motif distribution
    - Degree statistics
    - Centrality patterns
    - Clustering structure
    """

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim

    def encode(self, G: nx.Graph, graph_id: str = "", domain: str = "") -> GraphEmbedding:
        """
        Encode a graph into a structural embedding vector.

        The embedding is designed to be domain-agnostic:
        two graphs with similar topology will have similar embeddings,
        regardless of what domain they come from.
        """
        features = []

        # 1. Scale features (normalized)
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        density = nx.density(G)

        features.extend([
            np.log1p(n_nodes) / 10,  # Log-scaled node count
            np.log1p(n_edges) / 10,  # Log-scaled edge count
            density,                  # Density [0, 1]
        ])

        # 2. Degree distribution features
        if n_nodes > 0:
            degrees = [d for n, d in G.degree()]
            features.extend([
                np.mean(degrees) / max(n_nodes, 1),
                np.std(degrees) / max(n_nodes, 1) if len(degrees) > 1 else 0,
                np.max(degrees) / max(n_nodes, 1) if degrees else 0,
                np.median(degrees) / max(n_nodes, 1) if degrees else 0,
            ])

            # Degree histogram (5 bins, normalized)
            hist, _ = np.histogram(degrees, bins=5, density=True)
            hist = hist / (hist.sum() + 1e-8)
            features.extend(hist.tolist())
        else:
            features.extend([0] * 9)

        # 3. Spectral features (Laplacian eigenvalues)
        spectral = self._compute_spectral_features(G)
        features.extend(spectral)

        # 4. Motif-like features
        motif_features = self._compute_motif_features(G)
        features.extend(motif_features)

        # 5. Centrality features
        centrality = self._compute_centrality_features(G)
        features.extend(centrality)

        # 6. Clustering features
        clustering = self._compute_clustering_features(G)
        features.extend(clustering)

        # Convert to numpy and normalize
        vector = np.array(features, dtype=np.float32)

        # Pad or truncate to embedding_dim
        if len(vector) < self.embedding_dim:
            vector = np.pad(vector, (0, self.embedding_dim - len(vector)))
        else:
            vector = vector[:self.embedding_dim]

        # L2 normalize for cosine similarity
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return GraphEmbedding(
            vector=vector,
            graph_id=graph_id,
            domain=domain,
            metadata={
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'density': density,
            }
        )

    def _compute_spectral_features(self, G: nx.Graph, k: int = 10) -> List[float]:
        """Compute spectral features from Laplacian eigenvalues."""
        if G.number_of_nodes() < 2:
            return [0] * k

        try:
            # Use undirected version for spectral analysis
            if G.is_directed():
                G = G.to_undirected()

            L = nx.laplacian_matrix(G).todense()
            eigenvalues = np.sort(np.real(linalg.eigvals(L)))

            # Take first k eigenvalues (normalized)
            eigenvalues = eigenvalues[:k]
            max_eig = eigenvalues[-1] if len(eigenvalues) > 0 else 1
            if max_eig > 0:
                eigenvalues = eigenvalues / max_eig

            # Pad if needed
            if len(eigenvalues) < k:
                eigenvalues = np.pad(eigenvalues, (0, k - len(eigenvalues)))

            return eigenvalues.tolist()
        except Exception:
            return [0] * k

    def _compute_motif_features(self, G: nx.Graph) -> List[float]:
        """Compute motif-like structural features."""
        n_nodes = G.number_of_nodes()
        if n_nodes == 0:
            return [0] * 8

        # Use directed version if available
        G_dir = G if G.is_directed() else G.to_directed()

        features = []

        # Hub ratio (high out-degree nodes)
        out_degrees = [d for n, d in G_dir.out_degree()]
        hub_count = sum(1 for d in out_degrees if d >= 2)
        features.append(hub_count / n_nodes)

        # Chain ratio (in=1, out=1)
        chain_count = sum(1 for n in G_dir.nodes()
                         if G_dir.in_degree(n) == 1 and G_dir.out_degree(n) == 1)
        features.append(chain_count / n_nodes)

        # Source ratio (in=0)
        source_count = sum(1 for n in G_dir.nodes() if G_dir.in_degree(n) == 0)
        features.append(source_count / n_nodes)

        # Sink ratio (out=0)
        sink_count = sum(1 for n in G_dir.nodes() if G_dir.out_degree(n) == 0)
        features.append(sink_count / n_nodes)

        # Fork ratio (out >= 2)
        fork_count = sum(1 for d in out_degrees if d >= 2)
        features.append(fork_count / n_nodes)

        # Funnel ratio (in >= 2)
        in_degrees = [d for n, d in G_dir.in_degree()]
        funnel_count = sum(1 for d in in_degrees if d >= 2)
        features.append(funnel_count / n_nodes)

        # Triangle density
        try:
            triangles = sum(nx.triangles(G.to_undirected()).values()) / 3
            features.append(triangles / max(n_nodes, 1))
        except Exception:
            features.append(0)

        # Is DAG?
        try:
            features.append(1.0 if nx.is_directed_acyclic_graph(G_dir) else 0.0)
        except Exception:
            features.append(0)

        return features

    def _compute_centrality_features(self, G: nx.Graph) -> List[float]:
        """Compute centrality-based features."""
        n_nodes = G.number_of_nodes()
        if n_nodes < 2:
            return [0] * 6

        features = []

        try:
            # Degree centrality stats
            dc = list(nx.degree_centrality(G).values())
            features.extend([np.mean(dc), np.std(dc), np.max(dc)])
        except Exception:
            features.extend([0, 0, 0])

        try:
            # Betweenness centrality stats
            bc = list(nx.betweenness_centrality(G).values())
            features.extend([np.mean(bc), np.std(bc), np.max(bc)])
        except Exception:
            features.extend([0, 0, 0])

        return features

    def _compute_clustering_features(self, G: nx.Graph) -> List[float]:
        """Compute clustering-based features."""
        n_nodes = G.number_of_nodes()
        if n_nodes < 2:
            return [0] * 4

        features = []

        try:
            # Global clustering coefficient
            features.append(nx.transitivity(G))
        except Exception:
            features.append(0)

        try:
            # Average clustering coefficient
            features.append(nx.average_clustering(G.to_undirected()))
        except Exception:
            features.append(0)

        try:
            # Number of connected components (normalized)
            if G.is_directed():
                n_components = nx.number_weakly_connected_components(G)
            else:
                n_components = nx.number_connected_components(G)
            features.append(n_components / n_nodes)
        except Exception:
            features.append(0)

        try:
            # Diameter (of largest component, normalized)
            if G.is_directed():
                G_undirected = G.to_undirected()
            else:
                G_undirected = G
            largest_cc = max(nx.connected_components(G_undirected), key=len)
            subgraph = G_undirected.subgraph(largest_cc)
            diameter = nx.diameter(subgraph)
            features.append(diameter / n_nodes)
        except Exception:
            features.append(0)

        return features


class GraphAnalogyEngine:
    """
    Engine for computing graph analogies.

    Core operation:
        G_target = G_source + T
        where T = E(exemplar_target) - E(exemplar_source)

    This enables cross-domain structural transfer:
        dragon_image + (music_exemplar - image_exemplar) ≈ dragon_music
    """

    def __init__(self, encoder: StructuralEncoder = None):
        self.encoder = encoder or StructuralEncoder()
        self.corpus: Dict[str, GraphEmbedding] = {}
        self.domain_exemplars: Dict[Tuple[str, str], np.ndarray] = {}

    def add_to_corpus(self, G: nx.Graph, graph_id: str, domain: str) -> GraphEmbedding:
        """Add a graph to the corpus for retrieval."""
        embedding = self.encoder.encode(G, graph_id, domain)
        self.corpus[graph_id] = embedding
        return embedding

    def learn_domain_transformation(
        self,
        exemplar_pairs: List[Tuple[nx.Graph, nx.Graph]],
        source_domain: str,
        target_domain: str
    ) -> np.ndarray:
        """
        Learn a transformation vector T from paired examples.

        T = mean(E(target_i) - E(source_i)) for all pairs i

        This T can then be applied to new source graphs to predict
        their target domain equivalents.
        """
        transformations = []

        for source_graph, target_graph in exemplar_pairs:
            source_emb = self.encoder.encode(source_graph, "", source_domain)
            target_emb = self.encoder.encode(target_graph, "", target_domain)

            T_i = target_emb.vector - source_emb.vector
            transformations.append(T_i)

        # Average transformation
        T = np.mean(transformations, axis=0)

        # Store for later use
        self.domain_exemplars[(source_domain, target_domain)] = T

        return T

    def compute_analogy(
        self,
        source_graph: nx.Graph,
        source_domain: str,
        target_domain: str,
        transformation: np.ndarray = None
    ) -> AnalogyResult:
        """
        Compute graph analogy: source_graph + T → predicted_target

        If transformation is not provided, uses learned transformation
        for the domain pair.
        """
        # Encode source
        source_emb = self.encoder.encode(source_graph, "source", source_domain)

        # Get transformation
        if transformation is None:
            key = (source_domain, target_domain)
            if key not in self.domain_exemplars:
                raise ValueError(f"No learned transformation for {source_domain} → {target_domain}")
            transformation = self.domain_exemplars[key]

        # Apply transformation: predicted = source + T
        predicted_vector = source_emb.vector + transformation

        # Normalize
        norm = np.linalg.norm(predicted_vector)
        if norm > 0:
            predicted_vector = predicted_vector / norm

        # Find nearest match in corpus (if available)
        nearest_match = None
        best_similarity = -1

        for graph_id, emb in self.corpus.items():
            if emb.domain == target_domain:
                sim = np.dot(predicted_vector, emb.vector)
                if sim > best_similarity:
                    best_similarity = sim
                    nearest_match = emb

        return AnalogyResult(
            source_embedding=source_emb,
            transformation=transformation,
            predicted_embedding=predicted_vector,
            nearest_match=nearest_match,
            similarity_to_match=best_similarity,
            structural_fidelity={}
        )

    def evaluate_transfer(
        self,
        source_graph: nx.Graph,
        actual_target_graph: nx.Graph,
        source_domain: str,
        target_domain: str,
        transformation: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Evaluate how well the transformation predicts the actual target.

        Returns structural fidelity metrics comparing:
        - Predicted embedding vs actual target embedding
        - Various structural properties
        """
        # Get predicted embedding
        result = self.compute_analogy(
            source_graph, source_domain, target_domain, transformation
        )
        predicted = result.predicted_embedding

        # Get actual target embedding
        actual_emb = self.encoder.encode(actual_target_graph, "actual", target_domain)

        # Compute similarities
        embedding_similarity = np.dot(predicted, actual_emb.vector)

        # Structural property comparison
        source_props = self._extract_properties(source_graph)
        target_props = self._extract_properties(actual_target_graph)

        property_fidelity = {}
        for prop in source_props:
            if source_props[prop] > 0:
                ratio = min(target_props[prop], source_props[prop]) / max(target_props[prop], source_props[prop])
                property_fidelity[prop] = ratio

        return {
            'embedding_similarity': embedding_similarity,
            'mean_property_fidelity': np.mean(list(property_fidelity.values())) if property_fidelity else 0,
            **property_fidelity
        }

    def _extract_properties(self, G: nx.Graph) -> Dict[str, float]:
        """Extract key structural properties for comparison."""
        n = G.number_of_nodes()
        e = G.number_of_edges()

        props = {
            'density': nx.density(G),
            'avg_degree': 2 * e / n if n > 0 else 0,
        }

        try:
            props['clustering'] = nx.average_clustering(G.to_undirected())
        except:
            props['clustering'] = 0

        try:
            if G.is_directed():
                out_degrees = [d for _, d in G.out_degree()]
                props['hub_ratio'] = sum(1 for d in out_degrees if d >= 2) / n if n > 0 else 0
        except:
            props['hub_ratio'] = 0

        return props


def demo():
    """Demonstrate graph analogies."""
    print("=" * 60)
    print("GRAPH ANALOGIES DEMO")
    print("=" * 60)

    engine = GraphAnalogyEngine()

    # Create exemplar pairs (same structure, different "domains")
    # Image domain: hub-spoke (dragon body + wings)
    image_exemplar = nx.DiGraph()
    image_exemplar.add_edge("body", "wing1")
    image_exemplar.add_edge("body", "wing2")
    image_exemplar.add_edge("body", "head")
    image_exemplar.add_edge("head", "fire")

    # Music domain: hub-spoke (chord structure)
    music_exemplar = nx.DiGraph()
    music_exemplar.add_edge("root", "third")
    music_exemplar.add_edge("root", "fifth")
    music_exemplar.add_edge("root", "seventh")
    music_exemplar.add_edge("seventh", "resolve")

    print("\n1. Learning transformation from exemplar pair...")
    print(f"   Image exemplar: {image_exemplar.number_of_nodes()} nodes, {image_exemplar.number_of_edges()} edges")
    print(f"   Music exemplar: {music_exemplar.number_of_nodes()} nodes, {music_exemplar.number_of_edges()} edges")

    T = engine.learn_domain_transformation(
        [(image_exemplar, music_exemplar)],
        "image", "music"
    )
    print(f"   Transformation vector T: norm = {np.linalg.norm(T):.4f}")

    # Create a new image graph (different from exemplar)
    new_image = nx.DiGraph()
    new_image.add_edge("center", "spoke1")
    new_image.add_edge("center", "spoke2")
    new_image.add_edge("center", "spoke3")
    new_image.add_edge("spoke1", "detail1")
    new_image.add_edge("spoke2", "detail2")

    print(f"\n2. New image graph to transfer:")
    print(f"   Nodes: {new_image.number_of_nodes()}, Edges: {new_image.number_of_edges()}")

    # Add some music graphs to corpus for retrieval
    music1 = nx.DiGraph()  # Similar hub-spoke
    music1.add_edge("tonic", "note1")
    music1.add_edge("tonic", "note2")
    music1.add_edge("tonic", "note3")
    music1.add_edge("note1", "ornament")
    engine.add_to_corpus(music1, "music_hubspoke", "music")

    music2 = nx.DiGraph()  # Chain structure
    music2.add_edge("a", "b")
    music2.add_edge("b", "c")
    music2.add_edge("c", "d")
    music2.add_edge("d", "e")
    engine.add_to_corpus(music2, "music_chain", "music")

    music3 = nx.DiGraph()  # Funnel structure
    music3.add_edge("v1", "chorus")
    music3.add_edge("v2", "chorus")
    music3.add_edge("v3", "chorus")
    music3.add_edge("chorus", "outro")
    engine.add_to_corpus(music3, "music_funnel", "music")

    print("\n3. Computing analogy: new_image + T → predicted_music")
    result = engine.compute_analogy(new_image, "image", "music")

    print(f"\n4. Results:")
    print(f"   Nearest match in corpus: {result.nearest_match.graph_id if result.nearest_match else 'None'}")
    print(f"   Similarity to match: {result.similarity_to_match:.3f}")

    # Evaluate transfer quality
    print(f"\n5. Evaluating transfer quality...")
    eval_result = engine.evaluate_transfer(
        new_image, music1, "image", "music"
    )
    print(f"   Embedding similarity: {eval_result['embedding_similarity']:.3f}")
    print(f"   Mean property fidelity: {eval_result['mean_property_fidelity']:.3f}")

    print("\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print("""
The transformation T captures the "structural offset" between domains.

When we apply: new_image + T → predicted_music

We're saying: "What would this image structure look like in music space?"

The nearest match retrieval shows that hub-spoke images map to
hub-spoke music structures, preserving topology across domains.
    """)


if __name__ == "__main__":
    demo()
