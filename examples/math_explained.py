#!/usr/bin/env python3
"""
Mathematical Foundations - Code Examples

This shows the actual mathematics happening in Synesthesia:
1. Graph → Adjacency Matrix
2. Feature Extraction (Signature)
3. Similarity Computation
4. What a true transformation would look like
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import networkx as nx
from scipy import linalg
from collections import Counter


def demo_graph_to_matrix():
    """Show how a graph becomes a matrix."""
    print("=" * 60)
    print("STEP 1: GRAPH → ADJACENCY MATRIX")
    print("=" * 60)

    # Create a simple dragon-like graph
    G = nx.DiGraph()
    G.add_edge("body", "wing_L")
    G.add_edge("body", "wing_R")
    G.add_edge("body", "head")
    G.add_edge("head", "fire")
    G.add_edge("body", "tail")

    print("\nDragon Graph (simplified):")
    print("""
         wing_L    wing_R
            ↖      ↗
              body ──→ tail
               ↓
             head
               ↓
             fire
    """)

    # Get adjacency matrix
    nodes = list(G.nodes())
    n = len(nodes)
    A = nx.adjacency_matrix(G, nodelist=nodes).todense()

    print("Nodes:", nodes)
    print("\nAdjacency Matrix A:")
    print("     ", "  ".join(f"{n[:4]:>4}" for n in nodes))
    for i, row_node in enumerate(nodes):
        row = [int(A[i, j]) for j in range(n)]
        print(f"{row_node[:4]:>5}", row)

    print("\nMatrix properties:")
    print(f"  Shape: {A.shape}")
    print(f"  Non-zero entries: {np.count_nonzero(A)} (= number of edges)")
    print(f"  Symmetric: {np.allclose(A, A.T)} (False = directed graph)")

    return G, A, nodes


def demo_spectral_signature(G, A, nodes):
    """Show how we compute the spectral signature."""
    print("\n" + "=" * 60)
    print("STEP 2: SPECTRAL SIGNATURE (Laplacian Eigenvalues)")
    print("=" * 60)

    # For spectral analysis, use undirected version
    G_undirected = G.to_undirected()
    A_sym = nx.adjacency_matrix(G_undirected, nodelist=nodes).todense()

    # Degree matrix
    degrees = np.array(A_sym.sum(axis=1)).flatten()
    D = np.diag(degrees)

    print("\nDegree Matrix D (diagonal = node degrees):")
    print(f"  Degrees: {dict(zip(nodes, degrees))}")

    # Laplacian: L = D - A
    L = D - A_sym

    print("\nLaplacian Matrix L = D - A:")
    print(np.array(L))

    # Eigenvalues
    eigenvalues = np.sort(np.real(linalg.eigvals(L)))

    print("\nEigenvalues of L (SPECTRAL SIGNATURE):")
    print(f"  λ = {np.round(eigenvalues, 3)}")

    print("\nWhat eigenvalues tell us:")
    print(f"  λ₁ = {eigenvalues[0]:.3f} (always 0 for connected graphs)")
    print(f"  λ₂ = {eigenvalues[1]:.3f} (Fiedler value - algebraic connectivity)")
    print(f"  λ_max = {eigenvalues[-1]:.3f} (related to max degree)")

    return eigenvalues


def demo_motif_vector(G):
    """Show how we compute the motif vector."""
    print("\n" + "=" * 60)
    print("STEP 3: MOTIF VECTOR (Pattern Counts)")
    print("=" * 60)

    # Count simple motifs manually
    n_nodes = G.number_of_nodes()

    # Count out-degree patterns (simplified motif detection)
    out_degrees = dict(G.out_degree())
    in_degrees = dict(G.in_degree())

    # Stars: nodes with out-degree >= 2
    stars = sum(1 for d in out_degrees.values() if d >= 2)

    # Chains: nodes with in=1, out=1
    chains = sum(1 for n in G.nodes()
                 if in_degrees[n] == 1 and out_degrees[n] == 1)

    # Forks: nodes with out-degree >= 2
    forks = sum(1 for d in out_degrees.values() if d >= 2)

    # Leaves: nodes with out-degree = 0
    leaves = sum(1 for d in out_degrees.values() if d == 0)

    # Normalize by number of nodes
    motif_vector = {
        'star': stars / n_nodes,
        'chain': chains / n_nodes,
        'fork': forks / n_nodes,
        'leaf': leaves / n_nodes,
    }

    print("\nMotif counts (raw):")
    print(f"  Stars (out-degree ≥ 2): {stars}")
    print(f"  Chains (in=1, out=1): {chains}")
    print(f"  Forks (branching): {forks}")
    print(f"  Leaves (endpoints): {leaves}")

    print("\nMotif Vector m(G) (normalized by n_nodes):")
    for motif, freq in motif_vector.items():
        print(f"  {motif}: {freq:.3f}")

    return motif_vector


def demo_similarity_computation():
    """Show how we compute similarity between two graphs."""
    print("\n" + "=" * 60)
    print("STEP 4: SIMILARITY COMPUTATION")
    print("=" * 60)

    # Create two graphs to compare
    # Graph 1: Hub-spoke (like dragon body + wings)
    G1 = nx.DiGraph()
    G1.add_edge("hub", "spoke1")
    G1.add_edge("hub", "spoke2")
    G1.add_edge("hub", "spoke3")

    # Graph 2: Chain (like melody)
    G2 = nx.DiGraph()
    G2.add_edge("a", "b")
    G2.add_edge("b", "c")
    G2.add_edge("c", "d")

    # Graph 3: Another hub-spoke (should be similar to G1)
    G3 = nx.DiGraph()
    G3.add_edge("center", "leaf1")
    G3.add_edge("center", "leaf2")
    G3.add_edge("center", "leaf3")
    G3.add_edge("center", "leaf4")

    print("\nThree graphs to compare:")
    print("  G1: Hub-spoke (hub → 3 spokes)")
    print("  G2: Chain (a → b → c → d)")
    print("  G3: Hub-spoke (center → 4 leaves)")

    # Extract simple features
    def extract_features(G):
        n = G.number_of_nodes()
        e = G.number_of_edges()
        out_degs = list(dict(G.out_degree()).values())
        max_out = max(out_degs) if out_degs else 0
        hub_ratio = sum(1 for d in out_degs if d >= 2) / n if n > 0 else 0
        chain_ratio = sum(1 for node in G.nodes()
                         if G.in_degree(node) == 1 and G.out_degree(node) == 1) / n if n > 0 else 0
        return np.array([n, e, max_out, hub_ratio, chain_ratio])

    f1 = extract_features(G1)
    f2 = extract_features(G2)
    f3 = extract_features(G3)

    print("\nFeature vectors [n, e, max_out, hub_ratio, chain_ratio]:")
    print(f"  G1: {f1}")
    print(f"  G2: {f2}")
    print(f"  G3: {f3}")

    # Cosine similarity
    def cosine_sim(a, b):
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0
        return dot / (norm_a * norm_b)

    sim_12 = cosine_sim(f1, f2)
    sim_13 = cosine_sim(f1, f3)
    sim_23 = cosine_sim(f2, f3)

    print("\nCosine Similarity (formula: a·b / ||a|| × ||b||):")
    print(f"  sim(G1, G2) = {sim_12:.3f}  (hub vs chain)")
    print(f"  sim(G1, G3) = {sim_13:.3f}  (hub vs hub) ← HIGHEST!")
    print(f"  sim(G2, G3) = {sim_23:.3f}  (chain vs hub)")

    print("\nConclusion: Hub-spoke structures match each other,")
    print("regardless of labels (body/wing vs center/leaf)")


def demo_what_transformation_would_be():
    """Show what a true A × B = C transformation would look like."""
    print("\n" + "=" * 60)
    print("STEP 5: WHAT TRUE TRANSFORMATION WOULD LOOK LIKE")
    print("=" * 60)

    print("""
CURRENT SYSTEM (Retrieval):
─────────────────────────────
  Dragon Graph ──→ σ(Dragon) ──similarity──→ σ(Music) ──→ Music Graph
                        │                         ↑
                        └── compare ──────────────┘

  We FIND existing structures that match.
  No transformation matrix B is involved.


TRUE TRANSFORMATION (Generation):
─────────────────────────────────
                    ┌─────────────┐
  Dragon Graph ──→  │  Encoder E  │ ──→ z_dragon (latent vector)
                    └─────────────┘          │
                                             ▼
                                    ┌─────────────────┐
                                    │ Transform B     │
                                    │ z_music = B×z   │
                                    └─────────────────┘
                                             │
                                             ▼
                    ┌─────────────┐    z_music (latent vector)
  NEW Music Graph ←─│  Decoder D  │ ←──┘
                    └─────────────┘

  We CREATE new structures via transformation.
  Matrix B is learned from paired examples.
    """)

    # Demonstrate with simple example
    print("\nSimple example of transformation in latent space:")
    print("-" * 50)

    # Pretend we have learned latent representations
    z_dragon = np.array([0.8, 0.2, 0.6, 0.9])  # "hub-like, intense, flowing"
    print(f"z_dragon (latent) = {z_dragon}")
    print("  Interpretation: [hub_strength, chain_strength, flow, intensity]")

    # Transformation matrix (would be learned)
    B = np.array([
        [1.0, 0.0, 0.2, 0.0],   # hub → chorus
        [0.0, 1.0, 0.3, 0.0],   # chain → verse
        [0.2, 0.3, 0.8, 0.0],   # flow → melody
        [0.0, 0.0, 0.0, 1.0],   # intensity → dynamics
    ])
    print(f"\nTransformation matrix B (learned):")
    print(B)
    print("  Maps: [hub, chain, flow, intensity] → [chorus, verse, melody, dynamics]")

    # Transform
    z_music = B @ z_dragon
    print(f"\nz_music = B × z_dragon = {np.round(z_music, 2)}")
    print("  Interpretation: [chorus_strength, verse_strength, melody_flow, dynamics]")

    print("\nThis transformed latent vector would then be decoded")
    print("into an actual music graph structure.")

    print("""
THE KEY DIFFERENCE:
──────────────────
  Current: Find existing C where σ(A) ≈ σ(C)
  True:    Generate new C = Decode(B × Encode(A))

  Current is RETRIEVAL (matching)
  True is GENERATION (creating)
    """)


def demo_full_pipeline():
    """Run the full pipeline with actual Synesthesia code."""
    print("\n" + "=" * 60)
    print("STEP 6: FULL PIPELINE WITH SYNESTHESIA")
    print("=" * 60)

    from structural_rorschach import create_service_container

    container = create_service_container()
    extractor = container.signature_extractor
    similarity_service = container.similarity_service

    # Create dragon and music graphs
    dragon = nx.DiGraph()
    dragon.add_edge("body", "wing_L")
    dragon.add_edge("body", "wing_R")
    dragon.add_edge("body", "head")
    dragon.add_edge("head", "fire1")
    dragon.add_edge("fire1", "fire2")
    dragon.add_edge("body", "tail")

    music = nx.DiGraph()
    music.add_edge("intro", "verse")
    music.add_edge("intro", "chorus")
    music.add_edge("verse", "bridge")
    music.add_edge("chorus", "bridge")
    music.add_edge("bridge", "outro")

    # Extract signatures
    sig_dragon = extractor.extract_from_networkx(dragon, "image", "Dragon", "dragon")
    sig_music = extractor.extract_from_networkx(music, "music", "Song", "song")

    print("\nDragon signature (extracted features):")
    print(f"  Nodes: {sig_dragon.num_nodes}, Edges: {sig_dragon.num_edges}")
    print(f"  Density: {sig_dragon.density:.3f}")
    print(f"  Hub ratio: {sig_dragon.hub_ratio:.3f}")
    print(f"  Motif vector: {dict(list(sig_dragon.motif_vector.items())[:3])}")
    print(f"  Spectral (first 3): {sig_dragon.spectral_signature[:3]}")

    print("\nSong signature:")
    print(f"  Nodes: {sig_music.num_nodes}, Edges: {sig_music.num_edges}")
    print(f"  Density: {sig_music.density:.3f}")
    print(f"  Hub ratio: {sig_music.hub_ratio:.3f}")
    print(f"  Motif vector: {dict(list(sig_music.motif_vector.items())[:3])}")
    print(f"  Spectral (first 3): {sig_music.spectral_signature[:3]}")

    # Compute similarity
    result = similarity_service.compute_similarity(sig_dragon, sig_music)

    print("\nSimilarity computation:")
    print(f"  Motif similarity (cosine):    {result.motif_similarity:.3f}")
    print(f"  Spectral similarity:          {result.spectral_similarity:.3f}")
    print(f"  Scale similarity:             {result.scale_similarity:.3f}")
    print(f"  ─────────────────────────────────────")
    print(f"  OVERALL similarity:           {result.overall_score:.3f}")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   SYNESTHESIA - Mathematical Foundations                             ║
║                                                                      ║
║   Understanding the math behind cross-domain structural matching     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Run all demos
    G, A, nodes = demo_graph_to_matrix()
    eigenvalues = demo_spectral_signature(G, A, nodes)
    motif_vector = demo_motif_vector(G)
    demo_similarity_computation()
    demo_what_transformation_would_be()
    demo_full_pipeline()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Current Synesthesia is a RETRIEVAL system:
  1. Extract features: σ(G) = [motifs, spectral, scale, degree]
  2. Compute similarity: sim(σ_A, σ_B)
  3. Find best match: argmax sim

For true A × B = C transformation, we would need:
  1. Encode to latent: z_A = Encode(A)
  2. Transform: z_B = B × z_A (B is learned)
  3. Decode: C = Decode(z_B)

Both are valid approaches:
  - Retrieval: Fast, no training, finds existing matches
  - Generation: Creates new structures, requires training data
    """)


if __name__ == "__main__":
    main()
