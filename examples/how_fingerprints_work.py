#!/usr/bin/env python3
"""
HOW FINGERPRINTS AND RETRIEVAL WORK

This shows the exact mechanics:
1. How we create a fingerprint from a graph
2. How we compare fingerprints
3. How retrieval finds matches
"""

import numpy as np
import networkx as nx
from scipy import linalg
from collections import Counter

# =============================================================================
# STEP 1: THE FINGERPRINT (What goes into it)
# =============================================================================

def create_fingerprint(G, name="graph"):
    """
    Create a structural fingerprint from a graph.

    The fingerprint is a VECTOR of numbers that captures topology.
    Two graphs with similar topology will have similar vectors.
    """
    print(f"\n{'='*60}")
    print(f"CREATING FINGERPRINT FOR: {name}")
    print(f"{'='*60}")

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    print(f"\nInput graph: {n_nodes} nodes, {n_edges} edges")
    print(f"Edges: {list(G.edges())}")

    fingerprint = {}

    # ─────────────────────────────────────────────────────────────
    # COMPONENT 1: Scale Features
    # ─────────────────────────────────────────────────────────────
    print(f"\n[1] SCALE FEATURES")

    density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0

    fingerprint['n_nodes'] = n_nodes
    fingerprint['n_edges'] = n_edges
    fingerprint['density'] = density

    print(f"    n_nodes = {n_nodes}")
    print(f"    n_edges = {n_edges}")
    print(f"    density = {n_edges} / ({n_nodes} × {n_nodes-1}) = {density:.4f}")

    # ─────────────────────────────────────────────────────────────
    # COMPONENT 2: Degree Distribution
    # ─────────────────────────────────────────────────────────────
    print(f"\n[2] DEGREE DISTRIBUTION")

    out_degrees = [d for n, d in G.out_degree()]
    in_degrees = [d for n, d in G.in_degree()]

    avg_degree = np.mean(out_degrees) if out_degrees else 0
    max_degree = max(out_degrees) if out_degrees else 0

    # Hub ratio: what fraction of nodes have high out-degree?
    hub_threshold = 2
    hubs = [n for n, d in G.out_degree() if d >= hub_threshold]
    hub_ratio = len(hubs) / n_nodes if n_nodes > 0 else 0

    fingerprint['avg_degree'] = avg_degree
    fingerprint['max_degree'] = max_degree
    fingerprint['hub_ratio'] = hub_ratio

    print(f"    out_degrees = {out_degrees}")
    print(f"    avg_degree = {avg_degree:.2f}")
    print(f"    max_degree = {max_degree}")
    print(f"    hubs (degree ≥ {hub_threshold}) = {hubs}")
    print(f"    hub_ratio = {len(hubs)}/{n_nodes} = {hub_ratio:.3f}")

    # ─────────────────────────────────────────────────────────────
    # COMPONENT 3: Motif Counts (Pattern Detection)
    # ─────────────────────────────────────────────────────────────
    print(f"\n[3] MOTIF VECTOR (Pattern Counts)")

    # Count structural patterns
    motifs = {
        'stars': 0,      # Nodes with out-degree >= 2 (hub pattern)
        'chains': 0,     # Nodes with in=1, out=1 (linear flow)
        'sources': 0,    # Nodes with in=0 (entry points)
        'sinks': 0,      # Nodes with out=0 (endpoints)
        'forks': 0,      # Nodes that branch (out >= 2)
        'funnels': 0,    # Nodes that merge (in >= 2)
    }

    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)

        if out_deg >= 2:
            motifs['stars'] += 1
            motifs['forks'] += 1
        if in_deg == 1 and out_deg == 1:
            motifs['chains'] += 1
        if in_deg == 0:
            motifs['sources'] += 1
        if out_deg == 0:
            motifs['sinks'] += 1
        if in_deg >= 2:
            motifs['funnels'] += 1

    # Normalize by number of nodes
    motif_vector = {k: v / n_nodes for k, v in motifs.items()}
    fingerprint['motif_vector'] = motif_vector

    print(f"    Raw counts: {motifs}")
    print(f"    Normalized (÷ {n_nodes}):")
    for k, v in motif_vector.items():
        print(f"      {k}: {v:.3f}")

    # ─────────────────────────────────────────────────────────────
    # COMPONENT 4: Spectral Signature (Eigenvalues)
    # ─────────────────────────────────────────────────────────────
    print(f"\n[4] SPECTRAL SIGNATURE (Laplacian Eigenvalues)")

    # Use undirected version for spectral analysis
    G_undirected = G.to_undirected()

    # Build Laplacian: L = D - A
    nodes = list(G_undirected.nodes())
    A = nx.adjacency_matrix(G_undirected, nodelist=nodes).todense()
    degrees = np.array(A.sum(axis=1)).flatten()
    D = np.diag(degrees)
    L = D - A

    # Get eigenvalues
    eigenvalues = np.sort(np.real(linalg.eigvals(L)))

    # Take first k eigenvalues as signature
    k = min(5, len(eigenvalues))
    spectral_signature = list(eigenvalues[:k])
    fingerprint['spectral'] = spectral_signature

    print(f"    Laplacian L = D - A")
    print(f"    Eigenvalues: {np.round(eigenvalues, 3)}")
    print(f"    Spectral signature (first {k}): {np.round(spectral_signature, 3)}")
    print(f"    λ₂ (connectivity): {eigenvalues[1]:.3f}" if len(eigenvalues) > 1 else "")

    # ─────────────────────────────────────────────────────────────
    # FINAL FINGERPRINT VECTOR
    # ─────────────────────────────────────────────────────────────
    print(f"\n[5] FINAL FINGERPRINT VECTOR")

    # Combine into a single vector for comparison
    vector = [
        fingerprint['density'],
        fingerprint['avg_degree'],
        fingerprint['hub_ratio'],
        motif_vector['stars'],
        motif_vector['chains'],
        motif_vector['forks'],
        motif_vector['sinks'],
    ] + spectral_signature

    fingerprint['vector'] = np.array(vector)

    print(f"    Vector components:")
    print(f"      [density, avg_deg, hub_ratio, stars, chains, forks, sinks, λ₁, λ₂, ...]")
    print(f"    Vector: {np.round(fingerprint['vector'], 3)}")

    return fingerprint


# =============================================================================
# STEP 2: COMPARING FINGERPRINTS (Similarity)
# =============================================================================

def compare_fingerprints(fp1, fp2, name1="A", name2="B"):
    """
    Compare two fingerprints using cosine similarity.

    Cosine similarity measures the angle between two vectors:
    - 1.0 = identical direction (same structure)
    - 0.0 = perpendicular (unrelated)
    - -1.0 = opposite (inverse structure)
    """
    print(f"\n{'='*60}")
    print(f"COMPARING FINGERPRINTS: {name1} vs {name2}")
    print(f"{'='*60}")

    v1 = fp1['vector']
    v2 = fp2['vector']

    # Pad to same length if needed
    max_len = max(len(v1), len(v2))
    v1 = np.pad(v1, (0, max_len - len(v1)))
    v2 = np.pad(v2, (0, max_len - len(v2)))

    print(f"\nVector {name1}: {np.round(v1, 3)}")
    print(f"Vector {name2}: {np.round(v2, 3)}")

    # Cosine similarity formula
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    print(f"\nCosine Similarity Calculation:")
    print(f"  dot(v1, v2) = {dot_product:.4f}")
    print(f"  ||v1|| = {norm1:.4f}")
    print(f"  ||v2|| = {norm2:.4f}")

    if norm1 == 0 or norm2 == 0:
        similarity = 0
    else:
        similarity = dot_product / (norm1 * norm2)

    print(f"\n  similarity = dot(v1,v2) / (||v1|| × ||v2||)")
    print(f"             = {dot_product:.4f} / ({norm1:.4f} × {norm2:.4f})")
    print(f"             = {similarity:.4f}")
    print(f"\n  ★ SIMILARITY: {similarity:.1%}")

    return similarity


# =============================================================================
# STEP 3: RETRIEVAL (Finding Best Matches)
# =============================================================================

def retrieve_best_match(query_fp, corpus_fps, query_name="Query"):
    """
    Find the best matching fingerprint in a corpus.

    This is just: argmax similarity(query, each item in corpus)
    """
    print(f"\n{'='*60}")
    print(f"RETRIEVAL: Finding best match for '{query_name}'")
    print(f"{'='*60}")

    print(f"\nQuery fingerprint: {np.round(query_fp['vector'], 3)}")
    print(f"\nCorpus has {len(corpus_fps)} items")

    # Compare to each item in corpus
    results = []
    for name, fp in corpus_fps.items():
        v1 = query_fp['vector']
        v2 = fp['vector']

        # Pad to same length
        max_len = max(len(v1), len(v2))
        v1 = np.pad(v1, (0, max_len - len(v1)))
        v2 = np.pad(v2, (0, max_len - len(v2)))

        # Cosine similarity
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        sim = dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0

        results.append((name, sim, fp['vector']))
        print(f"  vs {name}: {sim:.1%}")

    # Sort by similarity (highest first)
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\nRanked results:")
    for i, (name, sim, vec) in enumerate(results, 1):
        print(f"  {i}. {name}: {sim:.1%}")

    best_name, best_sim, _ = results[0]
    print(f"\n★ BEST MATCH: {best_name} ({best_sim:.1%})")

    return results


# =============================================================================
# DEMO
# =============================================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║   HOW FINGERPRINTS AND RETRIEVAL WORK                                    ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)

    # ─────────────────────────────────────────────────────────────
    # Create some example graphs
    # ─────────────────────────────────────────────────────────────

    # Hub-spoke structure (like dragon body with wings)
    hub_spoke = nx.DiGraph()
    hub_spoke.add_edge("hub", "a")
    hub_spoke.add_edge("hub", "b")
    hub_spoke.add_edge("hub", "c")
    hub_spoke.add_edge("hub", "d")

    # Chain structure (like melody or narrative)
    chain = nx.DiGraph()
    chain.add_edge("1", "2")
    chain.add_edge("2", "3")
    chain.add_edge("3", "4")
    chain.add_edge("4", "5")

    # Another hub-spoke (different labels, same structure)
    hub_spoke_2 = nx.DiGraph()
    hub_spoke_2.add_edge("center", "leaf1")
    hub_spoke_2.add_edge("center", "leaf2")
    hub_spoke_2.add_edge("center", "leaf3")

    # Funnel/convergent structure
    funnel = nx.DiGraph()
    funnel.add_edge("a", "merge")
    funnel.add_edge("b", "merge")
    funnel.add_edge("c", "merge")
    funnel.add_edge("merge", "out")

    # ─────────────────────────────────────────────────────────────
    # Step 1: Create fingerprints
    # ─────────────────────────────────────────────────────────────

    fp_hub = create_fingerprint(hub_spoke, "Hub-Spoke (Dragon)")
    fp_chain = create_fingerprint(chain, "Chain (Melody)")
    fp_hub2 = create_fingerprint(hub_spoke_2, "Hub-Spoke-2 (Chord)")
    fp_funnel = create_fingerprint(funnel, "Funnel (Convergent)")

    # ─────────────────────────────────────────────────────────────
    # Step 2: Compare fingerprints
    # ─────────────────────────────────────────────────────────────

    sim_hub_chain = compare_fingerprints(fp_hub, fp_chain, "Hub-Spoke", "Chain")
    sim_hub_hub2 = compare_fingerprints(fp_hub, fp_hub2, "Hub-Spoke", "Hub-Spoke-2")

    # ─────────────────────────────────────────────────────────────
    # Step 3: Retrieval
    # ─────────────────────────────────────────────────────────────

    corpus = {
        "Chain (Melody)": fp_chain,
        "Hub-Spoke-2 (Chord)": fp_hub2,
        "Funnel (Convergent)": fp_funnel,
    }

    retrieve_best_match(fp_hub, corpus, "Hub-Spoke (Dragon)")

    # ─────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print("SUMMARY: THE FINGERPRINT PROCESS")
    print(f"{'='*60}")
    print("""
1. FINGERPRINT CREATION:
   Graph → Extract features → Vector of numbers

   Features extracted:
   • Scale: nodes, edges, density
   • Degrees: avg, max, hub ratio
   • Motifs: stars, chains, forks, sinks (normalized counts)
   • Spectral: Laplacian eigenvalues

2. COMPARISON:
   Cosine similarity between fingerprint vectors

   sim(A, B) = (A · B) / (||A|| × ||B||)

   Range: 0% (different) to 100% (identical structure)

3. RETRIEVAL:
   For each item in corpus:
     compute similarity(query, item)
   Return items sorted by similarity

THE KEY INSIGHT:
   The fingerprint captures TOPOLOGY, not labels.
   "hub → a,b,c,d" and "center → leaf1,leaf2,leaf3"
   have similar fingerprints because they have the
   same HUB-SPOKE structure.
    """)


if __name__ == "__main__":
    main()
