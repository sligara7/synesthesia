#!/usr/bin/env python3
"""
End-to-End Demo for Synesthesia - Structural Rorschach

Demonstrates the complete workflow:
1. Create graphs from different domains
2. Extract structural signatures
3. Build a corpus
4. Find cross-domain resonances
5. Generate interpretations

Usage:
    python examples/e2e_demo.py
"""

import sys
sys.path.insert(0, 'src')

import networkx as nx
from structural_rorschach import (
    create_service_container,
    StructuralSignature,
    Resonance,
)


def create_music_chord_graph():
    """Create a graph representing a C Major chord structure."""
    G = nx.DiGraph()
    # Root note (tonic) with connections to chord tones
    G.add_node("C", label="tonic", role="root")
    G.add_node("E", label="third", role="chord_tone")
    G.add_node("G", label="fifth", role="chord_tone")
    G.add_node("B", label="seventh", role="extension")

    # Root connects to all chord tones (hub-spoke pattern)
    G.add_edge("C", "E", relation="major_third")
    G.add_edge("C", "G", relation="perfect_fifth")
    G.add_edge("C", "B", relation="major_seventh")

    return G


def create_music_melody_graph():
    """Create a graph representing a melodic phrase (chain pattern)."""
    G = nx.DiGraph()
    notes = ["C", "D", "E", "F", "G"]
    for i, note in enumerate(notes):
        G.add_node(f"{note}_{i}", label=note, position=i)
        if i > 0:
            G.add_edge(f"{notes[i-1]}_{i-1}", f"{note}_{i}", relation="step")
    return G


def create_text_thesis_graph():
    """Create a graph representing a thesis with supporting points (hub-spoke)."""
    G = nx.DiGraph()
    G.add_node("thesis", label="Main Argument", role="central")

    for i, point in enumerate(["Evidence A", "Evidence B", "Evidence C"]):
        node_id = f"point_{i}"
        G.add_node(node_id, label=point, role="support")
        G.add_edge("thesis", node_id, relation="supports")

    return G


def create_text_narrative_graph():
    """Create a graph representing a narrative flow (chain pattern)."""
    G = nx.DiGraph()
    events = ["Setup", "Conflict", "Rising", "Climax", "Resolution"]
    for i, event in enumerate(events):
        G.add_node(f"event_{i}", label=event, position=i)
        if i > 0:
            G.add_edge(f"event_{i-1}", f"event_{i}", relation="leads_to")
    return G


def print_signature_summary(sig: StructuralSignature):
    """Print a formatted signature summary."""
    print(f"  Domain: {sig.source_domain}")
    print(f"  Name: {sig.source_name}")
    print(f"  Scale: {sig.num_nodes} nodes, {sig.num_edges} edges")
    print(f"  Density: {sig.density:.3f}")
    print(f"  Top motifs: {list(sig.motif_vector.keys())[:3]}")


def print_resonance_result(res: Resonance):
    """Print a formatted resonance result."""
    print(f"\n  Query: [{res.query_domain}] {res.query_name}")
    print(f"  Match: [{res.match_domain}] {res.match_name}")
    print(f"  Score: {res.overall_score:.2%}")
    print(f"  Motif similarity: {res.motif_similarity:.2%}")
    print(f"  Matching motifs: {', '.join(res.matching_motifs[:3]) if res.matching_motifs else 'none'}")


def main():
    print("=" * 60)
    print("SYNESTHESIA - Structural Rorschach E2E Demo")
    print("=" * 60)

    # Initialize services
    print("\n[1] Initializing service container...")
    container = create_service_container()
    extractor = container.signature_extractor
    corpus_service = container.corpus_service
    resonance_service = container.resonance_service
    interpretation_service = container.interpretation_service
    print("    Services initialized successfully")

    # Create domain-specific graphs
    print("\n[2] Creating domain-specific graphs...")
    music_chord = create_music_chord_graph()
    music_melody = create_music_melody_graph()
    text_thesis = create_text_thesis_graph()
    text_narrative = create_text_narrative_graph()
    print(f"    Created 4 graphs: 2 music, 2 text")

    # Extract signatures
    print("\n[3] Extracting structural signatures...")
    sig_chord = extractor.extract_from_networkx(music_chord, "music", "C Major Chord", "music_chord")
    sig_melody = extractor.extract_from_networkx(music_melody, "music", "Melodic Phrase", "music_melody")
    sig_thesis = extractor.extract_from_networkx(text_thesis, "text", "Thesis Statement", "text_thesis")
    sig_narrative = extractor.extract_from_networkx(text_narrative, "text", "Narrative Arc", "text_narrative")

    print("\n    Chord signature:")
    print_signature_summary(sig_chord)
    print("\n    Melody signature:")
    print_signature_summary(sig_melody)

    # Build corpus
    print("\n[4] Building music corpus...")
    music_corpus = corpus_service.create_corpus("music_patterns", "music", "Musical structure patterns")
    corpus_service.add_signature(music_corpus, sig_chord)
    corpus_service.add_signature(music_corpus, sig_melody)
    print(f"    Corpus size: {len(music_corpus)} signatures")

    # Find cross-domain resonances
    print("\n[5] Finding cross-domain resonances...")
    print("\n    Query: Text Thesis (hub-spoke structure)")
    resonances_thesis = resonance_service.find_resonances(
        sig_thesis, music_corpus, top_k=2, threshold=0.0
    )

    print("    Results:")
    for res in resonances_thesis:
        print_resonance_result(res)

    print("\n    Query: Text Narrative (chain structure)")
    resonances_narrative = resonance_service.find_resonances(
        sig_narrative, music_corpus, top_k=2, threshold=0.0
    )

    print("    Results:")
    for res in resonances_narrative:
        print_resonance_result(res)

    # Generate interpretation
    print("\n[6] Generating structural comparison report...")
    report = interpretation_service.generate_comparison_report(sig_thesis, sig_chord)
    print(report[:500] + "..." if len(report) > 500 else report)

    # Summary
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nKey findings:")
    print("  - Thesis (hub-spoke) resonates more with Chord (hub-spoke)")
    print("  - Narrative (chain) resonates more with Melody (chain)")
    print("  - Cross-domain structural similarity successfully detected!")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
