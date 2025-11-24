#!/usr/bin/env python3
"""
Interactive Synesthesia Demo

Run this and explore structural resonance with your own inputs.
"""

import sys
sys.path.insert(0, 'src')

import networkx as nx
from collections import Counter
from structural_rorschach import create_service_container


def text_to_graph(text: str) -> nx.DiGraph:
    """Convert text to a word transition graph."""
    words = text.lower().split()
    words = [w.strip('.,!?;:()[]"\'') for w in words if w.strip('.,!?;:()[]"\'')]

    G = nx.DiGraph()
    for word in set(words):
        G.add_node(word)

    for i in range(len(words) - 1):
        if not G.has_edge(words[i], words[i+1]):
            G.add_edge(words[i], words[i+1], weight=1)
        else:
            G[words[i]][words[i+1]]['weight'] += 1

    return G


def analyze_structure(G: nx.DiGraph) -> dict:
    """Analyze graph structure and return key metrics."""
    out_degrees = dict(G.out_degree())
    in_degrees = dict(G.in_degree())

    hubs = [n for n, d in out_degrees.items() if d > 2]
    chains = sum(1 for n in G.nodes() if in_degrees[n] == 1 and out_degrees[n] == 1)
    sources = [n for n in G.nodes() if in_degrees[n] == 0]
    sinks = [n for n in G.nodes() if out_degrees[n] == 0]

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "hubs": hubs[:5],
        "chain_nodes": chains,
        "sources": sources[:5],
        "sinks": sinks[:5],
        "density": nx.density(G),
    }


def print_analysis(name: str, analysis: dict):
    """Pretty print structure analysis."""
    print(f"\n  {name}:")
    print(f"    Nodes: {analysis['nodes']}, Edges: {analysis['edges']}")
    print(f"    Density: {analysis['density']:.3f}")
    if analysis['hubs']:
        print(f"    Hubs: {', '.join(analysis['hubs'])}")
    print(f"    Chain nodes: {analysis['chain_nodes']}")
    if analysis['sources']:
        print(f"    Entry points: {', '.join(analysis['sources'][:3])}")


def compare_texts():
    """Compare two pieces of text."""
    print("\n" + "="*60)
    print("TEXT COMPARISON")
    print("="*60)

    print("\nEnter first text (or press Enter for example):")
    text1 = input("> ").strip()
    if not text1:
        text1 = """
        The main function orchestrates the entire workflow.
        First it initializes the configuration and logging.
        Then it loads the data from multiple sources.
        After processing, it stores results in the database.
        Finally it sends notifications to all subscribers.
        """
        print(f"  Using example: '{text1[:50]}...'")

    print("\nEnter second text (or press Enter for example):")
    text2 = input("> ").strip()
    if not text2:
        text2 = """
        The chef begins by preparing the mise en place.
        First they wash and chop all vegetables.
        Then they measure out spices and seasonings.
        After heating the pan, they add ingredients in order.
        Finally they plate and garnish the finished dish.
        """
        print(f"  Using example: '{text2[:50]}...'")

    # Analyze
    container = create_service_container()
    extractor = container.signature_extractor
    similarity_service = container.similarity_service

    G1 = text_to_graph(text1)
    G2 = text_to_graph(text2)

    print("\nStructural Analysis:")
    print_analysis("Text 1", analyze_structure(G1))
    print_analysis("Text 2", analyze_structure(G2))

    sig1 = extractor.extract_from_networkx(G1, "text", "Text 1", "text1")
    sig2 = extractor.extract_from_networkx(G2, "text", "Text 2", "text2")

    result = similarity_service.compute_similarity(sig1, sig2)

    print("\n" + "-"*40)
    print("STRUCTURAL SIMILARITY")
    print("-"*40)
    print(f"  Overall:  {result.overall_score:.1%}")
    print(f"  Motif:    {result.motif_similarity:.1%}")
    print(f"  Spectral: {result.spectral_similarity:.1%}")
    print(f"  Scale:    {result.scale_similarity:.1%}")

    if result.overall_score > 0.8:
        print("\n  VERDICT: These texts have VERY SIMILAR structure!")
    elif result.overall_score > 0.5:
        print("\n  VERDICT: These texts share some structural patterns.")
    else:
        print("\n  VERDICT: These texts have DIFFERENT structures.")


def build_custom_corpus():
    """Build a corpus and find matches."""
    print("\n" + "="*60)
    print("BUILD YOUR OWN CORPUS")
    print("="*60)

    container = create_service_container()
    extractor = container.signature_extractor
    corpus_service = container.corpus_service
    resonance_service = container.resonance_service

    corpus = corpus_service.create_corpus("custom", "text")
    signatures = []

    print("\nAdd texts to your corpus (enter empty line when done):")
    i = 1
    while True:
        print(f"\nText {i} (or Enter to finish):")
        text = input("> ").strip()
        if not text:
            break

        G = text_to_graph(text)
        sig = extractor.extract_from_networkx(G, "text", f"Text {i}", f"text_{i}")
        corpus_service.add_signature(corpus, sig)
        signatures.append((f"Text {i}", sig))
        print(f"  Added! ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
        i += 1

    if len(signatures) < 2:
        print("\nNeed at least 2 texts to compare. Using examples...")
        examples = [
            "The quick brown fox jumps over the lazy dog.",
            "A fast red cat leaps across the sleepy hound.",
            "Programming requires logic and creativity to solve problems.",
        ]
        for j, text in enumerate(examples):
            G = text_to_graph(text)
            sig = extractor.extract_from_networkx(G, "text", f"Example {j+1}", f"ex_{j}")
            corpus_service.add_signature(corpus, sig)
            signatures.append((f"Example {j+1}", sig))

    print(f"\nCorpus has {len(signatures)} texts.")
    print("\nEnter a query text to find similar structures:")
    query_text = input("> ").strip()
    if not query_text:
        query_text = "The clever orange wolf hops past the tired cat."
        print(f"  Using: '{query_text}'")

    G = text_to_graph(query_text)
    query_sig = extractor.extract_from_networkx(G, "text", "Query", "query")

    print("\n" + "-"*40)
    print("MATCHES (by structural similarity)")
    print("-"*40)

    resonances = resonance_service.find_resonances(query_sig, corpus, top_k=5, threshold=0.0)
    for i, res in enumerate(resonances, 1):
        print(f"  {i}. {res.match_name}: {res.overall_score:.1%}")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║   SYNESTHESIA - Interactive Demo                                     ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    while True:
        print("\nWhat would you like to do?")
        print("  1. Compare two texts")
        print("  2. Build a corpus and find matches")
        print("  3. Exit")

        choice = input("\nChoice (1-3): ").strip()

        if choice == "1":
            compare_texts()
        elif choice == "2":
            build_custom_corpus()
        elif choice == "3":
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
