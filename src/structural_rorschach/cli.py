#!/usr/bin/env python3
"""
Structural Rorschach CLI - Extract and compare structural signatures

Usage:
    # Extract signature from a graph
    python -m structural_rorschach.cli extract data/kjv_gospels_dag.json --domain text

    # Compare two graphs
    python -m structural_rorschach.cli compare data/kjv_gospels_dag.json data/byzantine_gospels_dag.json

    # Analyze motifs in a graph
    python -m structural_rorschach.cli motifs data/kjv_gospels_dag.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from structural_rorschach.signature import StructuralSignature
from structural_rorschach.extractor import SignatureExtractor
from structural_rorschach.motifs import MotifDetector

try:
    import networkx as nx
except ImportError:
    print("NetworkX required: pip install networkx")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("NumPy required: pip install numpy")
    sys.exit(1)


def load_graph_as_networkx(file_path: str) -> nx.Graph:
    """Load a graph JSON file into NetworkX"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    graph_section = data.get('graph', {})
    is_directed = graph_section.get('directed', True)

    G = nx.DiGraph() if is_directed else nx.Graph()

    for node in graph_section.get('nodes', []):
        node_id = node.get('id', node.get('name'))
        G.add_node(node_id)

    for link in graph_section.get('links', []):
        G.add_edge(link['source'], link['target'], weight=link.get('weight', 1.0))

    return G


def cmd_extract(args):
    """Extract structural signature from a graph"""
    print(f"Extracting signature from: {args.input}")
    print(f"Domain: {args.domain}")
    print()

    extractor = SignatureExtractor()
    sig = extractor.extract_from_file(args.input, domain=args.domain)

    print(sig.summary())
    print()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(sig.to_json(indent=2))
        print(f"Signature saved to: {args.output}")
    else:
        print("Motif Vector:")
        for motif, value in sorted(sig.motif_vector.items(), key=lambda x: -x[1]):
            if value > 0:
                print(f"  {motif}: {value:.4f}")

    return 0


def cmd_compare(args):
    """Compare two graphs and compute similarity"""
    print(f"Comparing graphs:")
    print(f"  Graph A: {args.graph_a}")
    print(f"  Graph B: {args.graph_b}")
    print()

    extractor = SignatureExtractor()

    sig_a = extractor.extract_from_file(args.graph_a, domain=args.domain_a)
    sig_b = extractor.extract_from_file(args.graph_b, domain=args.domain_b)

    print("=" * 60)
    print("GRAPH A SIGNATURE")
    print("=" * 60)
    print(sig_a.summary())
    print()

    print("=" * 60)
    print("GRAPH B SIGNATURE")
    print("=" * 60)
    print(sig_b.summary())
    print()

    # Compute similarities
    print("=" * 60)
    print("SIMILARITY ANALYSIS")
    print("=" * 60)

    # Motif vector cosine similarity
    motif_sim = compute_motif_similarity(sig_a.motif_vector, sig_b.motif_vector)
    print(f"Motif Vector Similarity: {motif_sim:.4f}")

    # Spectral similarity
    if sig_a.spectral_signature and sig_b.spectral_signature:
        spectral_sim = compute_spectral_similarity(
            sig_a.spectral_signature, sig_b.spectral_signature
        )
        print(f"Spectral Similarity: {spectral_sim:.4f}")

    # Scale similarity
    scale_sim = compute_scale_similarity(sig_a, sig_b)
    print(f"Scale Similarity: {scale_sim:.4f}")

    # Overall
    overall = (motif_sim + scale_sim) / 2
    print(f"\nOverall Structural Similarity: {overall:.4f}")

    # Interpretation
    print()
    if overall > 0.8:
        print("Interpretation: STRONG RESONANCE")
        print("These graphs have very similar structural patterns.")
    elif overall > 0.6:
        print("Interpretation: MODERATE RESONANCE")
        print("These graphs share some structural characteristics.")
    elif overall > 0.4:
        print("Interpretation: WEAK RESONANCE")
        print("These graphs have some structural overlap but differ significantly.")
    else:
        print("Interpretation: NO RESONANCE")
        print("These graphs have fundamentally different structures.")

    return 0


def cmd_motifs(args):
    """Analyze motifs in a graph"""
    print(f"Analyzing motifs in: {args.input}")
    print()

    G = load_graph_as_networkx(args.input)
    detector = MotifDetector(G)

    print(detector.summary())
    print()

    if args.verbose:
        all_motifs = detector.detect_all()
        for motif_type, matches in all_motifs.items():
            if matches:
                print(f"\n{motif_type.upper()} instances ({len(matches)}):")
                for match in matches[:5]:  # Show first 5
                    print(f"  Nodes: {match.nodes[:5]}...")  # Truncate for display

    return 0


def compute_motif_similarity(vec_a: dict, vec_b: dict) -> float:
    """Compute cosine similarity between motif vectors"""
    all_keys = set(vec_a.keys()) | set(vec_b.keys())

    a = np.array([vec_a.get(k, 0) for k in all_keys])
    b = np.array([vec_b.get(k, 0) for k in all_keys])

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_spectral_similarity(spec_a: list, spec_b: list) -> float:
    """Compute similarity between spectral signatures"""
    min_len = min(len(spec_a), len(spec_b))
    if min_len == 0:
        return 0.0

    a = np.array(spec_a[:min_len])
    b = np.array(spec_b[:min_len])

    # Normalized Euclidean distance converted to similarity
    dist = np.linalg.norm(a - b)
    max_dist = np.sqrt(2 * min_len)  # Max possible distance for normalized values

    return 1.0 - (dist / max_dist) if max_dist > 0 else 1.0


def compute_scale_similarity(sig_a: StructuralSignature, sig_b: StructuralSignature) -> float:
    """Compute similarity based on graph scale metrics"""
    # Compare density, clustering, hub_ratio
    metrics = [
        ('density', sig_a.density, sig_b.density),
        ('clustering', sig_a.clustering_coefficient, sig_b.clustering_coefficient),
        ('hub_ratio', sig_a.hub_ratio, sig_b.hub_ratio),
        ('avg_degree', sig_a.avg_degree, sig_b.avg_degree),
    ]

    similarities = []
    for name, val_a, val_b in metrics:
        max_val = max(abs(val_a), abs(val_b))
        if max_val > 0:
            sim = 1 - abs(val_a - val_b) / max_val
            similarities.append(sim)
        else:
            similarities.append(1.0)  # Both zero = same

    return np.mean(similarities) if similarities else 0.0


def main():
    parser = argparse.ArgumentParser(
        description='Structural Rorschach - Cross-domain graph analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract signature from a text graph
  python -m structural_rorschach.cli extract data/kjv_gospels_dag.json --domain text

  # Compare two graphs across domains
  python -m structural_rorschach.cli compare data/kjv_gospels_dag.json data/test_image_graph.json \\
      --domain-a text --domain-b image

  # Analyze motifs
  python -m structural_rorschach.cli motifs data/kjv_gospels_dag.json -v
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract structural signature')
    extract_parser.add_argument('input', help='Input graph JSON file')
    extract_parser.add_argument('--domain', '-d', default='unknown',
                                help='Domain (text, music, image, etc.)')
    extract_parser.add_argument('--output', '-o', help='Output signature JSON file')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two graphs')
    compare_parser.add_argument('graph_a', help='First graph JSON file')
    compare_parser.add_argument('graph_b', help='Second graph JSON file')
    compare_parser.add_argument('--domain-a', default='unknown', help='Domain of graph A')
    compare_parser.add_argument('--domain-b', default='unknown', help='Domain of graph B')

    # Motifs command
    motifs_parser = subparsers.add_parser('motifs', help='Analyze motifs in a graph')
    motifs_parser.add_argument('input', help='Input graph JSON file')
    motifs_parser.add_argument('--verbose', '-v', action='store_true',
                               help='Show detailed motif instances')

    args = parser.parse_args()

    if args.command == 'extract':
        return cmd_extract(args)
    elif args.command == 'compare':
        return cmd_compare(args)
    elif args.command == 'motifs':
        return cmd_motifs(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
