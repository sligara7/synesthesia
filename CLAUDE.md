# Synesthesia - Structural Rorschach

## Project Overview

Synesthesia is a graph analysis system that finds resonant patterns between images, music, text, and any structured data by comparing graph topologies rather than semantic content. The core insight: "Show me an image, and I'll tell you what it sounds like - not by meaning, but by *shape*."

## Directory Structure

```
synesthesia/
├── src/structural_rorschach/    # Main Python package
│   ├── signature.py             # StructuralSignature & Resonance dataclasses
│   ├── extractor.py             # Extract signatures from graphs
│   ├── motifs.py                # Detect structural patterns (hub-spoke, chain, etc.)
│   ├── spectral.py              # Spectral analysis (SVD, Laplacian)
│   ├── pruning.py               # Graph noise reduction
│   └── cli.py                   # Command-line interface
├── docs/                        # Architecture & requirements docs
└── README.md
```

## Key Concepts

- **StructuralSignature**: Domain-agnostic representation of graph topology (degree patterns, clustering, flow, motifs, centrality, spectral properties)
- **Resonance**: A cross-domain structural match between graphs from different domains
- **Motifs**: Structural patterns (hub-spoke, chain, triangle, fork, funnel, bridge, cycle) that form the vocabulary for cross-domain matching

## Development Commands

```bash
# Run tests (when available)
python -m pytest

# Run CLI
python -m structural_rorschach.cli --help

# Extract signature from graph file
python -m structural_rorschach.cli extract <graph_file.json>

# Compare two graphs
python -m structural_rorschach.cli compare <graph1.json> <graph2.json>
```

## Dependencies

- NetworkX (3.x+): Graph algorithms
- NumPy: Numerical computations
- SciPy: Sparse linear algebra (SVD, eigendecomposition)
- Python 3.7+

## Code Style

- Pure Python implementation
- Type hints encouraged
- Dataclasses for data structures
- JSON serialization for signatures
