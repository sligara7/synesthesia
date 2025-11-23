# Synesthesia: Structural Rorschach Analysis

> "Show me an image, and I'll tell you what it sounds like - not by meaning, but by *shape*."

## Vision

A system that performs **"Structural Rorschach"** analysis across domains - finding resonant patterns between images, music, text, and any other structured data by comparing their graph topologies rather than their semantic content.

Inspired by synesthesia (cross-sensory perception) and the Rorschach inkblot test, this system interprets any structured input through the "lens" of other domains by matching graph patterns.

**Key Insight**: The *structure* of data (not its content) can resonate across domains.
- A flower's radial petal arrangement might match a chord's harmonic structure
- A narrative arc might match a melodic contour
- A code dependency graph might match an organizational hierarchy

## The Broader Vision

> "What sometimes is difficult to understand in one domain becomes immediately clear in another domain to see."

People learn differently - some are visual learners, others auditory. The same truth can be accessed through different perceptual doors, and different people have different doors that work best for them.

```
Traditional approach:     One representation → Everyone must adapt

Cross-domain approach:    One truth → Many representations → Each person
                          finds their natural door
```

## Installation

```bash
pip install networkx numpy scipy
```

## Project Structure

```
synesthesia/
├── src/structural_rorschach/
│   ├── __init__.py          # Package exports
│   ├── signature.py         # StructuralSignature dataclass
│   ├── extractor.py         # Extract signatures from graphs
│   ├── spectral.py          # Spectral analysis (SVD, Laplacian)
│   ├── motifs.py            # Structural pattern detection
│   ├── pruning.py           # Graph noise reduction
│   └── cli.py               # Command-line interface
├── docs/
│   ├── CROSS_DOMAIN_DAG_FOUNDATION.md  # Architecture and design
│   ├── FUNCTIONAL_REQUIREMENTS.md      # Full FRD
│   └── FUTURE_EXPLORATIONS.md          # Vision and applications
└── README.md
```

## Usage

### Extract Structural Signature from a Graph

```python
from structural_rorschach import extract_signature
import networkx as nx

# Any NetworkX graph
G = nx.karate_club_graph()
signature = extract_signature(G, domain="social", graph_id="karate")

print(f"Nodes: {signature.num_nodes}")
print(f"Communities: {signature.num_communities}")
print(f"Hub ratio: {signature.hub_ratio:.2%}")
```

### Compare Graphs Across Domains

```python
from structural_rorschach import find_resonances

# Graphs from different domains
image_sig = extract_signature(image_graph, domain="image")
music_sig = extract_signature(music_graph, domain="music")

resonances = find_resonances(image_sig, music_sig)
for r in resonances:
    print(f"{r.resonance_type}: {r.strength:.2%} - {r.interpretation}")
```

### Spectral Analysis for Large Graphs

```python
from structural_rorschach import SpectralSignature

# O(nk²) complexity instead of O(n³) for full analysis
spectral = SpectralSignature.from_graph(large_graph, n_components=50)
print(f"Estimated communities: {spectral.estimated_communities}")
```

### CLI

```bash
# Analyze a graph
python -m structural_rorschach.cli analyze graph.json

# Compare two graphs
python -m structural_rorschach.cli compare graph1.json graph2.json
```

## Core Concepts

### Domain-Agnostic Graphs
All domains convert to a common graph format, stripping away domain-specific semantics to reveal pure topology.

### Structural Signatures
Domain-agnostic features extracted from any graph:
- Scale metrics (nodes, edges, density)
- Degree patterns (distribution, entropy, hub ratio)
- Clustering patterns (communities, modularity)
- Flow patterns (path lengths, DAG properties)
- Motif frequencies (hub-spoke, chains, triangles, bridges)

### Spectral Signatures
Linear algebra-based compression for large graphs:
- SVD of adjacency matrix for structural fingerprint
- Laplacian eigendecomposition for connectivity estimates
- Handles graphs with thousands of nodes efficiently

## Applications

| Domain | Alternative Access | Who Benefits |
|--------|-------------------|--------------|
| Mathematics | Colors, shapes, music | Visual/auditory learners |
| Music | Visual patterns, vibration | Deaf musicians |
| Trading/Finance | Sound, games, physical metaphors | Pattern "feelers" |
| Programming | Visual flow, sound | Different cognitive styles |

## Documentation

- [Architecture & Design](docs/CROSS_DOMAIN_DAG_FOUNDATION.md)
- [Functional Requirements](docs/FUNCTIONAL_REQUIREMENTS.md)
- [Future Explorations](docs/FUTURE_EXPLORATIONS.md)

## License

MIT
