# Synesthesia - Structural Rorschach

## Project Overview

Synesthesia is a graph analysis system that finds resonant patterns between images, music, text, and any structured data by comparing graph topologies rather than semantic content. The core insight: "Show me an image, and I'll tell you what it sounds like - not by meaning, but by *shape*."

## Directory Structure

```
synesthesia/
├── src/structural_rorschach/    # Main Python package (v0.3.0)
│   ├── signature.py             # StructuralSignature & Resonance dataclasses
│   ├── extractor.py             # Extract signatures from graphs
│   ├── motifs.py                # Detect structural patterns
│   ├── spectral.py              # Spectral analysis (SVD, Laplacian)
│   ├── pruning.py               # Graph noise reduction
│   ├── similarity.py            # Compute structural similarity (NEW)
│   ├── corpus.py                # Manage signature collections (NEW)
│   ├── resonance.py             # Find cross-domain matches (NEW)
│   ├── interpretation.py        # Generate explanations (NEW)
│   ├── adapters.py              # Domain adapters (text/image/music/code) (NEW)
│   └── cli.py                   # Command-line interface
├── specs/                       # Architecture specifications
│   ├── functional/              # Functional architecture (40 functions)
│   └── machine/                 # Service architecture (8 services)
├── context/                     # Reflow working memory
├── docs/                        # Documentation & diagrams
└── README.md
```

## Key Services

| Service | Module | Functions |
|---------|--------|-----------|
| SignatureService | extractor.py | F-SE-01 to F-SE-07 |
| MotifService | motifs.py | F-MD-01 to F-MD-07 |
| SimilarityService | similarity.py | F-SC-01 to F-SC-04 |
| CorpusService | corpus.py | F-CM-01 to F-CM-06 |
| ResonanceService | resonance.py | F-RF-01 to F-RF-04 |
| InterpretationService | interpretation.py | F-IE-01 to F-IE-03 |
| DomainAdapterService | adapters.py | F-DA-01 to F-DA-05 |
| GraphUtilityService | pruning.py | F-GU-01 to F-GU-06 |

## Quick Start

```python
from structural_rorschach import (
    TextAdapter, SignatureExtractor,
    create_corpus, ResonanceService,
    explain_resonance
)

# 1. Convert text to graph
adapter = TextAdapter()
graph = adapter.adapt("Your text corpus here...")

# 2. Extract structural signature
extractor = SignatureExtractor()
signature = extractor.extract_from_dict(graph, domain="text", name="My Text")

# 3. Find resonances in a corpus
corpus = load_corpus("music_corpus.json")
service = ResonanceService()
resonances = service.find_resonances(signature, corpus)

# 4. Explain the match
for r in resonances:
    print(explain_resonance(r))
```

## Development Commands

```bash
# Run tests
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
- SciPy: Sparse linear algebra
- Python 3.7+
- Optional: scikit-image (image adapter), mido (MIDI adapter)

## Architecture

- **Functional Architecture**: 40 functions across 8 categories
- **Service Architecture**: 8 services with clear boundaries
- **100% Implementation Coverage**: All functional requirements implemented
