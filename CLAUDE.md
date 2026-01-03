# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Synesthesia is a graph analysis system that finds resonant patterns between images, music, text, and any structured data by comparing graph topologies rather than semantic content. The core insight: "Show me an image, and I'll tell you what it sounds like - not by meaning, but by *shape*."

**Key concept**: Structure is meaning. A flower's radial petal arrangement might match a chord's harmonic structure because they share *shape*, not semantic meaning.

## Development Commands

```bash
# Run all tests
python -m pytest

# Run single test file
python -m pytest tests/test_similarity.py

# Run single test
python -m pytest tests/test_integration.py::TestFullWorkflow::test_signature_extraction_from_graph

# Run with verbose output
python -m pytest -v

# Run CLI
python -m structural_rorschach.cli --help

# Extract signature from graph file
python -m structural_rorschach.cli extract <graph_file.json>

# Compare two graphs
python -m structural_rorschach.cli compare <graph1.json> <graph2.json>
```

## Architecture

### Core Pipeline

The system follows a clear data flow:
```
Domain Data → Adapter → Graph → Extractor → Signature → Corpus/Resonance → Explanation
```

### Service Container (Dependency Injection)

All services are wired through `container.py` using Protocol-based dependency injection:

```python
from structural_rorschach import create_service_container, get_container

container = create_service_container()
signature = container.signature_extractor.extract(graph, "text", "my_text")
```

Services can be swapped by passing custom implementations that satisfy the Protocol interfaces in `protocols.py`.

### Key Modules

| Module | Purpose |
|--------|---------|
| `signature.py` | `StructuralSignature` and `Resonance` dataclasses - the core data types |
| `extractor.py` | Extract signatures from NetworkX graphs |
| `motifs.py` | Detect structural patterns (hub-spoke, chain, triangle, fork, funnel) |
| `spectral.py` | SVD/Laplacian spectral analysis for large graphs |
| `similarity.py` | Compute structural similarity (40% motif, 30% spectral, 30% scale) |
| `corpus.py` | Manage collections of signatures |
| `resonance.py` | Find cross-domain structural matches |
| `interpretation.py` | Generate human-readable explanations |
| `adapters.py` | Domain adapters (TextAdapter, ImageAdapter, MusicAdapter, CodeAdapter) |
| `protocols.py` | Protocol interfaces for all services |
| `container.py` | Service container and dependency injection |

### Cave Trader Subsystem

A game-based market visualization where market structure becomes navigable terrain:
- `cave_trader.py` - Core game converting OHLCV data to cave geometry
- `simple_cave.py` - Nokia-era simplicity game implementation
- `cave_instruments.py` - 2D flight-style instrument panel
- `snake_trail.py` / `snake_music.py` - Position tracking and sonification
- `market_instruments.py` - Harmonic sonification of market data
- `market_harmonics.py` - True cycle detection via spectral analysis
- `game_levels.py` - Historical market data as playable levels

### Protocol System

Services communicate through typed Protocol interfaces (`protocols.py`):
- `CanExtractSignatures`, `CanComputeSimilarity`, `CanFindResonances`, etc.
- Enables swapping implementations without changing dependent code
- Use `create_test_container()` to inject mock services in tests

## Dependencies

- NetworkX (3.x+): Graph algorithms
- NumPy: Numerical computations
- SciPy: Sparse linear algebra
- Python 3.7+
- Optional: scikit-image (image adapter), mido (MIDI adapter)

## Quick Usage Example

```python
from structural_rorschach import (
    TextAdapter, SignatureExtractor,
    create_corpus, ResonanceService, explain_resonance
)

# Convert text to graph
adapter = TextAdapter()
graph = adapter.adapt("Your text corpus here...")

# Extract structural signature
extractor = SignatureExtractor()
signature = extractor.extract_from_dict(graph, domain="text", name="My Text")

# Find resonances in a corpus
corpus = load_corpus("music_corpus.json")
service = ResonanceService()
resonances = service.find_resonances(signature, corpus)

# Explain the match
for r in resonances:
    print(explain_resonance(r))
```
