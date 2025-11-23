"""
Structural Rorschach - Cross-Domain Graph Analysis

A system for finding structural resonances across domains (images, music, text)
by comparing graph topologies rather than semantic content.

Inspired by:
- Synesthesia: Cross-sensory perception
- Rorschach inkblot test: Structural interpretation

Core concept: Any structured data can be the "inkblot" that the system
interprets through other domain corpora by matching graph patterns.

Key components:
- StructuralSignature: Full graph analysis (motif-based)
- SpectralSignature: Compressed analysis using linear algebra (scales to any size)
- GraphPruner: Remove noise to focus on essential structure
"""

from .signature import StructuralSignature, Resonance
from .extractor import SignatureExtractor
from .motifs import MotifDetector
from .spectral import SpectralSignature, SpectralExtractor, spectral_similarity
from .pruning import GraphPruner, auto_prune, PruningStats

__version__ = "0.2.0"
__all__ = [
    # Core signature types
    "StructuralSignature",
    "SpectralSignature",
    "Resonance",
    # Extractors
    "SignatureExtractor",
    "SpectralExtractor",
    # Analysis
    "MotifDetector",
    "spectral_similarity",
    # Pruning
    "GraphPruner",
    "auto_prune",
    "PruningStats",
]
