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
- SimilarityService: Compute structural similarity
- CorpusService: Manage signature collections
- ResonanceService: Find cross-domain matches
- InterpretationService: Generate human-readable explanations
- DomainAdapters: Convert domain data to graphs
"""

# Core signature types
from .signature import StructuralSignature, Resonance

# Extractors
from .extractor import SignatureExtractor
from .spectral import SpectralSignature, SpectralExtractor, spectral_similarity

# Motif detection
from .motifs import MotifDetector

# Pruning
from .pruning import GraphPruner, auto_prune, PruningStats

# Similarity computation
from .similarity import (
    SimilarityService,
    SimilarityResult,
    compute_similarity
)

# Corpus management
from .corpus import (
    Corpus,
    CorpusService,
    create_corpus,
    load_corpus,
    save_corpus
)

# Resonance finding
from .resonance import (
    ResonanceService,
    find_resonances,
    find_cross_domain_resonances
)

# Interpretation
from .interpretation import (
    InterpretationService,
    explain_resonance,
    generate_comparison_report
)

# Domain adapters
from .adapters import (
    DomainAdapter,
    TextAdapter,
    ImageAdapter,
    MusicAdapter,
    CodeAdapter,
    get_adapter,
    adapt_to_graph,
    validate_graph,
    GraphValidationResult
)

__version__ = "0.3.0"
__all__ = [
    # Core signature types
    "StructuralSignature",
    "SpectralSignature",
    "Resonance",

    # Extractors
    "SignatureExtractor",
    "SpectralExtractor",

    # Motif detection
    "MotifDetector",
    "spectral_similarity",

    # Pruning
    "GraphPruner",
    "auto_prune",
    "PruningStats",

    # Similarity computation (NEW)
    "SimilarityService",
    "SimilarityResult",
    "compute_similarity",

    # Corpus management (NEW)
    "Corpus",
    "CorpusService",
    "create_corpus",
    "load_corpus",
    "save_corpus",

    # Resonance finding (NEW)
    "ResonanceService",
    "find_resonances",
    "find_cross_domain_resonances",

    # Interpretation (NEW)
    "InterpretationService",
    "explain_resonance",
    "generate_comparison_report",

    # Domain adapters (NEW)
    "DomainAdapter",
    "TextAdapter",
    "ImageAdapter",
    "MusicAdapter",
    "CodeAdapter",
    "get_adapter",
    "adapt_to_graph",
    "validate_graph",
    "GraphValidationResult",
]
