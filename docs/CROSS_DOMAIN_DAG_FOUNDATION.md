# Cross-Domain DAG Foundation: Structural Synesthesia

## Vision

Create a system that performs **"Structural Rorschach"** analysis across domains - finding resonant patterns between images, music, text, and any other structured data by comparing their graph topologies rather than their semantic content.

> "Show me an image, and I'll tell you what it sounds like - not by meaning, but by *shape*."

---

## Core Concepts

### 1. Structural Synesthesia
Inspired by synesthesia (cross-sensory perception) and the Rorschach inkblot test, this system interprets any structured input through the "lens" of other domains by matching graph patterns.

**Key Insight**: The *structure* of data (not its content) can resonate across domains.
- A flower's radial petal arrangement might match a chord's harmonic structure
- A narrative arc might match a melodic contour
- A code dependency graph might match an organizational hierarchy

### 2. Domain-Agnostic Graphs
All domains convert to a common graph format (`system_of_systems_graph.json`), stripping away domain-specific semantics to reveal pure topology.

### 3. The "Inkblot" Query
Any graph can be the "inkblot" - the ambiguous structure that the system interprets through other domain corpora.

---

## Linking Strategies (from chain_reflow)

The chain_reflow framework provides proven strategies for connecting graphs:

| Strategy | When to Use | Cross-Domain Application |
|----------|-------------|-------------------------|
| **HIERARCHICAL** | Clear parent-child relationships | Domain → Sub-domain (Music → Melody/Harmony/Rhythm) |
| **PAIRWISE** | Small peer systems (N ≤ 3) | Direct image↔music, music↔text comparisons |
| **NETWORK** | Medium peer networks (4-10) | Mesh of all domains, find central "hub" patterns |
| **PHASED** | Large graph counts (N > 10) | Cluster similar structures first, then cross-match |
| **PAIRWISE_WITH_INTERMEDIARIES** | Divergent domains need bridges | Create "adapter" signatures between very different domains |

### Recommended Strategy for Synesthesia: PAIRWISE_WITH_INTERMEDIARIES

Cross-domain graphs are inherently divergent (image regions vs. word transitions). We need **intermediary bridge structures**:

```
Image Graph ←→ [Structural Signature] ←→ Music Graph
                      ↑
              Domain-agnostic
              motif representation
```

---

## Architecture

### Layer 1: Domain Adapters (Existing)
Convert domain-specific data to unified graph format.

```
┌─────────────────────────────────────────────────────────────────┐
│  DOMAIN ADAPTERS (existing in wordplay)                         │
├─────────────────────────────────────────────────────────────────┤
│  image_graph_builder.py    → Image regions + adjacencies        │
│  word_graph_builder.py     → Word transitions + frequencies     │
│  midi_graph_builder.py     → Note transitions + durations       │
│  audio_graph_analyzer.py   → Audio nodes + harmonic/temporal    │
│  [future] code_graph.py    → AST nodes + dependencies           │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 2: Structural Signature Extractor (NEW)
Extract domain-agnostic structural features from any graph.

```python
@dataclass
class StructuralSignature:
    """The 'intermediary' that bridges domains"""

    # Identity
    source_domain: str           # "image", "music", "text"
    source_id: str               # Original graph identifier

    # Scale metrics
    num_nodes: int
    num_edges: int
    density: float               # edges / possible_edges

    # Degree patterns (normalized)
    degree_distribution: List[float]
    degree_entropy: float        # How uniform is connectivity?
    hub_ratio: float             # % of nodes that are hubs

    # Clustering patterns
    clustering_coefficient: float
    num_communities: int
    modularity: float            # How separable into clusters?
    community_sizes: List[float] # Normalized distribution

    # Flow patterns
    avg_path_length: float
    diameter: int
    is_dag: bool
    longest_path: int

    # Motif fingerprint
    motif_vector: Dict[str, float]  # Normalized counts of:
        # "star_3": 0.12,       # 3-node star patterns
        # "chain_3": 0.34,      # 3-node chains
        # "triangle": 0.08,     # Closed triangles
        # "fork": 0.21,         # Branching points
        # "cycle_4": 0.05,      # 4-node cycles

    # Centrality patterns
    centrality_gini: float       # Inequality of node importance
    betweenness_concentration: float
```

### Layer 3: Resonance Matcher (NEW)
Find structurally similar graphs across domains.

```python
class ResonanceMatcher:
    """Find cross-domain structural matches"""

    def __init__(self, corpora: Dict[str, List[StructuralSignature]]):
        """
        corpora = {
            "text": [kjv_sig, byzantine_sig, ...],
            "music": [bach_sig, beethoven_sig, ...],
            "image": [flower_sig, portrait_sig, ...]
        }
        """
        self.corpora = corpora

    def find_resonances(
        self,
        query: StructuralSignature,
        target_domains: List[str] = None,
        top_k: int = 5
    ) -> List[Resonance]:
        """
        Find top-k structurally similar items across target domains
        """
        pass

    def compute_similarity(
        self,
        sig1: StructuralSignature,
        sig2: StructuralSignature
    ) -> float:
        """
        Domain-agnostic structural similarity [0, 1]

        Methods:
        - Motif vector cosine similarity
        - Degree distribution KL-divergence
        - Spectral similarity (graph Laplacian eigenvalues)
        - Graph edit distance (approximate)
        """
        pass
```

### Layer 4: Interpretation Engine (NEW)
Generate human-readable explanations of why structures resonate.

```python
@dataclass
class Resonance:
    """A cross-domain structural match"""

    query_source: str            # "image:flower_001"
    match_source: str            # "music:bach_prelude_1"
    similarity_score: float      # 0.87

    # Why they match
    matching_motifs: List[str]   # ["hub_spoke", "radial_symmetry"]
    shared_properties: Dict      # {"clustering": 0.4, "hub_ratio": 0.15}

    # Human explanation
    explanation: str
    # "The flower's radial petal arrangement (hub-spoke topology)
    #  resonates with the arpeggiated chord structure in Bach's
    #  Prelude, where the tonic acts as a hub connecting to
    #  harmonic extensions."
```

---

## Motif Vocabulary

Define structural patterns that are meaningful across domains:

| Motif | Graph Pattern | Image Meaning | Music Meaning | Text Meaning |
|-------|--------------|---------------|---------------|--------------|
| **Hub-Spoke** | High-degree central node | Focal point, radial symmetry | Tonic/root, drone | Central theme, key word |
| **Chain** | Linear sequence | Edge, contour, flow | Melodic line, scale | Narrative, sequence |
| **Cluster** | Densely connected group | Color region, object | Chord voicing, harmony | Topic, paragraph |
| **Bridge** | Node connecting communities | Transition zone | Modulation, pivot chord | Plot twist, conjunction |
| **Cycle** | Closed loop | Symmetry, repetition | Ostinato, refrain | Repetition, callback |
| **Star-3** | One node, 3 leaves | Branching, choice | Triad | Alternatives |
| **Triangle** | 3 fully connected | Stable region | Major/minor triad | Circular reference |
| **Fork** | 1→many | Divergence | Arpeggio start | Enumeration |
| **Funnel** | Many→1 | Convergence | Resolution | Conclusion |

---

## Similarity Algorithms

### 1. Motif Vector Cosine Similarity
```python
def motif_similarity(sig1, sig2):
    """Compare motif frequency distributions"""
    v1 = np.array(list(sig1.motif_vector.values()))
    v2 = np.array(list(sig2.motif_vector.values()))
    return cosine_similarity(v1, v2)
```

### 2. Spectral Similarity
```python
def spectral_similarity(graph1, graph2):
    """Compare graph structure via Laplacian eigenvalues"""
    eig1 = nx.laplacian_spectrum(graph1)[:k]
    eig2 = nx.laplacian_spectrum(graph2)[:k]
    return 1 - np.linalg.norm(eig1 - eig2) / max(np.linalg.norm(eig1), np.linalg.norm(eig2))
```

### 3. Graph Edit Distance (Approximate)
```python
def structural_distance(graph1, graph2):
    """Minimum edits to transform one graph to another"""
    # Use approximation for large graphs
    return nx.graph_edit_distance(graph1, graph2, timeout=5)
```

### 4. Weisfeiler-Lehman Hash Similarity
```python
def wl_similarity(graph1, graph2, iterations=3):
    """Compare graphs using WL subtree kernel"""
    hash1 = nx.weisfeiler_lehman_graph_hash(graph1, iterations=iterations)
    hash2 = nx.weisfeiler_lehman_graph_hash(graph2, iterations=iterations)
    return hash1 == hash2  # Or use partial matching
```

---

## Implementation Plan

### Phase 1: Foundation (Current)
- [x] Document architecture and concepts
- [ ] Define `StructuralSignature` dataclass
- [ ] Implement motif extraction for existing graphs
- [ ] Build signature extractor that works with all domain adapters

### Phase 2: Corpus Building
- [ ] Generate signatures for KJV Gospels DAG
- [ ] Generate signatures for Byzantine Greek DAG
- [ ] Generate signatures for sample images
- [ ] Generate signatures for MIDI files
- [ ] Store corpus of signatures for fast lookup

### Phase 3: Resonance Matching
- [ ] Implement similarity algorithms
- [ ] Build `ResonanceMatcher` class
- [ ] Create CLI for cross-domain queries
- [ ] Add interpretation/explanation generation

### Phase 4: Exploration & Validation
- [ ] Test: Image of flower → find resonant music
- [ ] Test: Musical phrase → find resonant text
- [ ] Test: Text passage → find resonant images
- [ ] Evaluate: Are the matches meaningful/interesting?

### Phase 5: Extensions
- [ ] Subgraph matching (find patterns within larger graphs)
- [ ] Generation (create content in domain B that matches structure of domain A)
- [ ] Interactive exploration UI
- [ ] Additional domains (code, social networks, etc.)

---

## File Structure

```
src/
├── structural_rorschach/
│   ├── __init__.py
│   ├── signature.py           # StructuralSignature dataclass
│   ├── extractor.py           # Extract signatures from graphs
│   ├── motifs.py              # Motif detection algorithms
│   ├── similarity.py          # Similarity algorithms
│   ├── resonance.py           # ResonanceMatcher class
│   ├── interpreter.py         # Human-readable explanations
│   └── corpus.py              # Manage domain corpora
│
├── adapters/                   # (refactor existing)
│   ├── image_adapter.py       # image_graph_builder.py
│   ├── text_adapter.py        # word_graph_builder.py
│   ├── music_adapter.py       # midi_graph_builder.py
│   └── audio_adapter.py       # audio_graph_analyzer.py

data/
├── signatures/                 # Pre-computed signatures
│   ├── text_signatures.json
│   ├── music_signatures.json
│   └── image_signatures.json

docs/
├── CROSS_DOMAIN_DAG_FOUNDATION.md  # This document
└── MOTIF_VOCABULARY.md             # Detailed motif definitions
```

---

## Integration with chain_reflow

The chain_reflow workflow patterns can orchestrate cross-domain analysis:

```json
{
  "workflow": "structural-rorschach-query",
  "steps": [
    {
      "id": "01-extract-query-signature",
      "action": "Extract StructuralSignature from input graph"
    },
    {
      "id": "02-determine-strategy",
      "action": "Select linking strategy (PAIRWISE_WITH_INTERMEDIARIES)",
      "uses": "chain-01a-determine-strategy.json"
    },
    {
      "id": "03-find-resonances",
      "action": "Match against all domain corpora"
    },
    {
      "id": "04-interpret",
      "action": "Generate explanations for top matches"
    },
    {
      "id": "05-visualize",
      "action": "Create comparison visualizations"
    }
  ]
}
```

---

## Open Questions

1. **Scale normalization**: How do we compare a 50-node image graph to a 5000-node text graph?
   - Normalize all metrics?
   - Extract subgraphs of comparable size?
   - Use scale-invariant metrics only?

2. **Semantic grounding**: Pure structural matching might find "false resonances" - structurally similar but semantically meaningless. Do we need any semantic filtering?

3. **Directionality**: Text/music graphs are directed (flow), image graphs are undirected (adjacency). How to handle?
   - Convert all to undirected for comparison?
   - Use separate metrics for directed vs undirected?

4. **Subgraph vs whole graph**: Should we match whole graphs, or find resonant subgraphs within larger structures?

---

## References

- **chain_reflow**: https://github.com/sligara7/chain_reflow - Workflow patterns for graph linking
- **NetworkX motifs**: https://networkx.org/documentation/stable/reference/algorithms/isomorphism.html
- **Graph kernels**: Weisfeiler-Lehman, random walk kernels for graph similarity
- **Spectral graph theory**: Using eigenvalues for structural comparison

---

## Next Steps

1. Review this foundation document
2. Decide on Phase 1 priorities
3. Implement `StructuralSignature` and basic extractor
4. Run first cross-domain experiment with existing DAGs
