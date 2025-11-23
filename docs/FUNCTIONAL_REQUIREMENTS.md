# Functional Requirements Document
## Structural Rorschach: Cross-Domain DAG Analysis Service

**Version:** 1.0.0
**Status:** Draft
**Last Updated:** 2024

---

## 1. Overview

### 1.1 Purpose
Define the functional requirements for a cross-domain structural analysis service that finds "resonances" between graphs from different domains (images, music, text) based on their topological structure rather than semantic content.

### 1.2 Scope
This service enables:
- Converting domain-specific data into unified graph representations
- Extracting domain-agnostic structural signatures
- Finding cross-domain structural matches ("resonances")
- Providing interpretable explanations for matches

### 1.3 Inspiration
- **Synesthesia**: Cross-sensory perception where one sense triggers another
- **Rorschach Test**: Interpreting ambiguous patterns through different lenses

---

## 2. Functional Requirements

### 2.1 Domain Adapters (FR-DA)

Convert domain-specific data into unified graph format.

| ID | Function | Input | Output | Description |
|----|----------|-------|--------|-------------|
| FR-DA-01 | `adapt_text_to_graph()` | Text corpus | Graph JSON | Convert text into word transition graph |
| FR-DA-02 | `adapt_image_to_graph()` | Image file | Graph JSON | Convert image into region adjacency graph |
| FR-DA-03 | `adapt_music_to_graph()` | MIDI/Audio file | Graph JSON | Convert music into note transition graph |
| FR-DA-04 | `adapt_code_to_graph()` | Source code | Graph JSON | Convert code into AST/dependency graph |
| FR-DA-05 | `validate_graph_format()` | Graph JSON | Boolean + Errors | Validate against schema |

#### FR-DA-01: adapt_text_to_graph()
```python
def adapt_text_to_graph(
    text: str,
    name: str,
    tokenizer: str = "word",  # "word", "sentence", "character"
    normalize: bool = True
) -> dict:
    """
    Convert text into a word transition graph.

    Nodes: Unique words/tokens
    Edges: Word transitions (word_a -> word_b)
    Edge weights: Transition probability (normalized count)

    Returns: system_of_systems_graph.json format
    """
```

#### FR-DA-02: adapt_image_to_graph()
```python
def adapt_image_to_graph(
    image_path: str,
    name: str,
    n_segments: int = 100,
    mode: str = "rgb"  # "rgb" or "intensity"
) -> dict:
    """
    Convert image into region adjacency graph.

    Nodes: Image regions (superpixels)
    Edges: Spatial adjacency between regions
    Edge weights: Normalized boundary length

    Returns: system_of_systems_graph.json format
    """
```

#### FR-DA-03: adapt_music_to_graph()
```python
def adapt_music_to_graph(
    audio_path: str,
    name: str,
    source_type: str = "midi"  # "midi", "audio"
) -> dict:
    """
    Convert music into note transition graph.

    Nodes: Musical notes/events
    Edges: Note transitions (temporal sequence)
    Edge weights: Transition probability

    Returns: system_of_systems_graph.json format
    """
```

---

### 2.2 Signature Extraction (FR-SE)

Extract domain-agnostic structural signatures from graphs.

| ID | Function | Input | Output | Description |
|----|----------|-------|--------|-------------|
| FR-SE-01 | `extract_signature()` | Graph | StructuralSignature | Extract full signature |
| FR-SE-02 | `extract_degree_metrics()` | Graph | DegreeMetrics | Degree distribution, entropy, hub ratio |
| FR-SE-03 | `extract_clustering_metrics()` | Graph | ClusteringMetrics | Communities, modularity, coefficient |
| FR-SE-04 | `extract_flow_metrics()` | Graph | FlowMetrics | Path lengths, diameter, DAG status |
| FR-SE-05 | `extract_motif_vector()` | Graph | MotifVector | Normalized motif frequencies |
| FR-SE-06 | `extract_centrality_metrics()` | Graph | CentralityMetrics | Gini, betweenness, articulation points |
| FR-SE-07 | `extract_spectral_signature()` | Graph | List[float] | Laplacian eigenvalues |

#### FR-SE-01: extract_signature()
```python
def extract_signature(
    graph: Union[nx.Graph, dict],
    domain: str,
    name: str,
    source_id: str
) -> StructuralSignature:
    """
    Extract complete domain-agnostic structural signature.

    Combines all metric extractors into unified signature
    that can be compared across domains.

    Returns: StructuralSignature dataclass
    """
```

#### FR-SE-05: extract_motif_vector()
```python
def extract_motif_vector(
    graph: nx.Graph,
    normalize: bool = True
) -> Dict[str, float]:
    """
    Extract normalized motif frequency vector.

    Motifs detected:
    - hub_spoke: Central node with radial connections
    - chain: Linear sequence
    - triangle: Fully connected triad
    - fork: One-to-many divergence
    - funnel: Many-to-one convergence
    - cycle: Closed loop

    Returns: {motif_name: normalized_frequency}
    """
```

---

### 2.3 Motif Detection (FR-MD)

Detect and interpret structural patterns.

| ID | Function | Input | Output | Description |
|----|----------|-------|--------|-------------|
| FR-MD-01 | `detect_motifs()` | Graph | Dict[str, List[Match]] | Detect all motif instances |
| FR-MD-02 | `detect_hub_spokes()` | Graph | List[MotifMatch] | Find hub-spoke patterns |
| FR-MD-03 | `detect_chains()` | Graph | List[MotifMatch] | Find chain patterns |
| FR-MD-04 | `detect_triangles()` | Graph | List[MotifMatch] | Find triangle patterns |
| FR-MD-05 | `detect_bridges()` | Graph | List[MotifMatch] | Find bridge/connector nodes |
| FR-MD-06 | `detect_cycles()` | Graph | List[MotifMatch] | Find cycle patterns |
| FR-MD-07 | `get_motif_interpretation()` | Motif, Domain | str | Get domain-specific interpretation |

#### FR-MD-01: detect_motifs()
```python
def detect_motifs(
    graph: nx.Graph,
    motif_types: List[str] = None  # None = all types
) -> Dict[str, List[MotifMatch]]:
    """
    Detect all motif instances in the graph.

    Returns dictionary mapping motif type to list of matches,
    where each match includes the nodes and edges involved.
    """
```

#### FR-MD-07: get_motif_interpretation()
```python
def get_motif_interpretation(
    motif_type: str,
    domain: str
) -> str:
    """
    Get human-readable interpretation of motif in domain context.

    Example:
        get_motif_interpretation("hub_spoke", "music")
        -> "Tonic/root note with harmonic extensions"

        get_motif_interpretation("hub_spoke", "image")
        -> "Focal point with radial features"
    """
```

---

### 2.4 Similarity Computation (FR-SC)

Compute structural similarity between signatures.

| ID | Function | Input | Output | Description |
|----|----------|-------|--------|-------------|
| FR-SC-01 | `compute_similarity()` | Sig1, Sig2 | SimilarityResult | Overall structural similarity |
| FR-SC-02 | `compute_motif_similarity()` | Vec1, Vec2 | float | Motif vector cosine similarity |
| FR-SC-03 | `compute_spectral_similarity()` | Spec1, Spec2 | float | Spectral signature similarity |
| FR-SC-04 | `compute_scale_similarity()` | Sig1, Sig2 | float | Scale/density similarity |
| FR-SC-05 | `compute_graph_edit_distance()` | G1, G2 | int | Approximate edit distance |

#### FR-SC-01: compute_similarity()
```python
def compute_similarity(
    sig1: StructuralSignature,
    sig2: StructuralSignature,
    weights: Dict[str, float] = None
) -> SimilarityResult:
    """
    Compute overall structural similarity between two signatures.

    Default weights:
    - motif_similarity: 0.4
    - spectral_similarity: 0.3
    - scale_similarity: 0.3

    Returns: SimilarityResult with overall score and component scores
    """
```

#### FR-SC-02: compute_motif_similarity()
```python
def compute_motif_similarity(
    vec1: Dict[str, float],
    vec2: Dict[str, float]
) -> float:
    """
    Compute cosine similarity between motif vectors.

    Returns: Similarity score [0, 1]
    """
```

---

### 2.5 Resonance Finding (FR-RF)

Find cross-domain structural matches.

| ID | Function | Input | Output | Description |
|----|----------|-------|--------|-------------|
| FR-RF-01 | `find_resonances()` | Query, Corpus | List[Resonance] | Find structurally similar items |
| FR-RF-02 | `find_cross_domain_resonances()` | Query, Domains | Dict[str, List[Resonance]] | Find matches across multiple domains |
| FR-RF-03 | `find_subgraph_resonances()` | Query, Target | List[SubgraphMatch] | Find resonant subgraphs |
| FR-RF-04 | `rank_resonances()` | List[Resonance] | List[Resonance] | Rank by relevance |

#### FR-RF-01: find_resonances()
```python
def find_resonances(
    query: StructuralSignature,
    corpus: List[StructuralSignature],
    top_k: int = 10,
    min_similarity: float = 0.5
) -> List[Resonance]:
    """
    Find structurally similar items in a corpus.

    Args:
        query: The "inkblot" - structure to match against
        corpus: Collection of signatures to search
        top_k: Maximum number of results
        min_similarity: Minimum similarity threshold

    Returns: List of Resonance objects sorted by similarity
    """
```

#### FR-RF-02: find_cross_domain_resonances()
```python
def find_cross_domain_resonances(
    query: StructuralSignature,
    domain_corpora: Dict[str, List[StructuralSignature]],
    top_k_per_domain: int = 5
) -> Dict[str, List[Resonance]]:
    """
    Find resonances across multiple domains.

    Example:
        query = image_signature  # flower image
        domains = {"music": music_corpus, "text": text_corpus}

        Returns:
        {
            "music": [resonance_with_bach, resonance_with_mozart, ...],
            "text": [resonance_with_psalm, resonance_with_john, ...]
        }
    """
```

---

### 2.6 Interpretation Engine (FR-IE)

Generate human-readable explanations.

| ID | Function | Input | Output | Description |
|----|----------|-------|--------|-------------|
| FR-IE-01 | `explain_resonance()` | Resonance | str | Explain why two structures resonate |
| FR-IE-02 | `explain_motif_match()` | Motif, Domain1, Domain2 | str | Explain cross-domain motif meaning |
| FR-IE-03 | `generate_comparison_report()` | Sig1, Sig2 | str | Full comparison report |
| FR-IE-04 | `visualize_resonance()` | Resonance | Image | Visual comparison |

#### FR-IE-01: explain_resonance()
```python
def explain_resonance(
    resonance: Resonance,
    detail_level: str = "medium"  # "brief", "medium", "detailed"
) -> str:
    """
    Generate human-readable explanation for a resonance.

    Example output:
    "The flower image's radial petal arrangement (hub-spoke topology)
     resonates with Bach's Prelude No. 1, where the C major tonic acts
     as a hub connecting to harmonic extensions. Both structures exhibit
     high centrality concentration (0.73 vs 0.68) and similar clustering
     patterns."
    """
```

---

### 2.7 Corpus Management (FR-CM)

Manage collections of signatures for fast lookup.

| ID | Function | Input | Output | Description |
|----|----------|-------|--------|-------------|
| FR-CM-01 | `create_corpus()` | Name, Domain | Corpus | Create new signature corpus |
| FR-CM-02 | `add_to_corpus()` | Corpus, Signature | bool | Add signature to corpus |
| FR-CM-03 | `load_corpus()` | Path | Corpus | Load corpus from file |
| FR-CM-04 | `save_corpus()` | Corpus, Path | bool | Save corpus to file |
| FR-CM-05 | `index_corpus()` | Corpus | Index | Build search index for fast lookup |
| FR-CM-06 | `query_corpus()` | Corpus, Query | List[Sig] | Fast similarity search |

#### FR-CM-01: create_corpus()
```python
def create_corpus(
    name: str,
    domain: str,
    description: str = ""
) -> Corpus:
    """
    Create a new corpus for storing signatures.

    A corpus is a collection of StructuralSignatures from a single
    domain, optimized for fast similarity search.
    """
```

---

### 2.8 Graph Utilities (FR-GU)

General graph operations.

| ID | Function | Input | Output | Description |
|----|----------|-------|--------|-------------|
| FR-GU-01 | `load_graph()` | Path | nx.Graph | Load graph from JSON |
| FR-GU-02 | `save_graph()` | Graph, Path | bool | Save graph to JSON |
| FR-GU-03 | `normalize_graph()` | Graph | Graph | Normalize for comparison |
| FR-GU-04 | `extract_subgraph()` | Graph, Nodes | Graph | Extract induced subgraph |
| FR-GU-05 | `merge_graphs()` | List[Graph] | Graph | Merge multiple graphs |
| FR-GU-06 | `validate_dag()` | Graph | bool | Check if graph is DAG |

---

## 3. Data Structures

### 3.1 StructuralSignature
```python
@dataclass
class StructuralSignature:
    # Identity
    source_domain: str          # "image", "music", "text"
    source_id: str              # Unique identifier
    source_name: str            # Human-readable name

    # Scale
    num_nodes: int
    num_edges: int
    density: float
    is_directed: bool

    # Degree patterns
    degree_distribution: List[float]
    degree_entropy: float
    hub_ratio: float

    # Clustering
    clustering_coefficient: float
    num_communities: int
    modularity: float

    # Flow
    avg_path_length: float
    diameter: int
    is_dag: bool

    # Motifs
    motif_vector: Dict[str, float]

    # Centrality
    centrality_gini: float

    # Spectral
    spectral_signature: List[float]
```

### 3.2 Resonance
```python
@dataclass
class Resonance:
    query_domain: str
    query_id: str
    query_name: str

    match_domain: str
    match_id: str
    match_name: str

    overall_score: float
    motif_similarity: float
    spectral_similarity: float
    scale_similarity: float

    matching_motifs: List[str]
    shared_properties: Dict[str, float]
    explanation: str
```

### 3.3 MotifMatch
```python
@dataclass
class MotifMatch:
    motif_type: str
    nodes: Tuple[str, ...]
    edges: List[Tuple[str, str]]
    central_node: Optional[str]
```

---

## 4. API Endpoints (Future)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/extract` | POST | Extract signature from uploaded file |
| `/compare` | POST | Compare two signatures |
| `/search` | POST | Search corpus for resonances |
| `/corpus` | GET/POST | Manage corpora |
| `/interpret` | POST | Generate explanation for match |

---

## 5. Integration with chain_reflow

The service integrates with chain_reflow workflow patterns:

| chain_reflow Strategy | Application |
|-----------------------|-------------|
| HIERARCHICAL | Domain → Sub-domain relationships |
| PAIRWISE | Direct domain-to-domain comparisons |
| NETWORK | Multi-domain mesh comparisons |
| PAIRWISE_WITH_INTERMEDIARIES | Use signatures as bridge structures |

---

## 6. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-01 | Signature extraction should complete in < 30s for graphs up to 10,000 nodes |
| NFR-02 | Similarity computation should complete in < 100ms |
| NFR-03 | Corpus search should return results in < 1s for 10,000 signatures |
| NFR-04 | All functions should be thread-safe |
| NFR-05 | Service should support batch processing |

---

## 7. Future Enhancements

| ID | Enhancement |
|----|-------------|
| FE-01 | GPU acceleration for large graph analysis |
| FE-02 | Streaming signature extraction for very large graphs |
| FE-03 | ML-based similarity learning (fine-tune on user feedback) |
| FE-04 | Real-time cross-domain generation (image → music in real-time) |
| FE-05 | Web UI for interactive exploration |

---

## 8. Acceptance Criteria

1. All FR-* functions implemented and unit tested
2. Cross-domain matching works between text, image, and music domains
3. Resonance explanations are interpretable by non-technical users
4. Integration with existing wordplay DAG builders verified
5. Documentation complete with examples
