# Functional Architecture
## Synesthesia - Structural Rorschach Cross-Domain Analysis Service

**Version:** 1.0.0
**Last Updated:** 2024-11-23
**Status:** Draft

---

## 1. Overview

Synesthesia is a cross-domain structural analysis service that discovers "resonances" between graphs from different domains (images, music, text, code) based on topological structure rather than semantic content.

### Core Principle
> "Structure is meaning" - A flower's radial petal arrangement resonates with a musical chord's harmonic structure because they share *shape*, not meaning.

---

## 2. Functional Flows

### FLOW-001: Cross-Domain Pattern Discovery
**Purpose:** Find structurally similar patterns across domains

```
Load Graph -> Extract Signature -> Load Corpus -> Find Resonances -> Rank -> Explain
```

| Step | Function | Description |
|------|----------|-------------|
| 1 | F-GU-01 | Load graph from JSON file |
| 2 | F-SE-01 | Extract complete structural signature |
| 3 | F-CM-03 | Load target domain corpus |
| 4 | F-RF-01 | Find structurally similar items |
| 5 | F-RF-04 | Rank by relevance |
| 6 | F-IE-01 | Generate human-readable explanation |

### FLOW-002: Educational Demonstration
**Purpose:** Show students how musical structures relate to visual patterns

```
Upload Music -> Convert to Graph -> Extract Signature -> Find Visual Matches -> Explain
```

### FLOW-003: Corpus Building
**Purpose:** Build searchable collections of structural signatures

```
Create Corpus -> Batch Process -> Extract Signatures -> Add to Corpus -> Save
```

### FLOW-004: Code Structure Analysis
**Purpose:** Find structural patterns in code

```
Provide Code -> Convert to AST -> Extract Signature -> Find Patterns
```

---

## 3. Functional Components

### 3.1 Domain Adapters (FR-DA)
Convert domain-specific data into unified graph representations.

| ID | Function | Input | Output |
|----|----------|-------|--------|
| F-DA-01 | Adapt Text to Graph | Text corpus | Graph JSON |
| F-DA-02 | Adapt Image to Graph | Image file | Graph JSON |
| F-DA-03 | Adapt Music to Graph | MIDI/Audio | Graph JSON |
| F-DA-04 | Adapt Code to Graph | Source code | Graph JSON |
| F-DA-05 | Validate Graph Format | Graph JSON | Boolean + Errors |

### 3.2 Signature Extraction (FR-SE)
Extract domain-agnostic structural signatures from graphs.

| ID | Function | Description |
|----|----------|-------------|
| F-SE-01 | Extract Full Signature | Combines all metrics into StructuralSignature |
| F-SE-02 | Extract Degree Metrics | Distribution, entropy, hub ratio |
| F-SE-03 | Extract Clustering Metrics | Communities, modularity, coefficient |
| F-SE-04 | Extract Flow Metrics | Path lengths, diameter, DAG status |
| F-SE-05 | Extract Motif Vector | Normalized motif frequencies |
| F-SE-06 | Extract Centrality Metrics | Gini, betweenness, articulation points |
| F-SE-07 | Extract Spectral Signature | Laplacian eigenvalues |

### 3.3 Motif Detection (FR-MD)
Detect and interpret structural patterns.

| ID | Function | Pattern |
|----|----------|---------|
| F-MD-01 | Detect All Motifs | All supported patterns |
| F-MD-02 | Detect Hub-Spokes | Central node with radial connections |
| F-MD-03 | Detect Chains | Linear sequences |
| F-MD-04 | Detect Triangles | Fully connected triads |
| F-MD-05 | Detect Bridges | Connector nodes between communities |
| F-MD-06 | Detect Cycles | Closed loops |
| F-MD-07 | Get Motif Interpretation | Domain-specific meaning |

### 3.4 Similarity Computation (FR-SC)
Compute structural similarity between signatures.

| ID | Function | Method |
|----|----------|--------|
| F-SC-01 | Compute Overall Similarity | Weighted combination (40% motif, 30% spectral, 30% scale) |
| F-SC-02 | Compute Motif Similarity | Cosine similarity of motif vectors |
| F-SC-03 | Compute Spectral Similarity | Eigenvalue distribution comparison |
| F-SC-04 | Compute Scale Similarity | Node/edge/density comparison |

### 3.5 Resonance Finding (FR-RF)
Find cross-domain structural matches.

| ID | Function | Description |
|----|----------|-------------|
| F-RF-01 | Find Resonances | Search corpus for matches |
| F-RF-02 | Find Cross-Domain Resonances | Search multiple domain corpora |
| F-RF-04 | Rank Resonances | Sort by relevance |

### 3.6 Interpretation Engine (FR-IE)
Generate human-readable explanations.

| ID | Function | Output |
|----|----------|--------|
| F-IE-01 | Explain Resonance | Why two structures resonate |
| F-IE-02 | Explain Cross-Domain Motif | Motif meaning across domains |
| F-IE-03 | Generate Comparison Report | Full structural comparison |

### 3.7 Corpus Management (FR-CM)
Manage collections of signatures.

| ID | Function | Operation |
|----|----------|-----------|
| F-CM-01 | Create Corpus | Create new corpus |
| F-CM-02 | Add to Corpus | Add signature |
| F-CM-03 | Load Corpus | Load from storage |
| F-CM-04 | Save Corpus | Persist to storage |
| F-CM-05 | Index Corpus | Build search index |
| F-CM-06 | Query Corpus | Fast similarity search |

### 3.8 Graph Utilities (FR-GU)
General graph operations.

| ID | Function | Operation |
|----|----------|-----------|
| F-GU-01 | Load Graph | Load from JSON |
| F-GU-02 | Save Graph | Save to JSON |
| F-GU-03 | Normalize Graph | Prepare for comparison |
| F-GU-06 | Validate DAG | Check acyclicity |

---

## 4. Key Data Structures

### StructuralSignature
```
- source_domain: "image" | "music" | "text" | "code"
- source_id: unique identifier
- num_nodes, num_edges, density
- degree_distribution, degree_entropy, hub_ratio
- clustering_coefficient, num_communities, modularity
- avg_path_length, diameter, is_dag
- motif_vector: {hub_spoke: 0.3, chain: 0.2, ...}
- centrality_gini
- spectral_signature: [eigenvalues]
```

### Resonance
```
- query_domain, query_id
- match_domain, match_id
- overall_score, motif_similarity, spectral_similarity, scale_similarity
- matching_motifs: ["hub_spoke", "chain"]
- explanation: "human-readable text"
```

---

## 5. Architecture Summary

| Metric | Value |
|--------|-------|
| Total Functions | 40 |
| Total Dependencies | 37 |
| Entry Points | 7 |
| Exit Points | 5 |
| Is DAG | Yes |

### Entry Points
- F-DA-01 to F-DA-04 (Domain Adapters)
- F-GU-01 (Load Graph)
- F-CM-01 (Create Corpus)
- F-CM-03 (Load Corpus)

### Exit Points
- F-IE-01, F-IE-02, F-IE-03 (Explanations/Reports)
- F-GU-02 (Save Graph)
- F-CM-04 (Save Corpus)

---

## 6. Motif Vocabulary

| Motif | Image | Music | Text |
|-------|-------|-------|------|
| Hub-Spoke | Focal point with radial features | Tonic with harmonic extensions | Central theme with supporting points |
| Chain | Edge, contour line | Melodic phrase | Narrative sequence |
| Triangle | Stable region | Triad chord | Circular reference |
| Fork | Branching structure | Voice split, arpeggio | Enumeration |
| Funnel | Focus point | Resolution | Conclusion |
| Bridge | Transition zone | Modulation | Plot turn |
| Cycle | Symmetry, enclosed area | Ostinato, refrain | Repetition |

---

## 7. Diagrams

See `docs/diagrams/functional/` for:
- `functional_dependencies.mmd` - Full function dependency graph
- `process_flows.mmd` - High-level process flows
- `data_flows.mmd` - Data flow diagram

---

## 8. Validation Status

| Check | Status |
|-------|--------|
| Format Validation | PASS |
| Technical Analysis | PASS (1 issue fixed) |
| Stakeholder Validation | Skipped (personal project) |
| DAG Verification | PASS |
