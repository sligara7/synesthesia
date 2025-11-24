# Mathematical Foundations of Synesthesia

## The User's Intuition

You're thinking of a transformation like:

```
A × B = C

where:
  A = Dragon graph (source domain)
  B = Translation matrix
  C = Music graph (target domain)
```

This is a natural way to think about it! But our current system works differently. Let me explain both what we do and what a true transformation would require.

---

## What We Actually Do: Retrieval, Not Transformation

### Current Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Dragon      │     │  Feature     │     │  Signature  │
│ Graph (A)   │ ──▶ │  Extraction  │ ──▶ │  Vector σ_A │
└─────────────┘     │  f(·)        │     └──────┬──────┘
                    └──────────────┘            │
                                                │ similarity
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Music       │     │  Feature     │     │  Signature  │
│ Corpus {C}  │ ──▶ │  Extraction  │ ──▶ │  Vectors    │ ──▶ Best Match
└─────────────┘     │  f(·)        │     │  {σ_C}      │
                    └──────────────┘     └─────────────┘
```

We're **matching**, not **transforming**.

### Step 1: Graph to Adjacency Matrix

A graph G with n nodes becomes an adjacency matrix A ∈ {0,1}^(n×n):

```
Dragon graph (simplified):       Adjacency Matrix:
                                      body wing fire tail
    wing                    body  [   0    1    1    1  ]
      ↑                     wing  [   0    0    0    0  ]
    body ──▶ fire           fire  [   0    0    0    0  ]
      ↓                     tail  [   0    0    0    0  ]
    tail
```

### Step 2: Feature Extraction (The Signature)

We extract a **feature vector** σ(G) that captures topology:

```python
σ(G) = [
    # Motif vector (normalized pattern counts)
    m(G) = [star_3_freq, chain_3_freq, triangle_freq, fork_freq, ...],

    # Spectral signature (Laplacian eigenvalues)
    λ(G) = [λ_1, λ_2, ..., λ_k],

    # Scale features
    s(G) = [n_nodes, n_edges, density, clustering_coeff, ...],

    # Degree distribution (histogram)
    d(G) = [bin_0, bin_1, ..., bin_9]
]
```

#### Motif Vector Computation

For each motif pattern, count occurrences and normalize:

```
m_i(G) = count(motif_i in G) / n_nodes
```

#### Spectral Signature

The Laplacian matrix L = D - A, where D is the degree matrix.

```
L = D - A

For our dragon example:
D = diag(3, 0, 0, 0)  # body has degree 3, others have 0
A = adjacency matrix above

L = [ 3  -1  -1  -1]
    [-1   0   0   0]  (for undirected version)
    [-1   0   0   0]
    [-1   0   0   0]

Eigenvalues: λ = [0, 1, 1, 4]  ← This IS the spectral signature
```

The eigenvalue distribution reveals structure:
- λ_2 (Fiedler value) indicates connectivity
- Eigenvalue gaps reveal clusters
- Distribution shape indicates graph type

### Step 3: Similarity Computation

For two graphs G_1 and G_2:

#### Motif Similarity (Cosine)
```
sim_motif = (m₁ · m₂) / (||m₁|| × ||m₂||)

         = Σᵢ m₁ᵢ × m₂ᵢ / √(Σᵢ m₁ᵢ²) × √(Σᵢ m₂ᵢ²)
```

#### Spectral Similarity
```
sim_spectral = 1 - ||λ₁ - λ₂||₂ / max_distance
```

#### Overall Similarity
```
sim = w₁ × sim_motif + w₂ × sim_spectral + w₃ × sim_scale
```

### Step 4: Retrieval

```
match = argmax_{C ∈ corpus} sim(σ(Dragon), σ(C))
```

This finds the music structure with the most similar topology.

---

## What a True Transformation Would Look Like

Your intuition about A × B = C is actually closer to what a **generative** system would do.

### Option 1: Graph-to-Graph Neural Network

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Dragon      │     │   Encoder   │     │  Transform  │     │   Decoder   │
│ Graph A     │ ──▶ │   E(A)      │ ──▶ │   T(z)      │ ──▶ │   D(z')     │ ──▶ Music Graph C
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                          ↓                   ↓                   ↓
                    latent z_A          latent z_C           generated C
```

The transformation matrix B is learned:
```
z_C = B × z_A  (in latent space)
```

This requires **training data**: paired examples of (image structure, music structure).

### Option 2: Optimal Transport

Find the optimal "transport plan" π to move mass from A to C:

```
min_π Σᵢⱼ πᵢⱼ × cost(i,j)

subject to:
  Σⱼ πᵢⱼ = distribution_A(i)  (row sums)
  Σᵢ πᵢⱼ = distribution_C(j)  (column sums)
```

The Wasserstein distance provides a meaningful metric between distributions.

### Option 3: Spectral Graph Matching

If we decompose both graphs:
```
A = V_A × Λ_A × V_A^T  (eigendecomposition)
C = V_C × Λ_C × V_C^T
```

Transformation via eigenvalue transplant:
```
C' = V_C × Λ_A × V_C^T
```

This creates a graph with C's structure but A's "energy distribution".

---

## The Gap: What's Missing

### Current System (Retrieval)
```
Dragon ──▶ σ(Dragon) ──similarity──▶ σ(Music) ──▶ Music (pre-existing)
```

We find existing structures that match, we don't create new ones.

### True Transformation (Generation)
```
Dragon ──▶ z_Dragon ──▶ B × z_Dragon ──▶ z_Music ──▶ NEW Music Structure
```

This would generate novel music structures that preserve dragon topology.

### The Translation Matrix B

For true A × B = C, matrix B would encode:

```
B: maps features of domain A → features of domain C

B might encode:
  - "Hub in image" → "Chorus in music"
  - "Flow pattern" → "Verse sequence"
  - "Intensity gradient" → "Dynamic crescendo"
```

This B would need to be:
1. **Learned** from paired training data, OR
2. **Hand-crafted** based on domain knowledge

---

## Mathematical Summary

| Aspect | Current System | True Transformation |
|--------|---------------|---------------------|
| Operation | Retrieval | Generation |
| Math | sim(σ_A, σ_C) | C = f(A, B) |
| Output | Existing structure | Novel structure |
| Training | Not needed | Required |
| Matrix B | Not used | Learned/crafted |

### Current Equations

```
σ(G) = [m(G), λ(G), s(G), d(G)]           # Feature extraction

sim(G₁, G₂) = Σᵢ wᵢ × simᵢ(σ(G₁), σ(G₂))  # Similarity

match = argmax_C sim(σ(Query), σ(C))       # Retrieval
```

### What's Needed for True A × B = C

```
z_A = Encoder(A)                           # Encode source
z_C = Transform(z_A; B)                    # Transform (B is learned)
C = Decoder(z_C)                           # Decode to target domain
```

---

## Conclusion

Your intuition is mathematically sound! The A × B = C model is exactly what a **generative cross-domain translation** system would do.

Our current system is a **retrieval** system:
- We don't transform dragon → music
- We find which existing music has the same shape as the dragon

To build true transformation:
1. Need paired training data (image structures ↔ music structures)
2. Learn the transformation matrix B in latent space
3. Train encoder/decoder for each domain

This is an exciting future direction for Synesthesia!
