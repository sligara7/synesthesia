# Novel Research Directions for Synesthesia

**Status**: Research Proposal
**Date**: 2024-11-23
**Focus**: Cross-Domain Structural Transfer (not just matching)

---

## The Gap We're Filling

### What Exists (Not Novel)
- **NetSimile (2012)**: Graph fingerprints + similarity matching
- **Graph Kernels**: Weisfeiler-Lehman, random walk kernels
- **Graph Neural Networks**: Node/graph classification, link prediction
- **Cross-Modal Generation**: Art2Mus, Mozart's Touch (semantic/emotional transfer)

### What's Missing (Our Contribution)
**Pure structural cross-domain transfer**: Preserving graph *topology* across domains without relying on semantic interpretation.

> "What SHAPE does a dragon have that could be expressed as music?"
>
> Not: "What does a dragon MEAN emotionally in music?"

---

## Three Novel Research Directions

### Direction 1: Graph Analogies (A × B = C)

**Inspiration**: Word2Vec's `king - man + woman = queen`

**Core Idea**: Learn transformation operators in graph embedding space.

```
G_dragon_image : G_dragon_music :: G_flower_image : G_flower_music

Solve via:
G_flower_music = G_flower_image + (G_dragon_music - G_dragon_image)
                                   └──────────────┬──────────────┘
                                          "domain offset" T
```

**Mathematical Formulation**:

Let `E: Graph → ℝ^d` be an encoder mapping graphs to embeddings.

```
E(G_target) = E(G_source) + T

where T = E(G_exemplar_target) - E(G_exemplar_source)
```

The transformation `T` captures the "structural translation" between domains.

**Training Approach**:

1. **Paired Examples**: Collect (image_structure, music_structure) pairs that humans judge as "structurally similar"
2. **Contrastive Learning**: Learn embeddings where structurally similar cross-domain pairs are close
3. **Linear Probing**: Test if domain transfer is linear (as in word2vec)

**Key Questions**:
- Is the transformation linear or does it require nonlinear mapping?
- What structural properties transfer vs. transform?
- Can we learn T from few examples (few-shot)?

**Novelty Assessment**: ★★★★★
- Analogical reasoning has NOT been applied to graph structures
- If linear transfer works, it's a significant finding
- Could enable zero-shot cross-domain generation

---

### Direction 2: Structural Synesthesia VAE

**Core Idea**: Variational autoencoder with domain-agnostic structural latent space.

```
┌─────────────┐     ┌─────────────────────┐     ┌─────────────┐
│ G_image     │ ──▶ │ Encoder             │ ──▶ │ z_structure │
└─────────────┘     │ (domain-agnostic)   │     │ (shared)    │
                    └─────────────────────┘     └──────┬──────┘
                                                       │
                                                       ▼
┌─────────────┐     ┌─────────────────────┐     ┌─────────────┐
│ G_music_new │ ◀── │ Decoder             │ ◀── │ z + domain  │
└─────────────┘     │ (domain-conditioned)│     │ "music"     │
                    └─────────────────────┘     └─────────────┘
```

**Mathematical Formulation**:

```
Encoder: q(z | G, domain) ≈ q(z | G)  # domain-invariant
Decoder: p(G | z, domain)              # domain-specific

Loss = Reconstruction + KL + Structural_Fidelity

where:
  Structural_Fidelity = ||σ(G_input) - σ(G_output)||²
  σ(G) = structural signature (motifs, spectral, etc.)
```

**Architecture**:

1. **Encoder**: Graph Neural Network (GIN/GAT) → pooling → z
2. **Latent Space**: Regularized to capture structure, not content
3. **Decoder**: Autoregressive graph generator (like GraphRNN) conditioned on domain

**Key Questions**:
- How do we ensure latent space captures *structure* not *content*?
- What's the right structural fidelity loss?
- Can we disentangle structure from domain-specific features?

**Novelty Assessment**: ★★★★
- VAEs for graphs exist, but not for cross-domain structural transfer
- Explicit structural fidelity loss is novel
- Enables true generation, not just matching

---

### Direction 3: Multi-Scale Resonance Matching

**Core Idea**: Match structures at multiple granularities, enabling richer cross-domain mappings.

```
┌─────────────────────────────────────────────────────────────┐
│ Level 1: GLOBAL                                             │
│   Overall graph properties (density, diameter, etc.)        │
│   "Dragon image has 25 nodes, hub-spoke structure"          │
│   "Symphony has 20 nodes, hub-spoke structure"              │
│   Match: 95%                                                │
├─────────────────────────────────────────────────────────────┤
│ Level 2: MESO (Communities/Subgraphs)                       │
│   Community structure, subgraph patterns                    │
│   "Dragon wings = 2 radiating communities"                  │
│   "Symphony movements = 3 connected communities"            │
│   Match: 78%                                                │
├─────────────────────────────────────────────────────────────┤
│ Level 3: LOCAL (Motifs)                                     │
│   Motif frequencies, local patterns                         │
│   "Dragon fire = chain motif (cascading)"                   │
│   "Symphony crescendo = chain motif (building)"             │
│   Match: 92%                                                │
└─────────────────────────────────────────────────────────────┘
```

**Mathematical Formulation**:

```
Similarity(G1, G2) = Σ_l  w_l × sim_l(G1, G2)

where l ∈ {global, meso, local}

sim_global = cosine(σ_global(G1), σ_global(G2))
sim_meso = alignment_score(communities(G1), communities(G2))
sim_local = cosine(motif_vector(G1), motif_vector(G2))
```

**Key Components**:

1. **Global Features**: Spectral signature, degree distribution, diameter
2. **Meso Features**: Community detection, modularity, subgraph census
3. **Local Features**: Motif counts, clustering coefficients, centrality

**Alignment at Meso Level**:
- Use optimal transport to match communities between domains
- Wasserstein distance between community distributions

**Key Questions**:
- What's the right weighting between scales?
- How do we align substructures across domains?
- Can we learn scale-specific transformations?

**Novelty Assessment**: ★★★
- Multi-scale graph analysis exists, but not for cross-domain matching
- Hierarchical resonance is a novel framing
- More interpretable than single-vector approaches

---

## Comparison of Directions

| Aspect | Graph Analogies | VAE | Multi-Scale |
|--------|----------------|-----|-------------|
| **Core Operation** | Vector arithmetic | Generation | Hierarchical matching |
| **Output** | Transformed embedding | New graph | Similarity scores |
| **Training Data** | Paired examples | Paired examples | None (or few) |
| **Novelty** | ★★★★★ | ★★★★ | ★★★ |
| **Effort** | Medium | High | Low |
| **Interpretability** | Medium | Low | High |
| **A×B=C?** | Yes, directly | Implicit | No |

---

## Recommended Research Plan

### Phase 1: Graph Analogies (Primary Focus)

1. **Week 1-2**: Implement graph encoder (GNN-based)
2. **Week 3-4**: Collect/synthesize paired training data
3. **Week 5-6**: Train contrastive model, test linear transfer hypothesis
4. **Week 7-8**: Evaluate on held-out domain pairs

**Success Metric**: Can we predict G_music from G_image + T with >70% structural fidelity?

### Phase 2: Multi-Scale Extension

1. Add hierarchical features to encoder
2. Test if transfer works better at different scales
3. Enable "partial matching" (wings match crescendo, body matches drone)

### Phase 3: Generative Extension (VAE)

1. Add decoder to generate actual graphs
2. Condition on domain
3. Evaluate generation quality

---

## Key Hypotheses to Test

### H1: Linear Transfer Hypothesis
> Cross-domain structural transfer can be approximated by linear transformation in embedding space.

**Test**: Compute T from exemplar pair, apply to new source, measure structural similarity to ground truth.

### H2: Structural Invariance Hypothesis
> Core structural properties (motif distribution, spectral signature) are preserved under domain transfer.

**Test**: Compare σ(G_source) with σ(G_generated) across different structural metrics.

### H3: Scale-Specific Transfer Hypothesis
> Some structural properties transfer better at global scale, others at local scale.

**Test**: Ablate different scale features and measure transfer quality.

---

## Potential Publications

### Paper 1: "Graph Analogies: Cross-Domain Structural Transfer via Embedding Arithmetic"
- **Venue**: ICML, NeurIPS, ICLR
- **Contribution**: First application of analogical reasoning to graph structures

### Paper 2: "Structural Synesthesia: A Framework for Topology-Preserving Cross-Domain Graph Generation"
- **Venue**: KDD, WWW, AAAI
- **Contribution**: VAE with explicit structural fidelity loss

### Paper 3: "Multi-Scale Resonance: Hierarchical Cross-Domain Graph Matching"
- **Venue**: CIKM, WSDM
- **Contribution**: Hierarchical structural similarity with interpretable mappings

---

## Next Steps

1. **Prototype Graph Analogies** with simple GNN encoder
2. **Create synthetic paired data** (same structure, different domains)
3. **Test linear transfer hypothesis**
4. **Iterate based on findings**
