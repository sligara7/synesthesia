# User Scenarios
## Synesthesia: Structural Rorschach Cross-Domain Analysis Service

**Version:** 1.0.0
**Created:** 2024

---

## User Personas

### 1. Research Scientist (Dr. Maya)
- **Background**: Computational neuroscientist studying brain connectivity patterns
- **Goal**: Find structural similarities between neural networks and social networks
- **Technical Level**: High - comfortable with Python, graph theory
- **Key Need**: Programmatic API access, batch processing

### 2. Music Educator (Marcus)
- **Background**: Music theory teacher exploring visual learning
- **Goal**: Show students how musical structures relate to visual patterns
- **Technical Level**: Medium - uses software tools but not a programmer
- **Key Need**: Clear explanations, visual outputs

### 3. Data Analyst (Sarah)
- **Background**: Business analyst looking for cross-domain insights
- **Goal**: Find patterns in organizational data that map to known successful structures
- **Technical Level**: Medium - SQL, basic Python
- **Key Need**: Pre-built domain adapters, interpretable results

### 4. Digital Artist (Alex)
- **Background**: Generative artist creating multi-modal works
- **Goal**: Generate art inspired by the "structure" of music
- **Technical Level**: High - creative coder
- **Key Need**: Real-time processing, flexible inputs

---

## Use Cases

### UC-01: Cross-Domain Pattern Discovery
**Actor**: Research Scientist
**Precondition**: Has graph data from two different domains
**Main Flow**:
1. User loads source graph (e.g., neural connectivity data)
2. System extracts structural signature
3. User queries against corpus of target domain (e.g., social networks)
4. System returns ranked resonances with explanations
5. User explores matching structures

**Success Criteria**: Returns meaningful matches with similarity > 0.5

---

### UC-02: Educational Demonstration
**Actor**: Music Educator
**Precondition**: Has sheet music or audio file
**Main Flow**:
1. User uploads musical piece
2. System converts to note transition graph
3. User requests visual domain matches
4. System shows structurally similar image patterns
5. User uses visual to explain musical concept

**Success Criteria**: Student comprehension improves

---

### UC-03: Signature Extraction & Storage
**Actor**: Data Analyst
**Precondition**: Has collection of graphs to index
**Main Flow**:
1. User creates new corpus for domain
2. User batch-processes graphs through adapter
3. System extracts and stores signatures
4. User can later query against this corpus

**Success Criteria**: < 30s per graph, corpus persists

---

### UC-04: Real-time Resonance Finding
**Actor**: Digital Artist
**Precondition**: Has live data stream
**Main Flow**:
1. User connects live audio stream
2. System continuously extracts signatures
3. User views real-time visual resonances
4. Artist incorporates into live performance

**Success Criteria**: < 1s latency end-to-end

---

### UC-05: Code Structure Analysis
**Actor**: Research Scientist
**Precondition**: Has source code repository
**Main Flow**:
1. User provides code repository path
2. System converts to AST/dependency graph
3. User queries for structural patterns
4. System identifies code regions matching known anti-patterns or good patterns

**Success Criteria**: Identifies structural issues in code

---

## User Journey Map

```
Discovery → Adaptation → Extraction → Comparison → Interpretation → Action
    |           |            |            |             |            |
    v           v            v            v             v            v
 "What can   "Convert    "Extract     "Find       "Understand   "Apply
  I learn    my data     signature"   matches"     why they     insight"
  across                                           resonate"
  domains?"
```

---

## Accessibility Considerations

- **Visual learners**: Graph and motif visualizations
- **Auditory learners**: Sonification of structural patterns (future)
- **Kinesthetic learners**: Interactive exploration (future)
- **API access**: For users who prefer programmatic interaction
- **Explanation levels**: Brief, medium, detailed explanations
