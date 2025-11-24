"""
Integration tests for Synesthesia - Full workflow testing

Tests the complete workflow from data adaptation through resonance finding.
"""

import pytest
import tempfile
import os
import sys
sys.path.insert(0, 'src')

import networkx as nx
from structural_rorschach import (
    # Container
    create_service_container,
    ServiceContainer,
    # Core types
    StructuralSignature,
    Corpus,
    Resonance,
    # Services
    SignatureExtractor,
    # Adapters
    TextAdapter,
)


@pytest.fixture
def container():
    """Create a fresh service container for each test."""
    return create_service_container()


@pytest.fixture
def sample_graph_1():
    """Create a sample graph representing a hub-spoke pattern."""
    G = nx.DiGraph()
    # Hub node connected to 5 spokes
    G.add_node("hub", label="central")
    for i in range(5):
        G.add_node(f"spoke_{i}", label=f"peripheral_{i}")
        G.add_edge("hub", f"spoke_{i}")
    return G


@pytest.fixture
def sample_graph_2():
    """Create a sample graph representing a chain pattern."""
    G = nx.DiGraph()
    # Linear chain of 6 nodes
    for i in range(6):
        G.add_node(f"node_{i}", label=f"step_{i}")
        if i > 0:
            G.add_edge(f"node_{i-1}", f"node_{i}")
    return G


@pytest.fixture
def sample_graph_3():
    """Create a graph similar to sample_graph_1 (hub-spoke)."""
    G = nx.DiGraph()
    # Another hub-spoke pattern (should be similar to graph_1)
    G.add_node("center", label="main")
    for i in range(4):
        G.add_node(f"leaf_{i}", label=f"child_{i}")
        G.add_edge("center", f"leaf_{i}")
    return G


class TestFullWorkflow:
    """Integration tests for the complete Synesthesia workflow."""

    def test_signature_extraction_from_graph(self, sample_graph_1):
        """Test extracting a signature from a NetworkX graph."""
        extractor = SignatureExtractor()
        signature = extractor.extract_from_networkx(
            sample_graph_1,
            domain="test",
            name="Hub-Spoke Graph",
            source_id="graph_1"
        )

        assert isinstance(signature, StructuralSignature)
        assert signature.source_domain == "test"
        assert signature.source_name == "Hub-Spoke Graph"
        assert signature.num_nodes == 6  # hub + 5 spokes
        assert signature.num_edges == 5

    def test_corpus_workflow(self, container, sample_graph_1, sample_graph_2, sample_graph_3):
        """Test creating corpus, adding signatures, and querying."""
        extractor = container.signature_extractor
        corpus_service = container.corpus_service

        # Create corpus
        corpus = corpus_service.create_corpus(
            name="test_corpus",
            domain="test",
            description="Integration test corpus"
        )

        # Extract and add signatures
        sig1 = extractor.extract_from_networkx(sample_graph_1, "test", "Hub-Spoke 1", "sig_1")
        sig2 = extractor.extract_from_networkx(sample_graph_2, "test", "Chain", "sig_2")
        sig3 = extractor.extract_from_networkx(sample_graph_3, "test", "Hub-Spoke 2", "sig_3")

        corpus_service.add_signature(corpus, sig1)
        corpus_service.add_signature(corpus, sig2)
        corpus_service.add_signature(corpus, sig3)

        assert len(corpus) == 3

        # Query corpus with sig3 (hub-spoke) - should match sig1 better than sig2
        results = corpus_service.query(corpus, sig3, top_k=2)

        assert len(results) == 2
        # First result should be sig1 (similar hub-spoke structure)
        assert results[0][0].source_name == "Hub-Spoke 1"

    def test_resonance_finding(self, container, sample_graph_1, sample_graph_2, sample_graph_3):
        """Test finding resonances between structures."""
        extractor = container.signature_extractor
        corpus_service = container.corpus_service
        resonance_service = container.resonance_service

        # Create corpus with different structures
        corpus = corpus_service.create_corpus("resonance_test", "test")

        sig1 = extractor.extract_from_networkx(sample_graph_1, "test", "Hub-Spoke 1", "res_1")
        sig2 = extractor.extract_from_networkx(sample_graph_2, "test", "Chain", "res_2")

        corpus_service.add_signature(corpus, sig1)
        corpus_service.add_signature(corpus, sig2)

        # Query with similar hub-spoke structure
        query_sig = extractor.extract_from_networkx(sample_graph_3, "test", "Query Hub-Spoke", "query")

        resonances = resonance_service.find_resonances(
            query_sig, corpus, top_k=2, threshold=0.0
        )

        assert len(resonances) == 2
        assert isinstance(resonances[0], Resonance)
        # Hub-spoke should resonate more with hub-spoke than with chain
        assert resonances[0].match_name == "Hub-Spoke 1"

    def test_interpretation_of_resonance(self, container, sample_graph_1, sample_graph_3):
        """Test generating human-readable explanations."""
        extractor = container.signature_extractor
        interpretation_service = container.interpretation_service
        similarity_service = container.similarity_service

        sig1 = extractor.extract_from_networkx(sample_graph_1, "test", "Structure A", "interp_1")
        sig2 = extractor.extract_from_networkx(sample_graph_3, "test", "Structure B", "interp_2")

        # Compute similarity
        sim_result = similarity_service.compute_similarity(sig1, sig2)

        # Generate comparison report
        report = interpretation_service.generate_comparison_report(sig1, sig2)

        assert isinstance(report, str)
        assert "STRUCTURAL COMPARISON REPORT" in report
        assert "Structure A" in report
        assert "Structure B" in report

    def test_corpus_persistence(self, container, sample_graph_1, sample_graph_2):
        """Test saving and loading corpus."""
        extractor = container.signature_extractor
        corpus_service = container.corpus_service

        # Create and populate corpus
        corpus = corpus_service.create_corpus("persist_test", "test")
        sig1 = extractor.extract_from_networkx(sample_graph_1, "test", "Graph 1", "persist_1")
        sig2 = extractor.extract_from_networkx(sample_graph_2, "test", "Graph 2", "persist_2")
        corpus_service.add_signature(corpus, sig1)
        corpus_service.add_signature(corpus, sig2)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            # Save corpus
            assert corpus_service.save_corpus(corpus, temp_path) is True

            # Load in new service
            new_service = type(corpus_service)(container.similarity_service)
            loaded = new_service.load_corpus(temp_path)

            assert loaded is not None
            assert loaded.name == "persist_test"
            assert len(loaded) == 2
            assert loaded.signatures[0].source_name == "Graph 1"
        finally:
            os.unlink(temp_path)


class TestCrossDomainResonance:
    """Test cross-domain resonance finding (the core use case)."""

    def test_cross_domain_similar_structures(self, container):
        """Test that similar structures resonate across domains."""
        extractor = container.signature_extractor
        corpus_service = container.corpus_service
        resonance_service = container.resonance_service

        # Create "music" corpus with hub-spoke (chord-like) structure
        music_corpus = corpus_service.create_corpus("music", "music")
        music_graph = nx.DiGraph()
        music_graph.add_node("tonic", label="C")
        for note in ["E", "G", "B"]:
            music_graph.add_node(note, label=note)
            music_graph.add_edge("tonic", note)
        music_sig = extractor.extract_from_networkx(music_graph, "music", "C Major Chord", "music_chord")
        corpus_service.add_signature(music_corpus, music_sig)

        # Create "text" query with similar hub-spoke structure
        text_graph = nx.DiGraph()
        text_graph.add_node("main_idea", label="thesis")
        for point in ["point1", "point2", "point3"]:
            text_graph.add_node(point, label=point)
            text_graph.add_edge("main_idea", point)
        text_sig = extractor.extract_from_networkx(text_graph, "text", "Thesis with Points", "text_thesis")

        # Find cross-domain resonance
        resonances = resonance_service.find_resonances(
            text_sig, music_corpus, threshold=0.0
        )

        assert len(resonances) > 0
        assert resonances[0].query_domain == "text"
        assert resonances[0].match_domain == "music"
        # Similar hub-spoke structures should have high similarity
        assert resonances[0].overall_score > 0.5


class TestAdapterIntegration:
    """Test domain adapters work with the full pipeline."""

    def test_text_adapter_produces_valid_graph(self, container):
        """Test text adaptation produces valid graph structure."""
        text_adapter = container.adapters["text"]

        # Adapt text to graph
        text = "The quick brown fox jumps over the lazy dog"
        graph_json = text_adapter.adapt(text)

        # Verify graph structure
        assert "nodes" in graph_json
        assert "edges" in graph_json
        assert len(graph_json["nodes"]) > 0

    def test_adapter_graph_to_signature(self, container):
        """Test converting adapter output to signature."""
        text_adapter = container.adapters["text"]
        extractor = container.signature_extractor

        # Adapt text to graph JSON
        text = "Hello world example text"
        graph_json = text_adapter.adapt(text)

        # Convert to NetworkX manually (adapters use 'edges' not 'links')
        G = nx.DiGraph()
        for node in graph_json["nodes"]:
            G.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
        for edge in graph_json["edges"]:
            G.add_edge(edge["source"], edge["target"])

        # Extract signature
        sig = extractor.extract_from_networkx(G, "text", "Sample text", "sample_text")

        assert sig.source_domain == "text"
        assert sig.num_nodes == len(graph_json["nodes"])


class TestProtocolCompliance:
    """Verify all services implement their protocols correctly."""

    def test_all_services_satisfy_protocols(self, container):
        """Integration test that all protocol compliance holds."""
        from structural_rorschach import (
            FullSimilarityService,
            FullCorpusService,
            FullInterpretationService,
            CanFindResonances,
            CanAdaptToGraph,
        )

        assert isinstance(container.similarity_service, FullSimilarityService)
        assert isinstance(container.corpus_service, FullCorpusService)
        assert isinstance(container.interpretation_service, FullInterpretationService)
        assert isinstance(container.resonance_service, CanFindResonances)

        for domain, adapter in container.adapters.items():
            assert isinstance(adapter, CanAdaptToGraph), f"{domain} adapter"
