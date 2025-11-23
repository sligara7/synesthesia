"""
Domain Adapter Service - Convert domain-specific data to unified graph representations

Implements F-DA-01 through F-DA-05:
- F-DA-01: Adapt Text to Graph (word transition graph)
- F-DA-02: Adapt Image to Graph (region adjacency graph)
- F-DA-03: Adapt Music to Graph (note transition graph)
- F-DA-04: Adapt Code to Graph (AST/dependency graph)
- F-DA-05: Validate Graph Format
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import Counter
import json
import re


@dataclass
class GraphValidationResult:
    """Result of graph format validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    node_count: int
    edge_count: int


class DomainAdapter(ABC):
    """Abstract base class for domain adapters."""

    @property
    @abstractmethod
    def domain(self) -> str:
        """Return the domain name."""
        pass

    @abstractmethod
    def adapt(self, data: Any, **kwargs) -> Dict:
        """Convert domain data to graph JSON."""
        pass


class TextAdapter(DomainAdapter):
    """
    Convert text to word transition graph.

    F-DA-01: Adapt Text to Graph

    Nodes represent tokens (words/characters/sentences).
    Edges represent transitions with probability weights.
    """

    @property
    def domain(self) -> str:
        return "text"

    def adapt(
        self,
        text: str,
        tokenizer: str = "word",
        min_frequency: int = 1,
        **kwargs
    ) -> Dict:
        """
        Convert text to word transition graph.

        Args:
            text: Input text corpus
            tokenizer: "word", "char", or "sentence"
            min_frequency: Minimum token frequency to include

        Returns:
            Graph JSON dictionary
        """
        # Tokenize
        tokens = self._tokenize(text, tokenizer)

        if len(tokens) < 2:
            return self._empty_graph("text")

        # Count tokens and transitions
        token_counts = Counter(tokens)
        transition_counts = Counter()

        for i in range(len(tokens) - 1):
            transition_counts[(tokens[i], tokens[i + 1])] += 1

        # Filter by frequency
        valid_tokens = {t for t, c in token_counts.items() if c >= min_frequency}

        # Build graph
        nodes = []
        node_ids = {}
        for i, token in enumerate(sorted(valid_tokens)):
            node_id = f"n_{i}"
            node_ids[token] = node_id
            nodes.append({
                "id": node_id,
                "label": token,
                "attributes": {
                    "frequency": token_counts[token],
                    "type": "token"
                }
            })

        edges = []
        for (src, tgt), count in transition_counts.items():
            if src in valid_tokens and tgt in valid_tokens:
                # Calculate transition probability
                prob = count / token_counts[src]
                edges.append({
                    "source": node_ids[src],
                    "target": node_ids[tgt],
                    "attributes": {
                        "weight": prob,
                        "count": count,
                        "type": "transition"
                    }
                })

        return {
            "metadata": {
                "domain": "text",
                "adapter": "TextAdapter",
                "tokenizer": tokenizer,
                "original_token_count": len(tokens)
            },
            "nodes": nodes,
            "edges": edges,
            "directed": True
        }

    def _tokenize(self, text: str, method: str) -> List[str]:
        """Tokenize text using specified method."""
        if method == "char":
            return list(text)
        elif method == "sentence":
            # Simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
        else:  # word
            # Simple word tokenization
            words = re.findall(r'\b\w+\b', text.lower())
            return words


class ImageAdapter(DomainAdapter):
    """
    Convert image to region adjacency graph.

    F-DA-02: Adapt Image to Graph

    Nodes represent image regions (superpixels).
    Edges represent spatial adjacency with boundary length weights.

    Note: Requires scikit-image for full functionality.
    """

    @property
    def domain(self) -> str:
        return "image"

    def adapt(
        self,
        image_data: Any,
        n_segments: int = 100,
        **kwargs
    ) -> Dict:
        """
        Convert image to region adjacency graph.

        Args:
            image_data: Image array (H x W x C) or path string
            n_segments: Number of superpixel segments

        Returns:
            Graph JSON dictionary
        """
        try:
            # Try to use scikit-image if available
            from skimage.segmentation import slic
            from skimage.future import graph as rag_module
            import numpy as np

            # Load image if path provided
            if isinstance(image_data, str):
                from skimage import io
                image = io.imread(image_data)
            else:
                image = np.array(image_data)

            # Generate superpixels
            segments = slic(image, n_segments=n_segments, compactness=10)

            # Build region adjacency graph
            rag = rag_module.rag_mean_color(image, segments)

            # Convert to our format
            nodes = []
            for node_id in rag.nodes():
                nodes.append({
                    "id": f"r_{node_id}",
                    "label": f"region_{node_id}",
                    "attributes": {
                        "type": "region",
                        "segment_id": int(node_id)
                    }
                })

            edges = []
            for u, v, data in rag.edges(data=True):
                edges.append({
                    "source": f"r_{u}",
                    "target": f"r_{v}",
                    "attributes": {
                        "weight": data.get("weight", 1.0),
                        "type": "adjacency"
                    }
                })

            return {
                "metadata": {
                    "domain": "image",
                    "adapter": "ImageAdapter",
                    "n_segments": n_segments,
                    "actual_segments": len(nodes)
                },
                "nodes": nodes,
                "edges": edges,
                "directed": False
            }

        except ImportError:
            # Fallback: return placeholder structure
            return self._placeholder_graph(n_segments)

    def _placeholder_graph(self, n_nodes: int) -> Dict:
        """Generate placeholder graph when scikit-image unavailable."""
        nodes = [
            {"id": f"r_{i}", "label": f"region_{i}", "attributes": {"type": "region"}}
            for i in range(min(n_nodes, 50))
        ]
        # Simple grid-like adjacency
        edges = []
        side = int(min(n_nodes, 50) ** 0.5)
        for i in range(side):
            for j in range(side - 1):
                idx = i * side + j
                if idx < len(nodes) - 1:
                    edges.append({
                        "source": f"r_{idx}",
                        "target": f"r_{idx + 1}",
                        "attributes": {"weight": 1.0, "type": "adjacency"}
                    })

        return {
            "metadata": {
                "domain": "image",
                "adapter": "ImageAdapter",
                "placeholder": True,
                "note": "scikit-image not available, using placeholder"
            },
            "nodes": nodes,
            "edges": edges,
            "directed": False
        }


class MusicAdapter(DomainAdapter):
    """
    Convert music to note transition graph.

    F-DA-03: Adapt Music to Graph

    Nodes represent musical notes/events.
    Edges represent temporal transitions with probability weights.

    Note: Requires mido for MIDI, librosa for audio.
    """

    @property
    def domain(self) -> str:
        return "music"

    def adapt(
        self,
        music_data: Any,
        source_type: str = "midi",
        **kwargs
    ) -> Dict:
        """
        Convert music to note transition graph.

        Args:
            music_data: MIDI file path, audio array, or note list
            source_type: "midi", "audio", or "notes"

        Returns:
            Graph JSON dictionary
        """
        if source_type == "midi":
            return self._adapt_midi(music_data)
        elif source_type == "notes":
            return self._adapt_note_list(music_data)
        else:
            return self._adapt_audio(music_data)

    def _adapt_midi(self, midi_path: str) -> Dict:
        """Convert MIDI file to note graph."""
        try:
            import mido

            midi = mido.MidiFile(midi_path)
            notes = []

            for track in midi.tracks:
                for msg in track:
                    if msg.type == 'note_on' and msg.velocity > 0:
                        notes.append(msg.note)

            return self._adapt_note_list(notes)

        except ImportError:
            return self._placeholder_music_graph()

    def _adapt_note_list(self, notes: List[int]) -> Dict:
        """Convert list of note numbers to graph."""
        if len(notes) < 2:
            return self._empty_graph("music")

        # Count notes and transitions
        note_counts = Counter(notes)
        transition_counts = Counter()

        for i in range(len(notes) - 1):
            transition_counts[(notes[i], notes[i + 1])] += 1

        # Build graph
        nodes = []
        node_ids = {}
        for i, note in enumerate(sorted(note_counts.keys())):
            node_id = f"note_{note}"
            node_ids[note] = node_id
            nodes.append({
                "id": node_id,
                "label": self._note_name(note),
                "attributes": {
                    "midi_number": note,
                    "frequency": note_counts[note],
                    "type": "note"
                }
            })

        edges = []
        for (src, tgt), count in transition_counts.items():
            prob = count / note_counts[src]
            edges.append({
                "source": node_ids[src],
                "target": node_ids[tgt],
                "attributes": {
                    "weight": prob,
                    "count": count,
                    "type": "transition"
                }
            })

        return {
            "metadata": {
                "domain": "music",
                "adapter": "MusicAdapter",
                "note_count": len(notes),
                "unique_notes": len(nodes)
            },
            "nodes": nodes,
            "edges": edges,
            "directed": True
        }

    def _adapt_audio(self, audio_data: Any) -> Dict:
        """Convert audio to note graph (requires librosa)."""
        try:
            import librosa
            import numpy as np

            if isinstance(audio_data, str):
                y, sr = librosa.load(audio_data)
            else:
                y = audio_data
                sr = 22050

            # Extract pitches
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            notes = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    midi_note = int(librosa.hz_to_midi(pitch))
                    notes.append(midi_note)

            return self._adapt_note_list(notes)

        except ImportError:
            return self._placeholder_music_graph()

    def _note_name(self, midi_note: int) -> str:
        """Convert MIDI note number to name."""
        names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = midi_note // 12 - 1
        note = names[midi_note % 12]
        return f"{note}{octave}"

    def _placeholder_music_graph(self) -> Dict:
        """Generate placeholder music graph."""
        # Simple chromatic scale
        notes = list(range(60, 72))  # C4 to B4
        return self._adapt_note_list(notes * 4)


class CodeAdapter(DomainAdapter):
    """
    Convert source code to AST/dependency graph.

    F-DA-04: Adapt Code to Graph

    Nodes represent code elements (functions, classes, modules).
    Edges represent dependencies (calls, imports, inheritance).
    """

    @property
    def domain(self) -> str:
        return "code"

    def adapt(
        self,
        source_code: str,
        language: str = "python",
        **kwargs
    ) -> Dict:
        """
        Convert source code to dependency graph.

        Args:
            source_code: Source code string
            language: Programming language ("python" or "javascript")

        Returns:
            Graph JSON dictionary
        """
        if language == "python":
            return self._adapt_python(source_code)
        else:
            return self._adapt_generic(source_code)

    def _adapt_python(self, source_code: str) -> Dict:
        """Parse Python code to graph."""
        import ast

        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return self._empty_graph("code")

        nodes = []
        edges = []
        node_ids = {}
        current_id = 0

        # Collect all function and class definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                node_id = f"func_{current_id}"
                node_ids[node.name] = node_id
                nodes.append({
                    "id": node_id,
                    "label": node.name,
                    "attributes": {
                        "type": "function",
                        "lineno": node.lineno,
                        "args": len(node.args.args)
                    }
                })
                current_id += 1

            elif isinstance(node, ast.ClassDef):
                node_id = f"class_{current_id}"
                node_ids[node.name] = node_id
                nodes.append({
                    "id": node_id,
                    "label": node.name,
                    "attributes": {
                        "type": "class",
                        "lineno": node.lineno,
                        "bases": [b.id for b in node.bases if isinstance(b, ast.Name)]
                    }
                })
                current_id += 1

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    if module_name not in node_ids:
                        node_id = f"import_{current_id}"
                        node_ids[module_name] = node_id
                        nodes.append({
                            "id": node_id,
                            "label": module_name,
                            "attributes": {"type": "import"}
                        })
                        current_id += 1

        # Find function calls
        class CallVisitor(ast.NodeVisitor):
            def __init__(self, parent_func=None):
                self.parent_func = parent_func
                self.calls = []

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    self.calls.append(node.func.id)
                self.generic_visit(node)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                visitor = CallVisitor(node.name)
                visitor.visit(node)

                for called in visitor.calls:
                    if called in node_ids and node.name in node_ids:
                        edges.append({
                            "source": node_ids[node.name],
                            "target": node_ids[called],
                            "attributes": {"type": "call"}
                        })

        return {
            "metadata": {
                "domain": "code",
                "adapter": "CodeAdapter",
                "language": "python"
            },
            "nodes": nodes,
            "edges": edges,
            "directed": True
        }

    def _adapt_generic(self, source_code: str) -> Dict:
        """Simple regex-based parsing for other languages."""
        # Extract function-like patterns
        func_pattern = r'(?:function|def|func|fn)\s+(\w+)'
        class_pattern = r'(?:class|struct|interface)\s+(\w+)'

        functions = re.findall(func_pattern, source_code)
        classes = re.findall(class_pattern, source_code)

        nodes = []
        node_ids = {}

        for i, func in enumerate(functions):
            node_id = f"func_{i}"
            node_ids[func] = node_id
            nodes.append({
                "id": node_id,
                "label": func,
                "attributes": {"type": "function"}
            })

        for i, cls in enumerate(classes):
            node_id = f"class_{i}"
            node_ids[cls] = node_id
            nodes.append({
                "id": node_id,
                "label": cls,
                "attributes": {"type": "class"}
            })

        # Simple call detection
        edges = []
        for func in functions:
            for other in functions:
                if func != other and re.search(rf'\b{other}\s*\(', source_code):
                    if func in node_ids and other in node_ids:
                        edges.append({
                            "source": node_ids[func],
                            "target": node_ids[other],
                            "attributes": {"type": "call"}
                        })

        return {
            "metadata": {"domain": "code", "adapter": "CodeAdapter", "language": "generic"},
            "nodes": nodes,
            "edges": edges,
            "directed": True
        }


def validate_graph(graph_json: Dict) -> GraphValidationResult:
    """
    Validate graph JSON against schema.

    F-DA-05: Validate Graph Format

    Args:
        graph_json: Graph dictionary to validate

    Returns:
        GraphValidationResult with validation details
    """
    errors = []
    warnings = []

    # Check required fields
    if "nodes" not in graph_json:
        errors.append("Missing required field: 'nodes'")
    if "edges" not in graph_json:
        errors.append("Missing required field: 'edges'")

    nodes = graph_json.get("nodes", [])
    edges = graph_json.get("edges", [])

    if not isinstance(nodes, list):
        errors.append("'nodes' must be a list")
        nodes = []
    if not isinstance(edges, list):
        errors.append("'edges' must be a list")
        edges = []

    # Validate nodes
    node_ids = set()
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"Node {i} is not a dictionary")
            continue
        if "id" not in node:
            errors.append(f"Node {i} missing required field 'id'")
        else:
            if node["id"] in node_ids:
                errors.append(f"Duplicate node id: {node['id']}")
            node_ids.add(node["id"])
        if "label" not in node:
            warnings.append(f"Node {i} missing optional field 'label'")

    # Validate edges
    for i, edge in enumerate(edges):
        if not isinstance(edge, dict):
            errors.append(f"Edge {i} is not a dictionary")
            continue
        if "source" not in edge:
            errors.append(f"Edge {i} missing required field 'source'")
        elif edge["source"] not in node_ids:
            errors.append(f"Edge {i} references unknown source: {edge['source']}")
        if "target" not in edge:
            errors.append(f"Edge {i} missing required field 'target'")
        elif edge["target"] not in node_ids:
            errors.append(f"Edge {i} references unknown target: {edge['target']}")

    return GraphValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        node_count=len(nodes),
        edge_count=len(edges)
    )


def _empty_graph(domain: str) -> Dict:
    """Return empty graph structure."""
    return {
        "metadata": {"domain": domain, "empty": True},
        "nodes": [],
        "edges": [],
        "directed": True
    }


# Adapter factory
def get_adapter(domain: str) -> DomainAdapter:
    """Get adapter for specified domain."""
    adapters = {
        "text": TextAdapter,
        "image": ImageAdapter,
        "music": MusicAdapter,
        "code": CodeAdapter
    }
    adapter_class = adapters.get(domain)
    if adapter_class is None:
        raise ValueError(f"Unknown domain: {domain}")
    return adapter_class()


def adapt_to_graph(data: Any, domain: str, **kwargs) -> Dict:
    """Convert domain data to graph using appropriate adapter."""
    adapter = get_adapter(domain)
    return adapter.adapt(data, **kwargs)
