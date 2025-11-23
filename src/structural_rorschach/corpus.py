"""
Corpus Management Service - Manage collections of structural signatures

Implements F-CM-01 through F-CM-06:
- F-CM-01: Create Corpus
- F-CM-02: Add to Corpus
- F-CM-03: Load Corpus
- F-CM-04: Save Corpus
- F-CM-05: Index Corpus
- F-CM-06: Query Corpus
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from .signature import StructuralSignature
from .similarity import SimilarityService, SimilarityResult


@dataclass
class Corpus:
    """A collection of structural signatures for a domain."""
    name: str
    domain: str
    description: str = ""
    signatures: List[StructuralSignature] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    # Index for fast lookup (signature_id -> index in signatures list)
    _index: Dict[str, int] = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self.signatures)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "domain": self.domain,
            "description": self.description,
            "signatures": [s.to_dict() for s in self.signatures],
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Corpus":
        corpus = cls(
            name=data["name"],
            domain=data["domain"],
            description=data.get("description", ""),
            metadata=data.get("metadata", {})
        )
        for sig_data in data.get("signatures", []):
            sig = StructuralSignature.from_dict(sig_data)
            corpus.add(sig)
        return corpus


class CorpusService:
    """
    Manage collections of structural signatures.

    Provides CRUD operations and similarity search.
    """

    def __init__(self, similarity_service: Optional[SimilarityService] = None):
        self.similarity_service = similarity_service or SimilarityService()
        self._corpora: Dict[str, Corpus] = {}

    def create_corpus(
        self,
        name: str,
        domain: str,
        description: str = ""
    ) -> Corpus:
        """
        Create a new signature corpus for a domain.

        F-CM-01: Create Corpus

        Args:
            name: Unique name for the corpus
            domain: Domain type (e.g., "image", "music", "text")
            description: Optional description

        Returns:
            New Corpus instance
        """
        corpus = Corpus(name=name, domain=domain, description=description)
        self._corpora[name] = corpus
        return corpus

    def add_signature(
        self,
        corpus: Corpus,
        signature: StructuralSignature
    ) -> bool:
        """
        Add a signature to an existing corpus.

        F-CM-02: Add to Corpus

        Args:
            corpus: Target corpus
            signature: Signature to add

        Returns:
            True if added successfully
        """
        # Check for duplicates
        if signature.source_id in corpus._index:
            return False

        # Add to list and index
        index = len(corpus.signatures)
        corpus.signatures.append(signature)
        corpus._index[signature.source_id] = index
        return True

    def load_corpus(self, path: str) -> Optional[Corpus]:
        """
        Load a corpus from persistent storage.

        F-CM-03: Load Corpus

        Args:
            path: Path to corpus JSON file

        Returns:
            Loaded Corpus or None if failed
        """
        try:
            file_path = Path(path)
            if not file_path.exists():
                return None

            with open(file_path, 'r') as f:
                data = json.load(f)

            corpus = Corpus.from_dict(data)
            self._corpora[corpus.name] = corpus
            return corpus

        except (json.JSONDecodeError, KeyError, IOError):
            return None

    def save_corpus(self, corpus: Corpus, path: str) -> bool:
        """
        Save a corpus to persistent storage.

        F-CM-04: Save Corpus

        Args:
            corpus: Corpus to save
            path: Destination file path

        Returns:
            True if saved successfully
        """
        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w') as f:
                json.dump(corpus.to_dict(), f, indent=2)

            return True

        except IOError:
            return False

    def build_index(self, corpus: Corpus) -> None:
        """
        Build/rebuild search index for a corpus.

        F-CM-05: Index Corpus

        Args:
            corpus: Corpus to index
        """
        corpus._index.clear()
        for i, sig in enumerate(corpus.signatures):
            corpus._index[sig.source_id] = i

    def query(
        self,
        corpus: Corpus,
        query_signature: StructuralSignature,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[StructuralSignature, SimilarityResult]]:
        """
        Find similar signatures in a corpus.

        F-CM-06: Query Corpus

        Args:
            corpus: Corpus to search
            query_signature: Query signature
            top_k: Maximum number of results
            threshold: Minimum similarity score

        Returns:
            List of (signature, similarity_result) tuples, sorted by score
        """
        results = []

        for sig in corpus.signatures:
            # Skip self-comparison
            if sig.source_id == query_signature.source_id:
                continue

            sim_result = self.similarity_service.compute_similarity(
                query_signature, sig
            )

            if sim_result.overall_score >= threshold:
                results.append((sig, sim_result))

        # Sort by overall score descending
        results.sort(key=lambda x: x[1].overall_score, reverse=True)

        return results[:top_k]

    def get_corpus(self, name: str) -> Optional[Corpus]:
        """Get a corpus by name from the service's cache."""
        return self._corpora.get(name)

    def list_corpora(self) -> List[str]:
        """List all loaded corpus names."""
        return list(self._corpora.keys())


# Convenience functions
def create_corpus(name: str, domain: str, description: str = "") -> Corpus:
    """Create a new corpus."""
    return Corpus(name=name, domain=domain, description=description)


def load_corpus(path: str) -> Optional[Corpus]:
    """Load a corpus from file."""
    service = CorpusService()
    return service.load_corpus(path)


def save_corpus(corpus: Corpus, path: str) -> bool:
    """Save a corpus to file."""
    service = CorpusService()
    return service.save_corpus(corpus, path)
