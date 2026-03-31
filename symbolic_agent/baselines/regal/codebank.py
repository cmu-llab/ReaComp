"""ReGAL CodeBank and DemoBank — faithful to codebank/codebank.py in the original.

ReGALCodeBank stores verified helper functions and supports two retrieval modes:
  - 'sentence_transformers' (default): cosine similarity on name+description embeddings
  - 'chromadb': ChromaDB persistent vector store with sentence_transformer embeddings

ReGALDemoBank stores (query, program, helpers, success) tuples for ICL retrieval at test time.

Paper §3.2 — pruneCodeBank: score = |passing| − Σ_{p ∈ failing} 1/n_p, prune if score ≤ θ.
Paper §A.3 — retrieve up to 20 relevant helper functions using query↔(name+description) similarity.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .function import RegalFunction

logger = logging.getLogger(__name__)

# Prune threshold θ (paper §A.2 and Table 9: 0.0 for all experiments).
_PRUNE_THRESHOLD = 0.0
# Minimum number of uses before a function is eligible for pruning.
_MIN_USES_TO_PRUNE = 3


class ReGALCodeBank:
    """
    Library of verified helper functions learned during ReGAL training.

    Parameters
    ----------
    retrieval : {'sentence_transformers', 'chromadb'}
        Backend for vector similarity retrieval.
    embedding_model : str
        Sentence transformer model for encoding (used by both backends).
    chroma_path : str | None
        Persist directory for chromadb (only used when retrieval='chromadb').
        If None, an in-memory chroma client is used.
    """

    def __init__(
        self,
        retrieval: str = "sentence_transformers",
        embedding_model: str = "all-MiniLM-L6-v2",
        chroma_path: Optional[str] = None,
    ):
        self.retrieval = retrieval
        self.embedding_model = embedding_model
        self.chroma_path = chroma_path

        self.functions: List[RegalFunction] = []
        self._name_to_idx: Dict[str, int] = {}

        # sentence_transformers state
        self._encoder = None
        self._embeddings = None  # np array, shape (n_functions, dim) or None if stale

        # chromadb state
        self._chroma_client = None
        self._chroma_collection = None

        if retrieval == "chromadb":
            self._init_chromadb()

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.embedding_model)
            except ImportError:
                raise ImportError(
                    "sentence_transformers is not installed. "
                    "Run: pip install sentence_transformers"
                )
        return self._encoder

    def _init_chromadb(self) -> None:
        try:
            import chromadb
            if self.chroma_path:
                self._chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            else:
                self._chroma_client = chromadb.Client()
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name="regal_codebank",
                metadata={"hnsw:space": "cosine"},
            )
        except ImportError:
            raise ImportError(
                "chromadb is not installed. Run: pip install chromadb"
            )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add(self, func: RegalFunction) -> None:
        """Add a function (replaces existing entry with the same name)."""
        if func.name in self._name_to_idx:
            idx = self._name_to_idx[func.name]
            self.functions[idx] = func
            self._embeddings = None
            if self._chroma_collection is not None:
                self._chroma_upsert(func)
        else:
            self._name_to_idx[func.name] = len(self.functions)
            self.functions.append(func)
            self._embeddings = None
            if self._chroma_collection is not None:
                self._chroma_upsert(func)

    def remove(self, name: str) -> bool:
        """Remove a function by name. Returns True if removed, False if not found."""
        if name not in self._name_to_idx:
            return False
        self.functions.pop(self._name_to_idx[name])
        self._name_to_idx = {f.name: i for i, f in enumerate(self.functions)}
        self._embeddings = None
        if self._chroma_collection is not None:
            try:
                self._chroma_collection.delete(ids=[name])
            except Exception:
                pass
        return True

    def get(self, name: str) -> Optional[RegalFunction]:
        idx = self._name_to_idx.get(name)
        return self.functions[idx] if idx is not None else None

    def __len__(self) -> int:
        return len(self.functions)

    def __iter__(self):
        return iter(self.functions)

    def __contains__(self, name: str) -> bool:
        return name in self._name_to_idx

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 20) -> List[RegalFunction]:
        """Return up to k most relevant functions for a query (§A.3)."""
        if not self.functions:
            return []
        k = min(k, len(self.functions))
        if self.retrieval == "chromadb" and self._chroma_collection is not None:
            return self._retrieve_chromadb(query, k)
        return self._retrieve_sentence_transformers(query, k)

    def _retrieve_sentence_transformers(self, query: str, k: int) -> List[RegalFunction]:
        import numpy as np
        encoder = self._get_encoder()
        if self._embeddings is None or self._embeddings.shape[0] != len(self.functions):
            texts = [f"{f.name}: {f.description}" for f in self.functions]
            self._embeddings = encoder.encode(texts, convert_to_numpy=True)
        q_emb = encoder.encode([query], convert_to_numpy=True)[0]
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        normed = self._embeddings / (norms + 1e-9)
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        sims = normed @ q_norm
        top_idx = np.argsort(sims)[::-1][:k]
        return [self.functions[int(i)] for i in top_idx]

    def _retrieve_chromadb(self, query: str, k: int) -> List[RegalFunction]:
        encoder = self._get_encoder()
        q_emb = encoder.encode([query])[0].tolist()
        try:
            count = self._chroma_collection.count()
            if count == 0:
                return []
            results = self._chroma_collection.query(
                query_embeddings=[q_emb],
                n_results=min(k, count),
                include=["ids"],
            )
            return [f for name in results["ids"][0] if (f := self.get(name)) is not None]
        except Exception as exc:
            logger.warning("ChromaDB retrieval failed (%s) — falling back to sentence_transformers", exc)
            return self._retrieve_sentence_transformers(query, k)

    def _chroma_upsert(self, func: RegalFunction) -> None:
        encoder = self._get_encoder()
        text = f"{func.name}: {func.description}"
        emb = encoder.encode([text])[0].tolist()
        self._chroma_collection.upsert(
            ids=[func.name],
            embeddings=[emb],
            documents=[text],
        )

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def as_str(self, include_success: bool = False) -> str:
        """Format all functions for inclusion in a prompt."""
        if not self.functions:
            return ""
        return "\n\n".join(f.summarize(include_success=include_success) for f in self.functions)

    # ------------------------------------------------------------------
    # Pruning (Stage 3b, §3.2 and §A.2)
    # ------------------------------------------------------------------

    def prune(
        self,
        threshold: float = _PRUNE_THRESHOLD,
        min_uses: int = _MIN_USES_TO_PRUNE,
    ) -> List[str]:
        """
        Remove functions whose blame-normalized score ≤ threshold.

        Mirrors pruneCodeBank() from the paper (§A.2):
            score = |passing| − Σ_{p ∈ failing} 1/n_p
        where n_p = number of helper functions used in program p.

        Functions with fewer than min_uses unit test recordings are skipped.

        Returns
        -------
        pruned : list of removed function names
        """
        to_prune = []
        for func in list(self.functions):
            score, n = func.compute_success()
            if n >= min_uses and score <= threshold:
                to_prune.append(func.name)
        for name in to_prune:
            self.remove(name)
            logger.info("Pruned function: %s", name)
        return to_prune

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "retrieval": self.retrieval,
            "embedding_model": self.embedding_model,
            "chroma_path": self.chroma_path,
            "functions": [f.to_dict() for f in self.functions],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReGALCodeBank":
        bank = cls(
            retrieval=d.get("retrieval", "sentence_transformers"),
            embedding_model=d.get("embedding_model", "all-MiniLM-L6-v2"),
            chroma_path=d.get("chroma_path"),
        )
        for fd in d.get("functions", []):
            bank.add(RegalFunction.from_dict(fd))
        return bank

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("CodeBank saved to %s (%d functions)", path, len(self))

    @classmethod
    def load(cls, path: str) -> "ReGALCodeBank":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        bank = cls.from_dict(d)
        logger.info("CodeBank loaded from %s (%d functions)", path, len(bank))
        return bank


class ReGALDemoBank:
    """
    Bank of verified refactored programs used as ICL demonstrations at test time.

    Each demo:
        {
            "query":   str,   # natural-language query
            "program": str,   # refactored program (uses helpers)
            "helpers": str,   # helper functions used (may be empty)
            "success": bool,  # True if program passed verification
        }

    Retrieval: cosine similarity between test query and stored queries via
    sentence_transformers (always sentence_transformers — demos don't need chromadb).
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.embedding_model = embedding_model
        self.demos: List[dict] = []
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.embedding_model)
            except ImportError:
                raise ImportError("sentence_transformers is not installed.")
        return self._encoder

    def add(self, query: str, program: str, helpers: str, success: bool) -> None:
        self.demos.append({
            "query": query,
            "program": program,
            "helpers": helpers,
            "success": success,
        })

    def retrieve(
        self,
        query: str,
        k: int,
        success_only: bool = False,
    ) -> List[dict]:
        """
        Return up to k demos most similar to query.

        Parameters
        ----------
        success_only : bool
            If True, only retrieve demos that passed verification.
        """
        pool = [d for d in self.demos if not success_only or d["success"]]
        if not pool:
            return []
        k = min(k, len(pool))
        try:
            import numpy as np
            encoder = self._get_encoder()
            pool_embs = encoder.encode([d["query"] for d in pool], convert_to_numpy=True)
            q_emb = encoder.encode([query], convert_to_numpy=True)[0]
            norms = np.linalg.norm(pool_embs, axis=1, keepdims=True)
            normed = pool_embs / (norms + 1e-9)
            q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
            sims = normed @ q_norm
            top_idx = np.argsort(sims)[::-1][:k]
            return [pool[int(i)] for i in top_idx]
        except ImportError:
            # Fallback: return first k without ranking
            return pool[:k]

    def __len__(self) -> int:
        return len(self.demos)

    def to_dict(self) -> dict:
        return {
            "embedding_model": self.embedding_model,
            "demos": self.demos,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReGALDemoBank":
        bank = cls(embedding_model=d.get("embedding_model", "all-MiniLM-L6-v2"))
        bank.demos = d.get("demos", [])
        return bank

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "ReGALDemoBank":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(d)
