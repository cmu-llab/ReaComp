"""
ReAct episodic memory.

Stores (task_description, code, answer, reward) entries after each solved task
and retrieves the top-K most similar entries for in-context learning.

Retrieval hierarchy (best available wins):
  1. sentence_transformers (cosine similarity on embeddings) — most accurate
  2. rank_bm25 (BM25) — fast, dependency-free quality retrieval
  3. Jaccard fallback — always available, used only if both above are missing

To install better retrieval:
    pip install rank_bm25           # BM25
    pip install sentence_transformers  # semantic embeddings
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional


def _tokenize(text: str) -> List[str]:
    return re.sub(r"[^a-z0-9]", " ", text.lower()).split()


def _tokenize_set(text: str) -> set:
    return set(_tokenize(text))


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    if not u:
        return 0.0
    return len(a & b) / len(u)


def _bm25_scores(query_tokens: List[str], corpus: List[List[str]], k1: float = 1.5, b: float = 0.75) -> List[float]:
    """Pure-Python BM25 scorer (used as fallback when rank_bm25 is not installed)."""
    N = len(corpus)
    if N == 0:
        return []
    avgdl = sum(len(d) for d in corpus) / N
    # IDF
    df: Counter = Counter()
    for doc in corpus:
        for term in set(doc):
            df[term] += 1
    idf = {
        t: math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)
        for t in df
    }
    scores = []
    for doc in corpus:
        dl = len(doc)
        tf: Counter = Counter(doc)
        score = 0.0
        for qt in query_tokens:
            if qt not in tf:
                continue
            f = tf[qt]
            score += idf.get(qt, 0.0) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores.append(score)
    return scores


class ReActMemory:
    """
    Episodic memory that stores (task, code, answer, reward) entries
    and retrieves the top-K most relevant ones for a new task.

    Retrieval hierarchy:
      1. sentence_transformers — if encoder is provided
      2. rank_bm25 — if installed (pip install rank_bm25)
      3. Built-in BM25 — always available, no extra dependencies

    Parameters
    ----------
    k : int
        Default number of examples to retrieve.
    encoder : optional
        A SentenceTransformer encoder for semantic retrieval.
    """

    def __init__(self, k: int = 3, encoder=None):
        self.k = k
        self.encoder = encoder
        self._entries: List[Dict] = []
        self._use_rank_bm25 = self._check_rank_bm25()

    @staticmethod
    def _check_rank_bm25() -> bool:
        try:
            import rank_bm25  # noqa: F401
            return True
        except ImportError:
            return False

    def __len__(self) -> int:
        return len(self._entries)

    def store(
        self,
        task_description: str,
        code: Optional[str],
        answer: Any,
        reward: float,
    ) -> None:
        """Store a task solution. Only entries with reward > 0 are kept."""
        if reward <= 0.0:
            return
        entry: Dict = {
            "task": task_description,
            "code": code or "",
            "answer": answer,
            "reward": reward,
        }
        if self.encoder is not None:
            entry["embedding"] = self.encoder.encode(task_description, convert_to_numpy=True)
        self._entries.append(entry)

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """Return up to k most similar stored entries for query."""
        k = k if k is not None else self.k
        if not self._entries:
            return []
        if self.encoder is not None:
            return self._retrieve_semantic(query, k)
        if self._use_rank_bm25:
            return self._retrieve_rank_bm25(query, k)
        return self._retrieve_bm25(query, k)

    def _retrieve_bm25(self, query: str, k: int) -> List[Dict]:
        """Built-in BM25 (no external deps)."""
        corpus = [_tokenize(e["task"]) for e in self._entries]
        query_tokens = _tokenize(query)
        scores = _bm25_scores(query_tokens, corpus)
        ranked = sorted(zip(self._entries, scores), key=lambda x: x[1], reverse=True)
        return [e for e, _ in ranked[:k]]

    def _retrieve_rank_bm25(self, query: str, k: int) -> List[Dict]:
        """rank_bm25 library retrieval."""
        from rank_bm25 import BM25Okapi
        corpus = [_tokenize(e["task"]) for e in self._entries]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._entries, scores), key=lambda x: x[1], reverse=True)
        return [e for e, _ in ranked[:k]]

    def _retrieve_semantic(self, query: str, k: int) -> List[Dict]:
        import numpy as np
        qemb = self.encoder.encode(query, convert_to_numpy=True)
        norms = []
        for e in self._entries:
            emb = e.get("embedding")
            if emb is None:
                emb = self.encoder.encode(e["task"], convert_to_numpy=True)
                e["embedding"] = emb
            qn = np.linalg.norm(qemb)
            en = np.linalg.norm(emb)
            sim = float(np.dot(qemb, emb) / (qn * en + 1e-9))
            norms.append((e, sim))
        norms.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in norms[:k]]

    def to_dict(self) -> List[Dict]:
        """Serialise (embeddings omitted — not JSON-serialisable)."""
        return [
            {"task": e["task"], "code": e["code"], "answer": str(e["answer"]), "reward": e["reward"]}
            for e in self._entries
        ]

    @classmethod
    def from_dict(cls, data: List[Dict], k: int = 3, encoder=None) -> "ReActMemory":
        mem = cls(k=k, encoder=encoder)
        for d in data:
            mem._entries.append({
                "task": d["task"],
                "code": d.get("code", ""),
                "answer": d.get("answer", ""),
                "reward": float(d.get("reward", 1.0)),
            })
        return mem
