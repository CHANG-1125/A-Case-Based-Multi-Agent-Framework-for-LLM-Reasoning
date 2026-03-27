"""FAISS-backed semantic case retrieval (cosine similarity via normalized IP)."""

from __future__ import annotations

import os

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


class CaseBase:
    def __init__(self, embedding_model_name: str):
        # local_files_only: use only HF cache (no Hub HEAD/download). Set when SSL/proxy breaks Hub.
        kwargs: dict = {}
        if _env_truthy("EMBEDDING_LOCAL_FILES_ONLY"):
            kwargs["local_files_only"] = True
        cache = (os.environ.get("SENTENCE_TRANSFORMERS_HOME") or "").strip()
        if cache:
            kwargs["cache_folder"] = cache
        self._model = SentenceTransformer(embedding_model_name, **kwargs)
        self._dim = self._model.get_sentence_embedding_dimension()
        self._cases: list[dict] = []
        self._index: faiss.Index | None = None
        self._vecs: np.ndarray | None = None

    @property
    def size(self) -> int:
        return len(self._cases)

    def build(self, cases: list[dict]) -> None:
        """
        cases: list of {"question": str, "answer": str} (answer = full CoT + #### for GSM8K)
        """
        self._cases = list(cases)
        if not self._cases:
            self._index = None
            self._vecs = None
            return
        texts = [c["question"] for c in self._cases]
        emb = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        self._vecs = emb.astype(np.float32)
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(self._vecs)

    def add_case(self, question: str, answer: str) -> None:
        """Incrementally retain a verified case (rebuild index for simplicity / correctness)."""
        self._cases.append({"question": question, "answer": answer})
        self.build(self._cases)

    def retrieve(self, query: str, top_k: int, exclude_indices: set[int] | None = None) -> list[tuple[dict, float]]:
        if not self._index or self._index.ntotal == 0:
            return []
        q = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        k = min(top_k + (len(exclude_indices or {})), self._index.ntotal)
        scores, idx = self._index.search(q, k)
        out: list[tuple[dict, float]] = []
        exclude_indices = exclude_indices or set()
        for i, s in zip(idx[0], scores[0]):
            if i < 0:
                continue
            if i in exclude_indices:
                continue
            out.append((self._cases[i], float(s)))
            if len(out) >= top_k:
                break
        return out
