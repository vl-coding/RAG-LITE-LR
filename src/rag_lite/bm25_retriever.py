from pathlib import Path
from typing import List, Dict, Any, Optional, Set

import numpy as np
import bm25s

from .schemas import Document
from .tokenize import tokenize


class BM25Retriever:
    def __init__(self):
        self.doc_ids: List[str] = []
        self.doc_id_to_index: Dict[str, int] = {}
        self._bm25: Optional[bm25s.BM25] = None

    def build_index(self, documents: List[Document]) -> None:
        self.doc_ids = [d.doc_id for d in documents]
        self.doc_id_to_index = {did: i for i, did in enumerate(self.doc_ids)}
        corpus_tokens = [tokenize(d.text) for d in documents]
        self._bm25 = bm25s.BM25()
        self._bm25.index(corpus_tokens)

    def search(
        self,
        query: str,
        candidate_ids: Optional[Set[str]] = None,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        query_tokens = [tokenize(query)]
        n_docs = len(self.doc_ids)

        k = min(n_docs, max(top_n * 10, len(candidate_ids) * 2) if candidate_ids else top_n)

        doc_indices, scores = self._bm25.retrieve(query_tokens, k=k)

        results = []
        rank = 1
        for idx, score in zip(doc_indices[0], scores[0]):
            doc_id = self.doc_ids[int(idx)]
            if candidate_ids and doc_id not in candidate_ids:
                continue
            results.append({
                "doc_id": doc_id,
                "rank": rank,
                "score": float(score),
                "source": "bm25",
            })
            rank += 1
            if rank > top_n:
                break

        return results

    def save(self, path: str) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self._bm25.save(str(out / "index"))
        np.save(str(out / "doc_ids.npy"), np.array(self.doc_ids, dtype=object))

    @staticmethod
    def load(path: str, mmap: bool = False) -> "BM25Retriever":
        out = Path(path)
        r = BM25Retriever()
        r._bm25 = bm25s.BM25.load(str(out / "index"), load_corpus=False, mmap=mmap)
        r.doc_ids = np.load(str(out / "doc_ids.npy"), allow_pickle=True).tolist()
        r.doc_id_to_index = {did: i for i, did in enumerate(r.doc_ids)}
        return r
