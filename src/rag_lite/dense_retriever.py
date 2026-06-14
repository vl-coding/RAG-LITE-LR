from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from sentence_transformers import SentenceTransformer

from .schemas import Document


class DenseRetriever:
    """Brute-force numpy cosine-similarity dense retriever.

    Appropriate for corpora up to ~100K documents: embeddings are kept as a
    single in-memory matrix and ranked via one matmul per query, avoiding the
    persistent-DB overhead (and cold-start latency) of a vector database like
    ChromaDB.
    """

    def __init__(self, model_name: str, persist_dir: str):
        self.model = SentenceTransformer(model_name)
        self.persist_dir = Path(persist_dir)
        self.doc_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def build_index(self, documents: List[Document], batch_size: int = 128) -> None:
        self.doc_ids = [d.doc_id for d in documents]
        texts = [d.text for d in documents]

        all_embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_embeddings.append(batch_embeddings)

        self.embeddings = (
            np.vstack(all_embeddings) if all_embeddings else np.zeros((0, 0), dtype=np.float32)
        )

    def save(self) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(self.persist_dir / "embeddings.npy"), self.embeddings)
        np.save(str(self.persist_dir / "doc_ids.npy"), np.array(self.doc_ids, dtype=object))

    def load(self) -> None:
        self.embeddings = np.load(str(self.persist_dir / "embeddings.npy"))
        self.doc_ids = np.load(str(self.persist_dir / "doc_ids.npy"), allow_pickle=True).tolist()

    def search(
        self,
        query_text: str,
        candidate_ids: Optional[Set[str]] = None,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        if self.embeddings is None or len(self.doc_ids) == 0:
            return []

        query_embedding = self.model.encode(
            [query_text],
            normalize_embeddings=True,
        )[0]

        scores = self.embeddings @ query_embedding

        order = np.argsort(-scores)

        output = []
        rank = 1
        for idx in order:
            doc_id = self.doc_ids[int(idx)]
            if candidate_ids is not None and doc_id not in candidate_ids:
                continue
            output.append({
                "doc_id": doc_id,
                "rank": rank,
                "score": float(scores[idx]),
                "source": "dense",
            })
            rank += 1
            if rank > top_n:
                break

        return output
