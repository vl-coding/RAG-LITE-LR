"""Result-level near-duplicate detection.

Corpus-wide pairwise comparison is infeasible at scale, so detection is
scoped to the small final result set returned for a single query (typically
top 5-10 documents after RRF fusion). This keeps the comparison O(N^2) on a
tiny N, which is cheap.

Similarity is computed on the same SBERT embeddings (title + abstract,
`all-MiniLM-L6-v2`, L2-normalized) already used for dense retrieval, so no
extra model is introduced.
"""
from typing import Dict, List, Sequence

import numpy as np

# Cosine similarity threshold above which two results in the same result set
# are flagged as likely near-duplicates. 0.92 favors precision over recall --
# unrelated abstracts typically score well below 0.5, and topically-related-
# but-distinct documents commonly land in 0.6-0.85.
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.92


def find_near_duplicates(
    doc_ids: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
) -> Dict[str, List[str]]:
    """Find pairs of near-duplicate documents within a small result set.

    Args:
        doc_ids: ids of the documents in the result set, in any order.
        embeddings: parallel list of embedding vectors (one per doc_id).
            Vectors are L2-normalized internally, so callers may pass either
            normalized or unnormalized vectors.
        threshold: cosine similarity threshold above which two documents are
            considered near-duplicates of each other.

    Returns:
        A dict mapping each doc_id that has at least one near-duplicate to
        the sorted list of doc_ids it is a near-duplicate of. doc_ids with
        no flagged duplicates are omitted from the dict.
    """
    n = len(doc_ids)
    if n != len(embeddings):
        raise ValueError("doc_ids and embeddings must have the same length")

    if n < 2:
        return {}

    vectors = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = vectors / norms

    similarity = normalized @ normalized.T

    duplicates: Dict[str, List[str]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] >= threshold:
                duplicates.setdefault(doc_ids[i], []).append(doc_ids[j])
                duplicates.setdefault(doc_ids[j], []).append(doc_ids[i])

    for doc_id, others in duplicates.items():
        duplicates[doc_id] = sorted(set(others))

    return duplicates
