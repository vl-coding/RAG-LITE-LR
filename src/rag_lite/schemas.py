from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    title: str
    abstract: str
    authors: List[str] = []
    source: Optional[str] = None
    tags: List[str] = []
    year: int
    url: Optional[str] = None
    domain: Optional[str] = None

    @property
    def text(self) -> str:
        return f"{self.title}\n\n{self.abstract}"


class DocumentResult(BaseModel):
    rank: int
    doc_id: str
    title: str
    authors: List[str] = []
    year: int
    source: Optional[str] = None
    tags: List[str] = []
    url: Optional[str] = None
    domain: Optional[str] = None
    abstract_snippet: str = ""
    rrf_score: float = Field(
        description=(
            "Reciprocal Rank Fusion score used to order results. This is an "
            "ordinal fusion score, not a relevance probability, and is not "
            "comparable across different queries. Use `rank` for the "
            "result's position in this result set."
        )
    )
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    contribution: Optional[str] = None
    relevance_justification: Optional[str] = None
    relevance_score: Optional[float] = None
    specificity_score: Optional[float] = None
    possible_duplicate_of: Optional[List[str]] = Field(
        default=None,
        description=(
            "doc_ids of other documents in this same result set whose "
            "title+abstract embedding is near-identical to this document's "
            "(cosine similarity above the near-duplicate threshold). None "
            "or empty if no near-duplicates were found in this result set."
        ),
    )


class RetrievalTrace(BaseModel):
    total_corpus_size: int
    candidate_pool_size: Optional[int] = None
    hyde_document: str = ""
    dense_latency_seconds: float = 0.0
    bm25_latency_seconds: float = 0.0
    total_latency_seconds: float = 0.0


class SearchResponse(BaseModel):
    query: str
    results: List[DocumentResult] = []
    trace: RetrievalTrace
    metadata: Dict[str, Any] = {}
