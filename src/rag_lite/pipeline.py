import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .schemas import Document, DocumentResult, RetrievalTrace, SearchResponse
from .dedup import find_near_duplicates, DEFAULT_NEAR_DUPLICATE_THRESHOLD
from .hyde import ClaudeHyDE
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from .rrf import reciprocal_rank_fusion
from .justifier import ClaudeJustifier


def load_documents(jsonl_path: str) -> List[Document]:
    documents = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            documents.append(Document.model_validate_json(line))
    return documents


class RagLitePipeline:
    def __init__(self, config: dict):
        self.config = config

        print("Loading corpus ...", flush=True)
        self.documents = load_documents(config["data"]["processed_path"])
        self.documents_by_id: Dict[str, Document] = {d.doc_id: d for d in self.documents}
        print(f"Corpus ready: {len(self.documents):,} documents", flush=True)

        claude_max_retries = config["models"].get("claude_max_retries", 5)
        claude_timeout_seconds = config["models"].get("claude_timeout_seconds", 60.0)

        self.hyde = ClaudeHyDE(
            config["models"]["claude_model"], claude_max_retries, claude_timeout_seconds
        )
        self.justifier = ClaudeJustifier(
            config["models"]["claude_model"], claude_max_retries, claude_timeout_seconds
        )
        self._justifier_max_concurrency = config["models"].get(
            "claude_justifier_max_concurrency", 5
        )

        self.dense = DenseRetriever(
            model_name=config["models"]["embedding_model"],
            persist_dir=config["paths"]["dense_index_dir"],
        )
        self.dense.load()

        self.bm25 = BM25Retriever.load(config["paths"]["bm25_index"])

    def run(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_hyde: Optional[bool] = None,
        use_justification: Optional[bool] = None,
        domain: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> SearchResponse:
        def _report(step: str, fraction: float) -> None:
            if progress_callback is not None:
                progress_callback(step, fraction)

        retrieval_cfg = self.config["retrieval"]
        top_k = top_k if top_k is not None else retrieval_cfg.get("default_top_k", 5)
        use_hyde = use_hyde if use_hyde is not None else retrieval_cfg.get("use_hyde", True)
        use_justification = (
            use_justification
            if use_justification is not None
            else retrieval_cfg.get("use_justification", True)
        )

        start_total = time.time()

        candidate_ids = None
        if domain:
            candidate_ids = {
                doc_id for doc_id, doc in self.documents_by_id.items() if doc.domain == domain
            }

        _report("Generating hypothetical document ...", 0.1)
        hyde_document = ""
        dense_query_text = query
        if use_hyde:
            hyde_document = self.hyde.generate(query)
            dense_query_text = hyde_document

        _report("Running dense vector search ...", 0.35)
        dense_start = time.time()
        dense_results = self.dense.search(
            query_text=dense_query_text,
            candidate_ids=candidate_ids,
            top_n=retrieval_cfg.get("dense_candidates", 50),
        )
        dense_latency = time.time() - dense_start

        _report("Running BM25 keyword search ...", 0.55)
        bm25_start = time.time()
        bm25_results = self.bm25.search(
            query=query,
            candidate_ids=candidate_ids,
            top_n=retrieval_cfg.get("bm25_candidates", 50),
        )
        bm25_latency = time.time() - bm25_start

        _report("Fusing rankings (RRF) ...", 0.7)
        fused = reciprocal_rank_fusion(
            [dense_results, bm25_results], k=retrieval_cfg.get("rrf_k", 60)
        )
        top_items = fused[:top_k]

        _report("Loading document details ...", 0.75)
        top_docs = {item["doc_id"]: self.documents_by_id[item["doc_id"]] for item in top_items}

        justifications = {}
        if use_justification:
            _report("Generating relevance justifications ...", 0.85)

            def _justify(item):
                doc = top_docs[item["doc_id"]]
                return item["doc_id"], self.justifier.justify(
                    query=query,
                    title=doc.title,
                    abstract=doc.abstract,
                )

            justify_workers = min(max(len(top_items), 1), self._justifier_max_concurrency)
            with ThreadPoolExecutor(max_workers=justify_workers) as executor:
                futures = {executor.submit(_justify, item): item for item in top_items}
                for future in as_completed(futures):
                    doc_id, result = future.result()
                    justifications[doc_id] = result

        near_duplicates: Dict[str, List[str]] = {}
        if len(top_items) >= 2:
            dedup_ids = [item["doc_id"] for item in top_items]
            dedup_texts = [top_docs[doc_id].text for doc_id in dedup_ids]
            dedup_embeddings = self.dense.model.encode(
                dedup_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            near_duplicates = find_near_duplicates(
                dedup_ids,
                dedup_embeddings,
                threshold=retrieval_cfg.get(
                    "near_duplicate_threshold", DEFAULT_NEAR_DUPLICATE_THRESHOLD
                ),
            )

        final_results = []
        for item in top_items:
            doc = top_docs[item["doc_id"]]
            justification = justifications.get(item["doc_id"], {})
            final_results.append(
                DocumentResult(
                    rank=item["rank"],
                    doc_id=doc.doc_id,
                    title=doc.title,
                    authors=doc.authors,
                    year=doc.year,
                    source=doc.source,
                    tags=doc.tags,
                    url=doc.url,
                    domain=doc.domain,
                    abstract_snippet=doc.abstract[:500],
                    rrf_score=item["rrf_score"],
                    dense_rank=item.get("dense_rank"),
                    bm25_rank=item.get("bm25_rank"),
                    contribution=justification.get("contribution"),
                    relevance_justification=justification.get("relevance_justification"),
                    relevance_score=justification.get("relevance_score"),
                    specificity_score=justification.get("specificity_score"),
                    possible_duplicate_of=near_duplicates.get(item["doc_id"]),
                )
            )

        trace = RetrievalTrace(
            total_corpus_size=len(self.documents),
            candidate_pool_size=len(candidate_ids) if candidate_ids is not None else None,
            hyde_document=hyde_document,
            dense_latency_seconds=round(dense_latency, 3),
            bm25_latency_seconds=round(bm25_latency, 3),
            total_latency_seconds=round(time.time() - start_total, 3),
        )

        _report("Done", 1.0)

        return SearchResponse(
            query=query,
            results=final_results,
            trace=trace,
            metadata={
                "pipeline_version": "lite-v1",
                "retrieval_method": "hyde + dense + bm25 + rrf" if use_hyde else "dense + bm25 + rrf",
            },
        )
