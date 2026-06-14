# RAG-LITE-LR

**[Live demo](https://rag-lite-lr-4h68wkyewerffljjz9c9p3.streamlit.app/)**

A lightweight literature-search assistant for organizations with limited
technical resources — built for non-profits, advocacy groups, and small
research teams who need to search a focused document collection (grant
reports, program evaluations, journal articles) and get plain-language
relevance notes back.

> **New to this project?** See [FOR_NONPROFITS.md](FOR_NONPROFITS.md) for a
> non-technical overview of what this tool does, when to use it, and what
> the results mean.

RAG-LITE-LR is a stripped-down sibling of
[RAG-L-R-Assistant (RAGLR-A)](https://github.com/vl-coding/RAGLR-A), which
indexes the full ~3M-paper arXiv corpus. RAG-LITE-LR instead points at a
**small, topic-specific corpus you build yourself** (tens to tens-of-thousands
of documents) from open, no-API-key sources like [ERIC](https://eric.ed.gov)
(education research) and [DOAJ](https://doaj.org) (open-access journals
across all fields).

## How it works

1. You build a corpus: fetch documents on your topic(s) from ERIC/DOAJ (or
   bring your own JSONL).
2. `build_index.py` builds a dense (SBERT embeddings) and BM25 (keyword)
   index over that corpus — both stored as plain files, no database server.
3. For each query: Claude writes a hypothetical "ideal document" (HyDE),
   which is used for dense search; BM25 runs the raw query; both ranked
   lists are combined with Reciprocal Rank Fusion (RRF); the top results get
   a short Claude-written relevance note.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY

# 1. Build a corpus (education example via ERIC, no API key needed)
python scripts/fetch_eric.py \
  --query "early literacy intervention" \
  --query "teacher professional development" \
  --query "school-based mental health" \
  --rows 50 \
  --out data/processed/corpus.jsonl

# 2. Build the search indexes
python scripts/build_index.py

# 3. Run a query from the command line
python scripts/run_query.py --query "What programs improve early literacy outcomes for low-income students?"

# ... or launch the web UI
streamlit run app/streamlit_app.py
```

## Evaluation

`tests/eval/gold_queries.yaml` holds a small set of test queries (phrased the
way program staff at a non-profit would actually ask them) with known-relevant
document ids. Run:

```bash
python scripts/evaluate_retrieval.py
```

This reports Precision/Recall/NDCG/MRR @k of the final results against the
gold set, plus the distribution of Claude's relevance/specificity scores.

## Cost controls

- `retrieval.default_top_k` (default 5) — fewer results = fewer Claude
  justification calls per query.
- `retrieval.use_hyde` — HyDE adds one Claude call per query; disable for a
  pure embedding+BM25 search.
- `retrieval.use_justification` — disable to skip per-result Claude calls
  entirely (results are still ranked, just without relevance notes).
- `models.claude_model` — defaults to a Haiku-class model to keep
  per-query cost low.

## Docker

```bash
docker compose up --build
```

## Bring your own corpus

Any JSONL file with one JSON object per line and these fields works:

```json
{"doc_id": "source:123", "title": "...", "abstract": "...", "authors": ["..."], "source": "my-source", "tags": ["..."], "year": 2024, "url": "https://...", "domain": "education"}
```

`domain` is optional but powers the domain filter described below. Point
`data.processed_path` in `configs/config.yaml` at it and run `build_index.py`.

## Multi-domain corpora and the domain filter

A single corpus can mix documents from multiple subject areas by tagging
each `Document` with a `domain` (e.g. `education`, `environment`,
`research`, `arts_culture`). Both `fetch_eric.py` and `fetch_doaj.py` accept
a `--domain` flag that tags every record they write:

```bash
python scripts/fetch_doaj.py --domain environment \
  --query "biodiversity conservation community programs" \
  --rows 25 --out data/processed/corpus.jsonl
```

At query time, the Streamlit UI's "Research domain" selector (and
`run_query.py --domain <name>` / `pipeline.run(..., domain=<name>)`)
restricts dense and BM25 search to documents with that `domain` value —
indexes are built once over the whole corpus, and the filter is applied via
`candidate_ids` at search time (no per-domain rebuild needed). The response's
`trace.candidate_pool_size` reports how many documents the domain filter
restricted the search to, out of `trace.total_corpus_size` overall.

## Relationship to RAG-L-R-Assistant

| | RAGLR-A | RAG-LITE-LR |
|---|---|---|
| Corpus | ~3M arXiv papers | Your own (hundreds–tens of thousands) |
| Dense index | ChromaDB | In-memory numpy (no DB server) |
| Prefilter | Qwen keyword extraction | none (corpus is already small) |
| Setup | GPU recommended for full reindex | Runs on a laptop |
| Best for | Broad CS/ML/quant literature reviews | A specific program area, topic, or grant portfolio |

## Related projects

- [RAG-L-R-Assistant (RAGLR-A)](https://github.com/vl-coding/RAGLR-A) — the
  larger, research-focused sibling of this project, indexing the full ~3M-paper
  arXiv corpus.

---

Built by [vl-coding](https://github.com/vl-coding).
