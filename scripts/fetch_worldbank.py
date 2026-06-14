"""Fetch documents from the World Bank Documents & Reports API
(https://search.worldbank.org/api/v3/wds) and write them as generic
Document-schema JSONL records.

This is a free, no-API-key catalog of World Bank reports, working papers,
and project documents -- including a large body of program/impact
evaluations from low- and middle-income countries. It's a strong source for
the Research domain (evaluation methodology and findings) and can also
supplement Environment (climate/conservation reports) and Education
(school program evaluations).

Usage:
    python scripts/fetch_worldbank.py --query "impact evaluation community program" \
        --rows 20 --domain research --out data/processed/corpus.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

WORLDBANK_API_URL = "https://search.worldbank.org/api/v3/wds"


def fetch_query(query: str, rows: int) -> list:
    params = {
        "format": "json",
        "qterm": query,
        "rows": min(rows, 100),
        "fl": "docna,display_title,abstracts,authors,docdt,url,majdocty,historic_topic",
    }
    response = requests.get(WORLDBANK_API_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    documents = data.get("documents", {}) or {}
    return [doc for doc in documents.values() if isinstance(doc, dict) and doc.get("id")]


def _extract_abstract(abstracts) -> str:
    if not abstracts:
        return ""
    if isinstance(abstracts, dict):
        if "cdata!" in abstracts:
            return abstracts["cdata!"]
        parts = []
        for value in abstracts.values():
            if isinstance(value, dict) and "cdata!" in value:
                parts.append(value["cdata!"])
        return " ".join(parts)
    return ""


def to_document(doc: dict, domain: str = None) -> dict:
    wb_id = doc.get("id", "")
    title = doc.get("display_title") or ""
    abstract = _extract_abstract(doc.get("abstracts"))

    authors = []
    for value in (doc.get("authors") or {}).values():
        if isinstance(value, dict) and value.get("author"):
            authors.append(value["author"])

    docdt = doc.get("docdt", "") or ""
    try:
        year = int(docdt[:4])
    except (TypeError, ValueError):
        year = 0

    tags = []
    for key in ("majdocty", "historic_topic"):
        val = doc.get(key)
        if isinstance(val, str) and val:
            tags.append(val)

    url = doc.get("url")

    return {
        "doc_id": f"worldbank:{wb_id}",
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "source": "worldbank",
        "tags": tags,
        "year": year,
        "url": url,
        "domain": domain,
    }


def load_existing_ids(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    seen = set()
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seen.add(json.loads(line).get("doc_id"))
    return seen


def main():
    parser = argparse.ArgumentParser(description="Fetch documents from World Bank Documents & Reports")
    parser.add_argument("--query", action="append", required=True, dest="queries",
                         help="Search query (repeatable for multiple topics)")
    parser.add_argument("--rows", type=int, default=20,
                         help="Number of results to fetch per query (default: 20)")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--sleep", type=float, default=1.0,
                         help="Seconds to sleep between requests (default: 1.0)")
    parser.add_argument("--domain", default=None,
                         help="Tag fetched documents with this domain (e.g. research)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = load_existing_ids(out_path)
    documents = []

    for query in args.queries:
        print(f"Fetching World Bank results for: {query!r}", file=sys.stderr)
        try:
            results = fetch_query(query, args.rows)
        except requests.RequestException as exc:
            print(f"  request failed: {exc}", file=sys.stderr)
            continue

        for doc in results:
            record = to_document(doc, domain=args.domain)
            if not record["title"] or not record["abstract"]:
                continue
            if record["doc_id"] in seen_ids:
                continue
            seen_ids.add(record["doc_id"])
            documents.append(record)

        print(f"  got {len(results)} results, {len(documents)} new unique so far", file=sys.stderr)
        time.sleep(args.sleep)

    with open(out_path, "a", encoding="utf-8") as f:
        for record in documents:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(documents)} documents to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
