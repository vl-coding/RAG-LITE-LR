"""Fetch documents from the ERIC API (https://eric.ed.gov) and write them as
generic Document-schema JSONL records.

ERIC is a free, no-API-key education research database maintained by the
US Department of Education's Institute of Education Sciences. It is fast
and reliable, making it a good default corpus source for education-focused
non-profit deployments.

Usage:
    python scripts/fetch_eric.py --query "early literacy intervention" \
        --query "teacher professional development" \
        --rows 50 --out data/processed/corpus.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

ERIC_API_URL = "https://api.ies.ed.gov/eric/"


def fetch_query(query: str, rows: int, start: int = 0) -> list:
    params = {
        "search": query,
        "format": "json",
        "rows": rows,
        "start": start,
    }
    response = requests.get(ERIC_API_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data.get("response", {}).get("docs", [])


def to_document(doc: dict, domain: str = None) -> dict:
    eric_id = doc.get("id", "")
    title = doc.get("title", "")
    abstract = doc.get("description", "") or ""
    authors = doc.get("author", []) or []
    subjects = doc.get("subject", []) or []
    year_raw = doc.get("publicationdateyear")
    try:
        year = int(year_raw)
    except (TypeError, ValueError):
        year = 0

    return {
        "doc_id": f"eric:{eric_id}",
        "title": title,
        "abstract": abstract,
        "authors": authors if isinstance(authors, list) else [authors],
        "source": "eric",
        "tags": subjects if isinstance(subjects, list) else [subjects],
        "year": year,
        "url": f"https://eric.ed.gov/?id={eric_id}" if eric_id else None,
        "domain": domain,
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch documents from ERIC")
    parser.add_argument("--query", action="append", required=True, dest="queries",
                         help="Search query (repeatable for multiple topics)")
    parser.add_argument("--rows", type=int, default=50,
                         help="Number of results to fetch per query (default: 50)")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--sleep", type=float, default=0.5,
                         help="Seconds to sleep between requests (default: 0.5)")
    parser.add_argument("--domain", default=None,
                         help="Tag fetched documents with this domain (e.g. education)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = set()
    documents = []

    for query in args.queries:
        print(f"Fetching ERIC results for: {query!r}", file=sys.stderr)
        try:
            docs = fetch_query(query, args.rows)
        except requests.RequestException as exc:
            print(f"  request failed: {exc}", file=sys.stderr)
            continue

        for doc in docs:
            record = to_document(doc, domain=args.domain)
            if not record["title"] or not record["abstract"]:
                continue
            if record["doc_id"] in seen_ids:
                continue
            seen_ids.add(record["doc_id"])
            documents.append(record)

        print(f"  got {len(docs)} results, {len(documents)} total unique so far", file=sys.stderr)
        time.sleep(args.sleep)

    with open(out_path, "a", encoding="utf-8") as f:
        for record in documents:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(documents)} documents to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
