"""Fetch documents from the Crossref API (https://api.crossref.org) and write
them as generic Document-schema JSONL records.

Crossref is a free, no-API-key index of scholarly metadata covering tens of
millions of journal articles, books, and reports across every field. Many
records include an abstract (in JATS XML), making it a good broad-coverage
supplement to OpenAlex / ERIC / DOAJ for any domain.

Usage:
    python scripts/fetch_crossref.py --query "biodiversity conservation" \
        --rows 100 --pages 3 --domain environment --out data/processed/corpus.jsonl \
        --mailto you@example.org
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

CROSSREF_API_URL = "https://api.crossref.org/works"

_TAG_RE = re.compile(r"<[^>]+>")


def fetch_query(query: str, rows: int, offset: int = 0, mailto: str = None) -> list:
    params = {
        "query": query,
        "rows": min(rows, 100),
        "offset": offset,
        "filter": "has-abstract:true",
    }
    if mailto:
        params["mailto"] = mailto
    response = requests.get(CROSSREF_API_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("items", [])


def _clean_abstract(raw: str) -> str:
    if not raw:
        return ""
    return _TAG_RE.sub("", raw).replace("\n", " ").strip()


def _extract_year(item: dict) -> int:
    for key in ("published-print", "published-online", "issued", "created"):
        date_parts = (item.get(key) or {}).get("date-parts")
        if date_parts and date_parts[0] and date_parts[0][0]:
            try:
                return int(date_parts[0][0])
            except (TypeError, ValueError):
                continue
    return 0


def to_document(item: dict, domain: str = None) -> dict:
    doi = item.get("DOI", "")
    titles = item.get("title") or []
    title = titles[0] if titles else ""
    abstract = _clean_abstract(item.get("abstract", ""))

    authors = []
    for author in item.get("author", []) or []:
        name = " ".join(p for p in (author.get("given"), author.get("family")) if p)
        if name:
            authors.append(name)

    tags = item.get("subject", []) or []
    year = _extract_year(item)
    url = f"https://doi.org/{doi}" if doi else None

    return {
        "doc_id": f"crossref:{doi}",
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "source": "crossref",
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
    parser = argparse.ArgumentParser(description="Fetch documents from Crossref")
    parser.add_argument("--query", action="append", required=True, dest="queries",
                         help="Search query (repeatable for multiple topics)")
    parser.add_argument("--rows", type=int, default=50,
                         help="Number of results to fetch per query, per page (default: 50, max 100)")
    parser.add_argument("--pages", type=int, default=1,
                         help="Number of pages to fetch per query (default: 1). Each page "
                              "advances 'offset' by --rows.")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--sleep", type=float, default=0.5,
                         help="Seconds to sleep between requests (default: 0.5)")
    parser.add_argument("--domain", default=None,
                         help="Tag fetched documents with this domain (e.g. environment)")
    parser.add_argument("--mailto", default=None,
                         help="Contact email for Crossref's polite pool (faster, more reliable)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = load_existing_ids(out_path)
    documents = []

    for query in args.queries:
        for page in range(args.pages):
            offset = page * args.rows
            print(f"Fetching Crossref results for: {query!r} (offset {offset})", file=sys.stderr)
            try:
                items = fetch_query(query, args.rows, offset=offset, mailto=args.mailto)
            except requests.RequestException as exc:
                print(f"  request failed: {exc}", file=sys.stderr)
                break

            if not items:
                break

            for item in items:
                record = to_document(item, domain=args.domain)
                if not record["title"] or not record["abstract"]:
                    continue
                if record["doc_id"] in seen_ids:
                    continue
                seen_ids.add(record["doc_id"])
                documents.append(record)

            print(f"  got {len(items)} results, {len(documents)} new unique so far", file=sys.stderr)
            time.sleep(args.sleep)

    with open(out_path, "a", encoding="utf-8") as f:
        for record in documents:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(documents)} documents to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
