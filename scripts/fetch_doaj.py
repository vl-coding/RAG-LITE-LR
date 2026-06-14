"""Fetch documents from the DOAJ API (https://doaj.org) and write them as
generic Document-schema JSONL records.

DOAJ (Directory of Open Access Journals) is a free, no-API-key, multi-domain
index of open-access journal articles. It covers a much broader range of
subjects than ERIC, but each request is slow (often 30-45 seconds), so use
modest --rows values and expect this script to take a while for multiple
queries.

Usage:
    python scripts/fetch_doaj.py --query "biodiversity conservation" \
        --rows 30 --out data/processed/corpus.jsonl
"""
import argparse
import json
import sys
import time
import urllib.parse

import requests
from pathlib import Path

DOAJ_API_URL = "https://doaj.org/api/search/articles/{query}"


def fetch_query(query: str, rows: int) -> list:
    url = DOAJ_API_URL.format(query=urllib.parse.quote(query))
    response = requests.get(url, params={"pageSize": rows}, timeout=90)
    response.raise_for_status()
    data = response.json()
    return data.get("results", [])


def to_document(result: dict, domain: str = None) -> dict:
    bibjson = result.get("bibjson", {})
    doaj_id = result.get("id", "")
    title = bibjson.get("title", "") or ""
    abstract = bibjson.get("abstract", "") or ""

    authors = [a.get("name", "") for a in bibjson.get("author", []) if a.get("name")]

    subjects = [s.get("term", "") for s in bibjson.get("subject", []) if s.get("term")]
    keywords = bibjson.get("keyword", []) or []
    tags = subjects + [k for k in keywords if isinstance(k, str)]

    year_raw = bibjson.get("year")
    try:
        year = int(year_raw)
    except (TypeError, ValueError):
        year = 0

    url = None
    for link in bibjson.get("link", []) or []:
        if link.get("url"):
            url = link["url"]
            break

    return {
        "doc_id": f"doaj:{doaj_id}",
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "source": "doaj",
        "tags": tags,
        "year": year,
        "url": url,
        "domain": domain,
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch documents from DOAJ")
    parser.add_argument("--query", action="append", required=True, dest="queries",
                         help="Search query (repeatable for multiple topics)")
    parser.add_argument("--rows", type=int, default=30,
                         help="Number of results to fetch per query (default: 30)")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--sleep", type=float, default=1.0,
                         help="Seconds to sleep between requests (default: 1.0)")
    parser.add_argument("--domain", default=None,
                         help="Tag fetched documents with this domain (e.g. environment)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = set()
    documents = []

    for query in args.queries:
        print(f"Fetching DOAJ results for: {query!r} (this can take ~30-45s)", file=sys.stderr)
        try:
            results = fetch_query(query, args.rows)
        except requests.RequestException as exc:
            print(f"  request failed: {exc}", file=sys.stderr)
            continue

        for result in results:
            record = to_document(result, domain=args.domain)
            if not record["title"] or not record["abstract"]:
                continue
            if record["doc_id"] in seen_ids:
                continue
            seen_ids.add(record["doc_id"])
            documents.append(record)

        print(f"  got {len(results)} results, {len(documents)} total unique so far", file=sys.stderr)
        time.sleep(args.sleep)

    with open(out_path, "a", encoding="utf-8") as f:
        for record in documents:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(documents)} documents to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
