"""Fetch documents from the Europe PMC API (https://europepmc.org) and write
them as generic Document-schema JSONL records.

Europe PMC is a free, no-API-key literature database covering life sciences,
biomedical, and environmental health research. It's a good complement to
DOAJ for the Environment & Conservation domain, surfacing peer-reviewed
ecology, public-health, and climate-adaptation literature.

Usage:
    python scripts/fetch_europepmc.py --query "community-based conservation" \
        --rows 30 --domain environment --out data/processed/corpus.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

EUROPEPMC_API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def fetch_query(query: str, rows: int) -> list:
    params = {
        "query": query,
        "format": "json",
        "pageSize": min(rows, 1000),
        "resultType": "core",
    }
    response = requests.get(EUROPEPMC_API_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data.get("resultList", {}).get("result", [])


def to_document(result: dict, domain: str = None) -> dict:
    pmc_id = result.get("id", "")
    source = result.get("source", "")
    title = result.get("title", "") or ""
    abstract = result.get("abstractText", "") or ""

    author_string = result.get("authorString", "") or ""
    authors = [a.strip() for a in author_string.split(",") if a.strip()]

    tags = [
        kw for kw in (result.get("keywordList", {}) or {}).get("keyword", []) or []
        if isinstance(kw, str)
    ]

    year_raw = result.get("pubYear")
    try:
        year = int(year_raw)
    except (TypeError, ValueError):
        year = 0

    url = None
    full_text_urls = (result.get("fullTextUrlList") or {}).get("fullTextUrl") or []
    for entry in full_text_urls:
        if entry.get("url"):
            url = entry["url"]
            break
    if not url and result.get("doi"):
        url = f"https://doi.org/{result['doi']}"

    return {
        "doc_id": f"europepmc:{source}:{pmc_id}",
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "source": "europepmc",
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
    parser = argparse.ArgumentParser(description="Fetch documents from Europe PMC")
    parser.add_argument("--query", action="append", required=True, dest="queries",
                         help="Search query (repeatable for multiple topics)")
    parser.add_argument("--rows", type=int, default=30,
                         help="Number of results to fetch per query (default: 30)")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--sleep", type=float, default=0.5,
                         help="Seconds to sleep between requests (default: 0.5)")
    parser.add_argument("--domain", default=None,
                         help="Tag fetched documents with this domain (e.g. environment)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = load_existing_ids(out_path)
    documents = []

    for query in args.queries:
        print(f"Fetching Europe PMC results for: {query!r}", file=sys.stderr)
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

        print(f"  got {len(results)} results, {len(documents)} new unique so far", file=sys.stderr)
        time.sleep(args.sleep)

    with open(out_path, "a", encoding="utf-8") as f:
        for record in documents:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(documents)} documents to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
