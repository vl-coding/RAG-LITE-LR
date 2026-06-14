"""Fetch documents from the PLOS Search API (https://api.plos.org) and write
them as generic Document-schema JSONL records.

PLOS (Public Library of Science) is a free, no-API-key, open-access
publisher spanning health, medicine, climate, and social science journals
(PLOS ONE, PLOS Medicine, PLOS Climate, PLOS Global Public Health, etc.).
Abstracts are reliably included, making it a good complement to DOAJ /
OpenAlex / Europe PMC for the Environment and Research domains.

Usage:
    python scripts/fetch_plos.py --query "community health program evaluation" \
        --rows 20 --domain research --out data/processed/corpus.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

PLOS_API_URL = "https://api.plos.org/search"


def fetch_query(query: str, rows: int) -> list:
    params = {
        "q": query,
        "rows": min(rows, 100),
        "fl": "id,title,abstract,author_display,publication_date,journal",
    }
    response = requests.get(PLOS_API_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data.get("response", {}).get("docs", [])


def to_document(doc: dict, domain: str = None) -> dict:
    doi = doc.get("id", "")
    title = doc.get("title", "") or ""

    abstract_field = doc.get("abstract") or []
    abstract = " ".join(a.strip() for a in abstract_field if a) if abstract_field else ""

    authors = doc.get("author_display", []) or []

    pub_date = doc.get("publication_date", "") or ""
    try:
        year = int(pub_date[:4])
    except (TypeError, ValueError):
        year = 0

    tags = [doc["journal"]] if doc.get("journal") else []

    url = f"https://doi.org/{doi}" if doi else None

    return {
        "doc_id": f"plos:{doi}",
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "source": "plos",
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
    parser = argparse.ArgumentParser(description="Fetch documents from PLOS")
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
        print(f"Fetching PLOS results for: {query!r}", file=sys.stderr)
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
