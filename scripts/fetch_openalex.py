"""Fetch documents from the OpenAlex API (https://openalex.org) and write
them as generic Document-schema JSONL records.

OpenAlex is a free, no-API-key, multidisciplinary index of scholarly works
covering virtually every field (sciences, social sciences, humanities). It's
a good general-purpose source for broadening domains that aren't well served
by ERIC (education-only) or DOAJ alone -- e.g. Research and Arts & Culture.

Usage:
    python scripts/fetch_openalex.py --query "community-based participatory research" \
        --rows 30 --domain research --out data/processed/corpus.jsonl \
        --mailto you@example.org
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

OPENALEX_API_URL = "https://api.openalex.org/works"


def fetch_query(query: str, rows: int, mailto: str = None, page: int = 1) -> list:
    params = {
        "search": query,
        "per_page": min(rows, 200),
        "filter": "has_abstract:true",
        "page": page,
    }
    if mailto:
        params["mailto"] = mailto
    response = requests.get(OPENALEX_API_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data.get("results", [])


def _reconstruct_abstract(inverted_index: dict) -> str:
    if not inverted_index:
        return ""
    positions = []
    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions.append((idx, word))
    positions.sort()
    return " ".join(word for _, word in positions)


def to_document(work: dict, domain: str = None) -> dict:
    openalex_id = (work.get("id") or "").rsplit("/", 1)[-1]
    title = work.get("title") or work.get("display_name") or ""
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))

    authors = [
        a.get("author", {}).get("display_name", "")
        for a in work.get("authorships", []) or []
        if a.get("author", {}).get("display_name")
    ]

    tags = [
        c.get("display_name", "")
        for c in work.get("concepts", []) or []
        if c.get("display_name")
    ]

    year = work.get("publication_year") or 0

    primary_location = work.get("primary_location") or {}
    url = primary_location.get("landing_page_url")
    if not url and work.get("doi"):
        url = work["doi"]

    return {
        "doc_id": f"openalex:{openalex_id}",
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "source": "openalex",
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
    parser = argparse.ArgumentParser(description="Fetch documents from OpenAlex")
    parser.add_argument("--query", action="append", required=True, dest="queries",
                         help="Search query (repeatable for multiple topics)")
    parser.add_argument("--rows", type=int, default=30,
                         help="Number of results to fetch per query, per page (default: 30, max 200)")
    parser.add_argument("--pages", type=int, default=1,
                         help="Number of pages to fetch per query (default: 1). Each page returns "
                              "up to --rows results; page * rows must stay under OpenAlex's 10,000 "
                              "result pagination cap.")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--sleep", type=float, default=0.5,
                         help="Seconds to sleep between requests (default: 0.5)")
    parser.add_argument("--domain", default=None,
                         help="Tag fetched documents with this domain (e.g. research)")
    parser.add_argument("--mailto", default=None,
                         help="Contact email for OpenAlex's polite pool (faster, more reliable)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = load_existing_ids(out_path)
    documents = []

    for query in args.queries:
        for page in range(1, args.pages + 1):
            print(f"Fetching OpenAlex results for: {query!r} (page {page})", file=sys.stderr)
            try:
                results = fetch_query(query, args.rows, mailto=args.mailto, page=page)
            except requests.RequestException as exc:
                print(f"  request failed: {exc}", file=sys.stderr)
                break

            if not results:
                break

            for work in results:
                record = to_document(work, domain=args.domain)
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
