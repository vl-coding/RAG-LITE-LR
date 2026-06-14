"""Retrieval evaluation harness for RAG-LITE-LR.

Runs the gold query set (tests/eval/gold_queries.yaml) through the pipeline
and reports end-to-end Precision/Recall/NDCG/MRR @k of the final fused top-k
against the gold relevant_ids, plus (optionally) the distribution of Claude's
relevance/specificity scores.

Usage:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --top-k 5 --no-justification
    python scripts/evaluate_retrieval.py --output outputs/eval_report.json
"""
import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_lite.config import load_config, ensure_project_dirs
from rag_lite.eval_metrics import (
    load_gold_queries,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from rag_lite.pipeline import RagLitePipeline


def _describe(values: List[float]) -> Dict[str, Optional[float]]:
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": None, "stdev": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 3),
        "stdev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _print_table(headers: List[str], rows: List[List[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def _fmt(x, digits=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def evaluate_query(pipeline, entry, top_k, use_justification):
    relevant_ids = set(entry["relevant_ids"])
    relevance = {doc_id: 1.0 for doc_id in relevant_ids}

    response = pipeline.run(
        query=entry["query"],
        top_k=top_k,
        use_justification=use_justification,
        domain=entry.get("domain"),
    )

    retrieved_ids = [r.doc_id for r in response.results]

    result = {
        "query": entry["query"],
        "relevant_ids": sorted(relevant_ids),
        "e2e": {
            "precision@k": precision_at_k(retrieved_ids, relevant_ids, top_k),
            "recall@k": recall_at_k(retrieved_ids, relevant_ids, top_k),
            "ndcg@k": ndcg_at_k(retrieved_ids, relevance, top_k),
            "mrr": mrr(retrieved_ids, relevant_ids),
            "hits": [doc_id for doc_id in retrieved_ids if doc_id in relevant_ids],
            "retrieved_ids": retrieved_ids,
        },
    }

    if use_justification:
        result["justifier_records"] = [
            {
                "doc_id": r.doc_id,
                "is_known_relevant": r.doc_id in relevant_ids,
                "relevance_score": r.relevance_score,
                "specificity_score": r.specificity_score,
            }
            for r in response.results
        ]

    return result


def report_e2e(per_query_results, top_k):
    print(f"\n=== End-to-end relevance (top-{top_k} vs. gold relevant_ids) ===")
    rows = []
    for r in per_query_results:
        e = r["e2e"]
        rows.append([
            r["query"][:50],
            _fmt(e["precision@k"]), _fmt(e["recall@k"]), _fmt(e["ndcg@k"]), _fmt(e["mrr"]),
            ",".join(e["hits"]) or "-",
        ])
    _print_table(
        ["query", f"P@{top_k}", f"R@{top_k}", f"NDCG@{top_k}", "MRR", "hits"],
        rows,
    )
    for metric in ("precision@k", "recall@k", "ndcg@k", "mrr"):
        mean_val = statistics.mean(r["e2e"][metric] for r in per_query_results)
        print(f"mean {metric}={mean_val:.3f}", end="  ")
    print()


def report_calibration(per_query_results):
    all_records = [
        rec
        for result in per_query_results
        for rec in result.get("justifier_records", [])
    ]
    rel_scores = [r["relevance_score"] for r in all_records]
    spec_scores = [r["specificity_score"] for r in all_records]

    print("\n=== Justifier score calibration ===")
    print("Score distribution across all top-k results:")
    for name, values in (("relevance_score", rel_scores), ("specificity_score", spec_scores)):
        stats = _describe(values)
        print(f"  {name}: n={stats['n']} mean={_fmt(stats['mean'])} stdev={_fmt(stats['stdev'])} "
              f"min={_fmt(stats['min'])} max={_fmt(stats['max'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", default="tests/eval/gold_queries.yaml")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-justification", action="store_true")
    parser.add_argument("--output", default="outputs/eval_report.json")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    ensure_project_dirs(config)

    top_k = args.top_k if args.top_k is not None else config["retrieval"].get("default_top_k", 5)
    use_justification = not args.no_justification

    gold = load_gold_queries(args.gold)
    if not gold:
        raise SystemExit(f"No gold queries loaded from {args.gold}")

    print(f"Loaded {len(gold)} gold queries from {args.gold}")
    print(f"top_k={top_k} use_justification={use_justification}")

    pipeline = RagLitePipeline(config)

    per_query_results = []
    for i, entry in enumerate(gold, start=1):
        print(f"\n[{i}/{len(gold)}] {entry['query']!r}")
        result = evaluate_query(pipeline, entry, top_k=top_k, use_justification=use_justification)
        per_query_results.append(result)

    report = {"config": vars(args), "per_query": per_query_results}

    report_e2e(per_query_results, top_k)
    if use_justification:
        report_calibration(per_query_results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved full report to {output_path}")


if __name__ == "__main__":
    main()
