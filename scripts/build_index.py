"""Build the dense (numpy) and BM25 indexes from data/processed/corpus.jsonl.

Usage:
    python scripts/build_index.py --config configs/config.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_lite.config import load_config, ensure_project_dirs
from rag_lite.dense_retriever import DenseRetriever
from rag_lite.bm25_retriever import BM25Retriever
from rag_lite.pipeline import load_documents


def main():
    parser = argparse.ArgumentParser(description="Build dense + BM25 indexes")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_project_dirs(config)

    documents = load_documents(config["data"]["processed_path"])
    print(f"Loaded {len(documents)} documents")

    if not documents:
        raise SystemExit(
            f"No documents found in {config['data']['processed_path']}. "
            "Run a fetch script (e.g. scripts/fetch_eric.py) first."
        )

    print("Building dense index ...")
    dense = DenseRetriever(
        model_name=config["models"]["embedding_model"],
        persist_dir=config["paths"]["dense_index_dir"],
    )
    dense.build_index(documents)
    dense.save()
    print(f"  saved {len(documents)} embeddings to {config['paths']['dense_index_dir']}")

    print("Building BM25 index ...")
    bm25 = BM25Retriever()
    bm25.build_index(documents)
    bm25.save(config["paths"]["bm25_index"])
    print(f"  saved BM25 index to {config['paths']['bm25_index']}")

    print("Done.")


if __name__ == "__main__":
    main()
