import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_lite.config import load_config, ensure_project_dirs
from rag_lite.pipeline import RagLitePipeline


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-hyde", action="store_true")
    parser.add_argument("--no-justification", action="store_true")
    parser.add_argument("--domain", default=None,
                         help="Restrict search to documents tagged with this domain")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_project_dirs(config)
    pipeline = RagLitePipeline(config)

    response = pipeline.run(
        query=args.query,
        top_k=args.top_k,
        use_hyde=not args.no_hyde,
        use_justification=not args.no_justification,
        domain=args.domain,
    )

    output_path = Path(config["paths"]["outputs_dir"]) / "latest_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(response.model_dump(), f, indent=2)

    print(response.model_dump_json(indent=2))
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
