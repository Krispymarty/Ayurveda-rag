"""CLI for frozen, passage-level search evaluation."""
import argparse
import json
from .passage_evaluation import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://api:8000")
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    report = evaluate(api_url=args.api_url, partial=args.partial)
    print(json.dumps({key: value for key, value in report.items() if key not in ("results", "misses", "source_filtered_diagnostics")}, indent=2))
    raise SystemExit(0 if report["pass"] or args.partial else 2)
