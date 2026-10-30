"""Explicit native local-model experiments; no download or production activation."""
from __future__ import annotations

import argparse
import json

from ipindia_corpus.llm_experiment import OllamaExperimentClient, ROOT, compare, profile, runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=120)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("runtime")
    baseline = commands.add_parser("profile")
    baseline.add_argument("--model", default="ministral-3:3b")
    baseline.add_argument("--warm-requests", type=int, default=3)
    baseline.add_argument("--skip-cold", action="store_true")
    baseline.add_argument("--skip-stream", action="store_true")
    comparison = commands.add_parser("compare")
    comparison.add_argument("--models", nargs="+", required=True)
    comparison.add_argument("--passes", type=int, default=3)
    comparison.add_argument("--disable-supported-thinking", action="store_true")
    args = parser.parse_args()
    try:
        client = OllamaExperimentClient(args.endpoint, timeout=args.timeout)
        if args.command == "runtime":
            result = runtime(ROOT, client=client)
            summary = {"command": "runtime", "report": "reports/llm/runtime_diagnostics.json", "native_host": True,
                       "ollama_available": "version" in result["ollama_version"]}
        elif args.command == "profile":
            result = profile(ROOT, model=args.model, warm_requests=args.warm_requests, client=client,
                             include_cold=not args.skip_cold, include_stream=not args.skip_stream)
            summary = {key: result[key] for key in ("status", "model", "model_digest", "protected_state_unchanged", "summaries")}
        else:
            result = compare(ROOT, models=args.models, passes=args.passes, client=client,
                             disable_supported_thinking=args.disable_supported_thinking)
            summary = {key: result[key] for key in ("status", "cases_per_pass", "passes", "protected_state_unchanged", "winner")}
            summary["models"] = [{key: model[key] for key in ("model", "status", "summary", "reason") if key in model} for model in result["models"]]
        print(json.dumps(summary, indent=2))
    except Exception:
        print(json.dumps({"status": "unavailable", "reason": "local_experiment_prerequisite_or_runtime_unavailable"}))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
