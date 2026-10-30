"""Persist deterministic connector and static-check evidence."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    commands = [[sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
                [sys.executable, "-m", "ruff", "check", "ipindia_corpus", "tests", "tools", "--output-format", "concise"]]
    results = []
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        results.append({"command": " ".join(command[1:]), "exit_code": result.returncode,
                        "output": (result.stdout + result.stderr)[-3500:]})
    report = {"pass": all(item["exit_code"] == 0 for item in results), "checks": results,
              "checked_at": datetime.now(timezone.utc).isoformat()}
    target = ROOT / "reports/verification/offline.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
