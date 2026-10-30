"""Refresh controlled content tags in derived JSONL without touching raw bytes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ipindia_corpus.pipeline import controlled_topics  # noqa: E402


def retag(path: Path, is_document: bool) -> None:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            content = Path(row["normalized_path"]).read_text(encoding="utf-8") if is_document else row["text"]
            row["topics"] = controlled_topics(content)
            rows.append(row)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def main() -> None:
    for path in (ROOT / "documents.jsonl", ROOT / "data" / "documents" / "documents.jsonl"):
        retag(path, True)
    for path in (ROOT / "chunks.jsonl", ROOT / "data" / "chunks" / "chunks.jsonl"):
        retag(path, False)
    print("Controlled content tags refreshed in both derived exports")


if __name__ == "__main__":
    main()
