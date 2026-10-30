"""Verify transferred immutable data and inventory the user's original datasets."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    previous = Path("E:/SIH")
    if ROOT.resolve() != Path("E:/Ayurveda Rag").resolve():
        raise ValueError("unexpected_destination")
    checked, mismatches, linux_cache_links = 0, [], []
    for directory in ("corpus", "data/raw", "data/generations", "data/hf_cache", "data/manifests"):
        for source in (previous / directory).rglob("*"):
            try:
                regular = source.is_file()
            except OSError:
                if directory == "data/hf_cache":
                    linux_cache_links.append(str(source.relative_to(previous)))
                    continue
                raise
            if not regular:
                continue
            relative = source.relative_to(previous)
            target = ROOT / relative
            checked += 1
            if not target.is_file() or hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(target.read_bytes()).digest():
                mismatches.append(str(relative))
    original = []
    for directory in ("section_output", "tkdl_corpus_final"):
        files = [p for p in (ROOT / directory).rglob("*") if p.is_file()]
        original.append({"directory": directory, "files": len(files),
                         "bytes": sum(p.stat().st_size for p in files), "retrieval_admission": "pending_provenance_audit"})
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "source": str(previous),
              "destination": str(ROOT), "pass": not mismatches, "immutable_files_verified": checked,
              "mismatches": mismatches, "original_datasets": original,
              "linux_cache_links_not_readable_from_windows": linux_cache_links,
              "original_pdfs": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in ROOT.glob("*.pdf")],
              "docker_project": "sih", "preserved_database_volume": "sih_pgdata", "source_retained_as_backup": True}
    (ROOT / "reports/workspace_migration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
