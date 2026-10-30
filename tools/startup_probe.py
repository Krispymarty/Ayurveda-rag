"""Read-only validation/startup measurement in an isolated fresh Python process.

Use the existing image/network/environment via docker compose run --rm --no-deps
-T --entrypoint python api -B -m tools.startup_probe ... . Never starts an HTTP
server, writes an index, downloads models or changes active state.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/next_milestone"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def snapshot_module(name, source):
    path = DEST / "baseline_source" / f"{name}.py"
    expected = json.loads((DEST / "baseline_files.json").read_text(encoding="utf-8"))["files"][f"ipindia_corpus/{name}.py"]
    if sha(path.read_bytes()) != expected:
        raise ValueError("baseline_code_changed")
    spec = importlib.util.spec_from_file_location(f"ipindia_corpus._probe_{name}", path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source if source is not None else path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def loader(variant):
    from ipindia_corpus import indexer
    if variant in {"resolver", "parallel"}:
        workers = 4 if variant == "parallel" else 1
        return lambda root, **kw: indexer.generation(root, raw_workers=workers, **kw)
    source = (DEST / "baseline_source/indexer.py").read_text(encoding="utf-8")
    if variant == "same_bytes":
        old = '''    for file, key in zip(files, ("documents_sha256", "chunks_sha256")):
        if hashlib.sha256((directory / file).read_bytes()).hexdigest() != manifest[key]:
            raise ValueError(f"generation file hash mismatch: {file}")
    documents = [json.loads(line) for line in (directory / files[0]).read_text(encoding="utf-8").splitlines() if line]
    chunks = [json.loads(line) for line in (directory / files[1]).read_text(encoding="utf-8").splitlines() if line]'''
        new = '''    captured = []
    for file, key in zip(files, ("documents_sha256", "chunks_sha256")):
        raw = (directory / file).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest[key]:
            raise ValueError(f"generation file hash mismatch: {file}")
        captured.append([json.loads(line) for line in raw.decode("utf-8").splitlines() if line])
    documents, chunks = captured'''
        if source.count(old) != 1:
            raise ValueError("intermediate_variant_does_not_match_snapshot")
        source = source.replace(old, new)
    baseline = snapshot_module("indexer", source)
    baseline.archive_path = snapshot_module("paths", None).archive_path
    return baseline.generation


def run(mode, variant, gid):
    from ipindia_corpus import search_api, indexer
    load = loader(variant)
    row = {"at": datetime.now(timezone.utc).isoformat(), "mode": mode, "variant": variant,
           "generation_id": gid, "python": platform.python_version(), "startup_deadline_seconds": 60,
           "cache_condition": "fresh_process_existing_OS_and_model_file_cache_not_forced_cold",
           "probe_sha256": sha(Path(__file__).read_bytes()),
           "baseline_code_sha256": {name: sha((DEST / "baseline_source" / name).read_bytes())
                                    for name in ("indexer.py", "paths.py")},
           "model_revision": indexer.model_revision(), "stage_ms": {},
           "code_sha256": {name: sha((ROOT / "ipindia_corpus" / name).read_bytes())
                           for name in ("indexer.py", "paths.py", "search_api.py", "readiness.py", "connectors.py")}}
    before, _ = search_api._binding()
    if before[0] != gid:
        raise ValueError("unexpected_active_binding")
    started = time.monotonic()
    try:
        if mode == "generation":
            loaded, docs, chunks = load(ROOT, generation_id=gid)
        else:
            search_api.generation = load
            snapshot = search_api._initialize(lambda name, seconds: row["stage_ms"].update({name: seconds * 1000}))
            loaded, docs, chunks = snapshot["binding"][0], snapshot["documents"], snapshot["chunks"]
        row.update(documents=len(docs), chunks=len(chunks), loaded_generation=loaded)
        row["binding_unchanged"] = search_api._binding()[0] == before
        row["success"] = row["binding_unchanged"] and loaded == gid
    except Exception as exc:
        row.update(success=False, error_class=type(exc).__name__)
    row["elapsed_ms"] = (time.monotonic() - started) * 1000
    row["deadline_passed"] = mode != "startup" or row["elapsed_ms"] <= 60_000
    row["passed"] = row["success"] and row["deadline_passed"]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generation", "startup"), required=True)
    parser.add_argument("--variant", choices=("baseline", "same_bytes", "resolver", "parallel"), required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(DEST.resolve()) or output.exists():
        parser.error("output must be a new file under data/next_milestone")
    row = run(args.mode, args.variant, args.generation)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(row, stream, indent=2)
        stream.write("\n")
    print(json.dumps(row))
    return 0 if row["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
