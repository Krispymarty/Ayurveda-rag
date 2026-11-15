"""Build a portable read-only evidence/model snapshot, never a source admission."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from ipindia_corpus.indexer import generation, model_revision
from ipindia_corpus.paths import ArchiveResolver
from ipindia_corpus.sync_runner import ROOT
from tools.cloud_snapshot import ACTIVE, ROLLBACK


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(output, root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() or output == root or root.is_relative_to(output):
        raise ValueError("new_separate_staging_directory_required")
    output.mkdir(parents=True)
    resolver = ArchiveResolver(root)
    files, counts = {}, {}

    def copy(path, relative=None):
        relative = relative or path.relative_to(root).as_posix()
        destination = output / relative
        if relative in files:
            return
        if not destination.resolve().is_relative_to(output):
            raise ValueError("release_path_escape")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        original_hash = sha(path)
        if sha(destination) != original_hash:
            raise ValueError("release_copy_hash_mismatch")
        files[relative] = {"sha256": original_hash, "bytes": destination.stat().st_size}

    # Explicit application allowlist. Never copy an entire workspace or its .env.
    for folder, extensions in (("ipindia_corpus", {".py"}), ("web", {".html", ".css", ".js", ".svg", ".png", ".ttf"})):
        for path in sorted((root / folder).rglob("*")):
            if path.suffix in extensions and "__pycache__" not in path.parts:
                copy(path)
    for name in ("web/fonts/OFL-DM-Sans.txt", "web/fonts/OFL-Instrument-Serif.txt",
                 "tools/cloud_snapshot.py", "tools/cloud_serve.py",
                 "requirements.txt", "requirements-service.txt", "deploy/runtime-lock.txt", "deploy/embedding-files.lock.json",
                 "data/model.lock.json"):
        copy(root / name)
    originals = set()
    for gid in (ACTIVE, ROLLBACK):
        _, docs, chunks = generation(root, generation_id=gid)
        if any(d["source"] != "ipindia" or d["status"] != "unknown" for d in docs):
            raise ValueError("unapproved_source_in_release")
        counts[gid] = {"documents": len(docs), "chunks": len(chunks)}
        for name in ("manifest.json", "documents.jsonl", "chunks.jsonl"):
            copy(root / "data/generations" / gid / name)
        for doc in docs:
            original = resolver.resolve(doc["raw_path"])
            copy(original)
            originals.add(original.relative_to(root).as_posix())
    # No corpus pointer is copied or changed. Database binding chooses the generation.
    revision = model_revision(root)
    cache = root / "data/hf_cache/hub/models--intfloat--multilingual-e5-small"
    snapshot = cache / "snapshots" / revision
    model_lock = json.loads((root / "deploy/embedding-files.lock.json").read_bytes())
    if model_lock["revision"] != revision or model_lock["model_id"] != "intfloat/multilingual-e5-small":
        raise ValueError("model_file_lock_binding_failed")
    if not snapshot.is_dir():
        raise ValueError("pinned_embedding_snapshot_missing")
    for path in sorted(snapshot.rglob("*")):
        if path.is_file():
            # Dereference Linux HF links into regular files in the new snapshot.
            resolved = path.resolve(strict=True)
            hub = cache.parent.resolve()
            if not resolved.is_relative_to(hub):
                raise ValueError("model_cache_path_escape")
            if sha(resolved) != model_lock["files"].get(path.relative_to(snapshot).as_posix()):
                raise ValueError("model_file_lock_hash_mismatch")
            copy(resolved, path.relative_to(root).as_posix())
    if not (output / snapshot.relative_to(root) / "model.safetensors").is_file():
        raise ValueError("pinned_embedding_weights_missing")
    for gid in (ACTIVE, ROLLBACK):
        generation(output, generation_id=gid)
    manifest = {"version": "ayurip-readonly-release-v1", "active_generation": ACTIVE,
                "ranking_version": "v2", "rollback_generation": ROLLBACK, "rollback_ranker": "v1",
                "model_revision": revision, "generations": counts, "original_files": len(originals),
                "files": files, "public_ai_enabled": False, "synchronization_enabled": False,
                "source_certification": "archived_ipindia_only_status_unknown"}
    raw = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    (output / "release-manifest.json").write_bytes(raw)
    return {"files": len(files), "originals": len(originals), "bytes": sum(x["bytes"] for x in files.values()),
            "manifest_sha256": hashlib.sha256(raw).hexdigest(), "generations": counts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.root)))
