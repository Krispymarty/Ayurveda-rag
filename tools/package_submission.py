"""Package sanitized source and separately verified public corpus state, no weights."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

from ipindia_corpus.sync_runner import ROOT
from tools.cloud_serve import verify_release
from tools.cloud_snapshot import load_snapshot


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def package(root=ROOT):
    root = Path(root).resolve()
    base = root / "data/cloud_release_20261114"
    context = base / "runtime-v6"
    amendment = json.loads((base / "final_amendment.json").read_bytes())
    release = verify_release(context, amendment["manifest_sha256"])
    load_snapshot(base / "database", context)
    submission = root / "submission/AyurIP-v1"
    required = ("README.md", "DEPLOYMENT.md", "ARCHITECTURE.md", "TESTING.md", "DEMO.md", "PROJECT_SUMMARY.md", "FINAL_STATUS.md")
    if any(not (submission / name).is_file() for name in required):
        raise ValueError("submission_documents_incomplete")
    secret_values = []
    from psycopg.conninfo import conninfo_to_dict
    for file in (root / ".env", base / "local-ops/db.env", base / "local-ops/app.env"):
        for line in file.read_text(encoding="utf-8-sig").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            if key == "DATABASE_URL":
                value = conninfo_to_dict(value).get("password", "")
            elif not any(word in key.upper() for word in ("PASSWORD", "TOKEN", "SECRET", "API_KEY")):
                continue
            if len(value) >= 8:
                secret_values.append(value.encode())

    def safe(raw, name):
        if any(value in raw for value in secret_values):
            raise ValueError("actual_secret_in_submission:" + name)
        if re.search(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]{80,}?-----END", raw):
            raise ValueError("private_key_in_submission:" + name)

    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    roots = ("ipindia_corpus/", "web/", "tests/", "tools/", "config/", "deploy/")
    singles = {"README.md", ".gitignore", ".dockerignore", ".env.example", "Dockerfile", "compose.yaml", "requirements.txt", "requirements-service.txt", "data/model.lock.json", "data/reranker.lock.json"}
    source_files = []
    for name in tracked:
        if not name or not (name.startswith(roots) or name in singles):
            continue
        if Path(name).suffix.lower() in {".pdf", ".zip", ".exe", ".pem", ".key", ".gguf", ".safetensors", ".bin", ".sqlite"} or name.endswith("source_policies.json"):
            raise ValueError("forbidden_source_payload:" + name)
        source_files.append(name)
    for path in sorted((root / "data/benchmarks/ipindia-reviewed-20261015-v1").iterdir()):
        if path.is_file():
            source_files.append(path.relative_to(root).as_posix())
    for name in source_files:
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("source_path_escape")
        raw = path.read_bytes()
        safe(raw, name)
        target = submission / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    for path in (root / "deploy").iterdir():
        if path.is_file():
            safe(path.read_bytes(), path.name)
            (submission / "deploy").mkdir(exist_ok=True)
            shutil.copyfile(path, submission / "deploy" / path.name)
    evidence_names = ("offline_tests.json", "postgres_tests.json", "http_smoke.json", "startup_result.json", "restore_result.json", "readonly_role_result.json",
                      "restart_rollback.json", "runtime_inventory.json", "final_amendment.json", "final_checks.json", "final_container_result.json")
    (submission / "evidence").mkdir(exist_ok=True)
    for name in evidence_names:
        path = base / name
        safe(path.read_bytes(), name)
        shutil.copyfile(path, submission / "evidence" / name)
    browser = json.loads((base / "browser_final_pointer.json").read_bytes())
    for name in ("results.json", "local-desktop.png", "local-mobile.png"):
        source = Path(browser["directory"]) / name
        (submission / "screenshots").mkdir(exist_ok=True)
        shutil.copyfile(source, submission / "screenshots" / name)
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    payload = {p.relative_to(submission).as_posix(): digest(p.read_bytes()) for p in sorted(submission.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"}
    manifest = {"version": "AyurIP-v1-submission", "code_commit": code_commit, "cloud_deployed": False,
                "weights_included": False, "secrets_included": False, "payload": payload}
    (submission / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def archive(path, entries):
        hashes = {}
        with zipfile.ZipFile(path, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as output:
            for name, original in entries:
                raw = original.read_bytes()
                safe(raw, name)
                hashes[name] = digest(raw)
                output.writestr(name, raw)
            output.writestr("ARCHIVE_MANIFEST.json", json.dumps({"sha256": hashes}, sort_keys=True, indent=2))
        with zipfile.ZipFile(path) as check:
            if check.testzip() is not None:
                raise ValueError("zip_crc_failure")
            for name, expected in hashes.items():
                if digest(check.read(name)) != expected:
                    raise ValueError("zip_payload_hash_failure")
            if any(name.endswith((".gguf", ".safetensors", ".key", ".pem")) or "/.git/" in name for name in hashes):
                raise ValueError("forbidden_archive_member")
        return {"file": str(path), "sha256": digest(path.read_bytes()), "bytes": path.stat().st_size, "verified_members": len(hashes)}

    main = archive(root / "submission/AyurIP-v1.zip", [("AyurIP-v1/" + p.relative_to(submission).as_posix(), p) for p in sorted(submission.rglob("*")) if p.is_file()])
    state_entries = []
    for name in release["files"]:
        if name.startswith("data/raw/") or name.startswith("data/generations/") or name == "data/model.lock.json":
            state_entries.append((name, context / name))
    state_entries += [("database/" + p.name, p) for p in sorted((base / "database").iterdir()) if p.is_file()]
    state = archive(root / "submission/AyurIP-v1-state.zip", state_entries)
    result = {"source_submission": main, "verified_state": state, "code_commit": code_commit,
              "cloud_deployed": False, "model_weights_in_archives": False, "private_decisions_included": False}
    (base / "package_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for item in (main, state):
        Path(item["file"] + ".sha256").write_text(item["sha256"] + "\n", encoding="ascii")
    print(json.dumps(result))


if __name__ == "__main__":
    package()
