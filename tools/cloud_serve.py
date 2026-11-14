"""Production entrypoint: verify the immutable release, then start public search."""
import hashlib
import json
import os
from pathlib import Path

from ipindia_corpus.cloud_config import load_deployment_config
from ipindia_corpus.sync_runner import ROOT


def verify_release(root=ROOT, expected=None):
    root = Path(root).resolve()
    expected = expected or os.environ.get("AYURIP_SNAPSHOT_SHA256", "")
    path = root / "release-manifest.json"
    raw = path.read_bytes()
    if len(expected) != 64 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("release_manifest_binding_failed")
    manifest = json.loads(raw)
    if (manifest.get("version") != "ayurip-readonly-release-v1"
            or manifest.get("active_generation") != "20261014T204940358650Z"
            or manifest.get("ranking_version") != "v2"
            or manifest.get("rollback_generation") != "20261014T190100599320Z"
            or manifest.get("rollback_ranker") != "v1"
            or manifest.get("public_ai_enabled") is not False
            or manifest.get("synchronization_enabled") is not False):
        raise ValueError("release_policy_binding_failed")
    for name, item in manifest["files"].items():
        relative = Path(name)
        resolved = (root / relative).resolve()
        if relative.is_absolute() or not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError("release_file_containment_failed")
        digest = hashlib.sha256()
        size = 0
        with resolved.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
        if size != item["bytes"] or digest.hexdigest() != item["sha256"]:
            raise ValueError("release_file_integrity_failed")
    return manifest


def runtime_port():
    from psycopg.conninfo import conninfo_to_dict
    config = load_deployment_config()
    if not config.public_research:
        raise ValueError("public_research_mode_required")
    info = conninfo_to_dict(os.environ.get("DATABASE_URL", ""))
    if not info.get("dbname") or info.get("sslmode") != "verify-full":
        raise ValueError("database_verified_tls_required")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("offline_embedding_cache_required")
    port = int(os.environ.get("PORT", "8000"))
    if not 1 <= port <= 65535:
        raise ValueError("invalid_listening_port")
    return port


def main():
    port = runtime_port()
    verify_release()
    import uvicorn
    uvicorn.run("ipindia_corpus.search_api:app", host="0.0.0.0", port=port,
                proxy_headers=False, access_log=False, workers=1)


if __name__ == "__main__":
    main()
