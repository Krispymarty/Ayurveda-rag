"""Download only safe pinned model artifacts for the optional local experiment."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from ipindia_corpus.paths import immutable_write
from ipindia_corpus.reranker import MODEL_ID, load_lock
from ipindia_corpus.sync_runner import ROOT


def prepare(root=ROOT):
    from huggingface_hub import HfApi, snapshot_download
    root = Path(root)
    target = root / "data/reranker.lock.json"
    if target.exists():
        lock, _ = load_lock(root)
        return {"status": "already_prepared", "model_id": MODEL_ID, "revision": lock["revision"]}
    info = HfApi().model_info(MODEL_ID)
    if not info.sha or len(info.sha) != 40:
        raise ValueError("official_model_revision_unavailable")
    available = {file.rfilename for file in info.siblings}
    permitted = {"model.safetensors", "config.json", "tokenizer_config.json", "special_tokens_map.json",
                 "tokenizer.json", "vocab.txt", "README.md", "LICENSE", "LICENSE.txt", "added_tokens.json"}
    selected = sorted(available & permitted)
    if not {"model.safetensors", "config.json", "tokenizer_config.json"}.issubset(selected):
        raise ValueError("safe_model_artifacts_missing")
    cache = root / "data/hf_cache/hub"
    snapshot = Path(snapshot_download(MODEL_ID, revision=info.sha, allow_patterns=selected, cache_dir=str(cache)))
    files = {name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest() for name in selected}
    lock = {"schema_version": 1, "model_id": MODEL_ID, "revision": info.sha,
            "model_card_url": f"https://huggingface.co/{MODEL_ID}", "license": "apache-2.0", "language": "English",
            "prepared_at": datetime.now(timezone.utc).isoformat(), "snapshot_path": snapshot.relative_to(root).as_posix(),
            "files": files, "pickle_allowed": False, "trust_remote_code": False}
    immutable_write(target, json.dumps(lock, indent=2, sort_keys=True).encode())
    load_lock(root)
    return {"status": "prepared", "model_id": MODEL_ID, "revision": info.sha,
            "files": len(files), "bytes": sum((snapshot / name).stat().st_size for name in selected)}


if __name__ == "__main__":
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise SystemExit("Run explicitly with HF_HUB_OFFLINE=0 and TRANSFORMERS_OFFLINE=0 to prepare artifacts.")
    print(json.dumps(prepare(), indent=2))
