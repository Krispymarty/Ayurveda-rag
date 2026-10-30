"""Optional offline cross-encoder experiment; production ranking is unchanged."""
from __future__ import annotations

import hashlib
import json
import math
import re
from functools import lru_cache
from numbers import Real
from pathlib import Path

from .sync_runner import ROOT

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
CONFIG = {"cpu_threads": 4, "max_length": 512, "batch_size": 8,
          "max_candidates": 20, "score": "raw_logit", "ties": "original_candidate_order"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_lock(root=ROOT, *, verify_files=True):
    root = Path(root)
    path = root / "data/reranker.lock.json"
    if not path.is_file():
        raise ValueError("reranker_lock_missing")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("model_id") != MODEL_ID or not re.fullmatch(r"[0-9a-f]{40}", lock.get("revision", "")):
        raise ValueError("reranker_revision_invalid")
    snapshot = root / lock.get("snapshot_path", "")
    cache = (root / "data/hf_cache").resolve()
    if not snapshot.resolve().is_relative_to(cache):
        raise ValueError("reranker_snapshot_outside_cache")
    files = lock.get("files")
    if not isinstance(files, dict) or not {"config.json", "model.safetensors"}.issubset(files):
        raise ValueError("reranker_manifest_missing")
    for name, digest in files.items():
        if Path(name).name != name or name.endswith((".bin", ".pkl", ".pickle", ".pt", ".py")):
            raise ValueError("reranker_unsafe_file")
        path = snapshot / name
        if not path.resolve().is_relative_to(cache) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("reranker_manifest_invalid")
        if verify_files and (not path.is_file() or sha256(path) != digest):
            raise ValueError("reranker_artifact_hash_mismatch")
    return lock, snapshot


def configuration(root=ROOT):
    lock, _ = load_lock(root, verify_files=False)
    values = {"model_id": MODEL_ID, "revision": lock["revision"], **CONFIG,
              "implementation_sha256": sha256(__file__),
              "model_artifact_manifest_sha256": hashlib.sha256(json.dumps(lock["files"], sort_keys=True).encode()).hexdigest()}
    values["fingerprint"] = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    return values


class LocalPredictor:
    def __init__(self, snapshot):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(CONFIG["cpu_threads"])
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False, use_safetensors=True).to("cpu").eval()
        if self.model.config.num_labels != 1:
            raise ValueError("reranker_requires_single_logit")

    def predict(self, pairs):
        scores = []
        with self.torch.inference_mode():
            for offset in range(0, len(pairs), CONFIG["batch_size"]):
                batch = pairs[offset:offset + CONFIG["batch_size"]]
                features = self.tokenizer([q for q, _ in batch], [text for _, text in batch],
                                          padding=True, truncation=True, max_length=CONFIG["max_length"], return_tensors="pt")
                scores.extend(self.model(**features).logits[:, 0].tolist())
        return scores


@lru_cache(maxsize=2)
def _predictor(root, lock_hash):
    del lock_hash  # The cache key binds this model instance to a particular immutable lock.
    _, snapshot = load_lock(root)
    return LocalPredictor(snapshot)


def rerank(query, hits, top_k=5, *, root=ROOT, predictor=None):
    """Return copied hits with rerank_score; original score and citations are retained."""
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= 500:
        raise ValueError("reranker_query_invalid")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= CONFIG["max_candidates"]:
        raise ValueError("reranker_top_k_invalid")
    if not isinstance(hits, list) or len(hits) > CONFIG["max_candidates"]:
        raise ValueError("reranker_candidate_count_invalid")
    config = configuration(root)
    identifiers = set()
    for hit in hits:
        required = ("chunk_id", "document_id", "source_url", "raw_sha256", "text")
        if not isinstance(hit, dict) or any(not isinstance(hit.get(key), str) or not hit[key] for key in required):
            raise ValueError("reranker_candidate_invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", hit["raw_sha256"]):
            raise ValueError("reranker_candidate_hash_invalid")
        if not isinstance(hit.get("score"), Real) or isinstance(hit["score"], bool) or not math.isfinite(hit["score"]):
            raise ValueError("reranker_hybrid_score_invalid")
        identity = (hit["document_id"], hit["chunk_id"])
        if identity in identifiers:
            raise ValueError("reranker_duplicate_candidate")
        identifiers.add(identity)
    metadata = {**config, "candidate_count": len(hits), "adopted": False}
    if not hits:
        return [], metadata
    active = predictor or _predictor(str(Path(root).resolve()), sha256(Path(root) / "data/reranker.lock.json"))
    scores = active.predict([(query, hit["text"]) for hit in hits])
    if not hasattr(scores, "__len__") or len(scores) != len(hits):
        raise ValueError("reranker_output_length_invalid")
    values = []
    for score in scores:
        if not isinstance(score, Real) or isinstance(score, bool) or not math.isfinite(float(score)):
            raise ValueError("reranker_output_invalid")
        values.append(float(score))
    ordering = sorted(range(len(hits)), key=lambda index: (-values[index], index))
    return [{**hits[index], "rerank_score": values[index]} for index in ordering[:top_k]], metadata
