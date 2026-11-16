"""Native-host fixed-input Ollama experiments, isolated from public activation.

No model downloads, database writes, retrieval or corpus mutation are performed.
Each observation is retained by hash, including timeouts and invalid outputs.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import threading
import time
import uuid

import requests

from .grounded_answers import validate_endpoint
from .llm_scoring import score_case
from .paths import immutable_write

ROOT = Path(__file__).resolve().parents[1]
MAX_RESPONSE_BYTES = 4_000_000


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class ExperimentFailure(Exception):
    """Stable reason only; never expose transport errors or endpoint credentials."""


def validate_model(model):
    if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*(?::[A-Za-z0-9_.-]+)?", model) or any(part in {".", ".."} for part in model.split("/")):
        raise ValueError("exact_local_model_tag_required")
    return model


def _safe_response(response):
    value = json.loads(json.dumps(response))
    message = value.get("message") if isinstance(value, dict) else None
    if isinstance(message, dict) and "thinking" in message:
        thinking = message.pop("thinking")
        message["thinking_characters"] = len(thinking) if isinstance(thinking, str) else None
    return value


class OllamaExperimentClient:
    """Bounded native loopback transport; no redirects, proxy or remote endpoints."""

    def __init__(self, endpoint="http://127.0.0.1:11434", *, timeout=120, session=None, absolute_deadline=None):
        self.endpoint = validate_endpoint(endpoint)
        if "host.docker.internal" in self.endpoint:
            raise ValueError("native_loopback_endpoint_required")
        if type(timeout) is not int or not 1 <= timeout <= 120:
            raise ValueError("experimental_timeout_must_be_1_to_120_seconds")
        if absolute_deadline is not None and (type(absolute_deadline) not in (int, float)
                or not math.isfinite(absolute_deadline) or absolute_deadline < 0):
            raise ValueError("invalid_absolute_deadline")
        self.timeout = timeout
        self.absolute_deadline = absolute_deadline
        self.session = session or requests.Session()
        self.session.trust_env = False

    def request(self, method, path, payload=None, *, streaming=False):
        if (method, path) not in {("GET", "/api/version"), ("GET", "/api/tags"), ("GET", "/api/ps"),
                                  ("POST", "/api/show"), ("POST", "/api/chat"), ("POST", "/api/generate")}:
            raise ValueError("experiment_endpoint_not_allowed")
        started = time.monotonic()
        deadline = started + self.timeout
        if self.absolute_deadline is not None:
            deadline = min(deadline, self.absolute_deadline)
        remaining = deadline - started
        if remaining <= 0:
            raise ExperimentFailure("timeout")
        response = None
        try:
            response = self.session.request(method, self.endpoint + path, json=payload, stream=True,
                                            allow_redirects=False, timeout=(min(10, remaining), remaining))
            if time.monotonic() >= deadline:
                raise ExperimentFailure("timeout")
            if response.status_code != 200:
                raise ExperimentFailure("http_request_failed")
            raw = bytearray()
            content, thinking_chars, final = [], 0, None
            ttft_content_ms, ttft_any_ms = None, None
            if streaming:
                for line in response.iter_lines():
                    if time.monotonic() >= deadline:
                        raise ExperimentFailure("timeout")
                    if not line:
                        continue
                    raw.extend(line + b"\n")
                    if len(raw) > MAX_RESPONSE_BYTES:
                        raise ExperimentFailure("response_too_large")
                    event = json.loads(line)
                    if not isinstance(event, dict) or event.get("error"):
                        raise ExperimentFailure("invalid_model_response")
                    message = event.get("message", {})
                    if not isinstance(message, dict):
                        raise ExperimentFailure("invalid_model_response")
                    token = message.get("content", "")
                    thinking = message.get("thinking", "")
                    if not isinstance(token, str) or not isinstance(thinking, str):
                        raise ExperimentFailure("invalid_model_response")
                    elapsed = (time.monotonic() - started) * 1000
                    if (token or thinking) and ttft_any_ms is None:
                        ttft_any_ms = elapsed
                    if token and ttft_content_ms is None:
                        ttft_content_ms = elapsed
                    content.append(token)
                    thinking_chars += len(thinking)
                    final = event
                if not final:
                    raise ExperimentFailure("empty_model_response")
                parsed = _safe_response(final)
                parsed.setdefault("message", {})["content"] = "".join(content)
                parsed["message"]["thinking_characters"] = thinking_chars
            else:
                for block in response.iter_content(chunk_size=1024):
                    if time.monotonic() >= deadline:
                        raise ExperimentFailure("timeout")
                    raw.extend(block)
                    if len(raw) > MAX_RESPONSE_BYTES:
                        raise ExperimentFailure("response_too_large")
                parsed = _safe_response(json.loads(raw))
            # A final body/EOF can arrive after the last yielded block. Never
            # accept that output, even when its JSON and model IDs are valid.
            if time.monotonic() >= deadline:
                raise ExperimentFailure("timeout")
            if not isinstance(parsed, dict) or parsed.get("error"):
                raise ExperimentFailure("invalid_model_response")
            return {"response": parsed, "response_bytes_sha256": sha256(bytes(raw)),
                    "http_wall_ms": (time.monotonic() - started) * 1000,
                    "ttft_content_ms": ttft_content_ms, "ttft_any_generated_ms": ttft_any_ms,
                    "streaming": streaming}
        except requests.Timeout as exc:
            raise ExperimentFailure("timeout") from exc
        except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
            raise ExperimentFailure("invalid_or_unavailable_response") from exc
        finally:
            if response is not None:
                response.close()

    def tags(self):
        models = self.request("GET", "/api/tags")["response"].get("models")
        if not isinstance(models, list):
            raise ExperimentFailure("invalid_model_inventory")
        return models

    def placement(self):
        value = self.request("GET", "/api/ps")["response"]
        if not isinstance(value.get("models"), list):
            raise ExperimentFailure("invalid_model_placement")
        return value

    def receipt(self, model):
        model = validate_model(model)
        entry = next((row for row in self.tags() if row.get("name") == model), None)
        if entry is None:
            raise ExperimentFailure("exact_model_not_installed")
        digest = entry.get("digest", "")
        if not isinstance(digest, str):
            raise ExperimentFailure("model_digest_invalid")
        digest = digest.removeprefix("sha256:")
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ExperimentFailure("model_digest_invalid")
        show = self.request("POST", "/api/show", {"model": model})
        if "completion" not in show["response"].get("capabilities", []):
            raise ExperimentFailure("chat_capability_unverified")
        return {"model": model, "model_digest": digest, "tag_metadata": entry, "show_response": show["response"],
                "show_response_bytes_sha256": show["response_bytes_sha256"], "captured_at": timestamp(),
                "chat_capability_verified": True, "thinking_capable": "thinking" in show["response"].get("capabilities", [])}

    def verify_identity(self, receipt):
        entry = next((row for row in self.tags() if row.get("name") == receipt["model"]), None)
        if not entry or entry.get("digest", "").removeprefix("sha256:") != receipt["model_digest"]:
            raise ExperimentFailure("model_digest_changed")

    def unload_for_cold_measurement(self, model):
        """Never evict another resident model or attempt any download."""
        model = validate_model(model)
        resident = self.placement()["models"]
        if any(row.get("name", row.get("model")) != model for row in resident):
            return {"cold_measurement_available": False, "reason": "another_model_resident_unload_skipped"}
        if not resident:
            return {"cold_measurement_available": True, "reason": "model_not_resident"}
        self.request("POST", "/api/generate", {"model": model, "keep_alive": 0})
        for _ in range(30):
            if not self.placement()["models"]:
                return {"cold_measurement_available": True, "reason": "exact_model_unloaded"}
            time.sleep(.1)
        return {"cold_measurement_available": False, "reason": "model_unload_not_confirmed"}


def gpu_snapshot():
    command = shutil.which("nvidia-smi")
    if not command:
        return {"available": False, "reason": "nvidia_smi_unavailable"}
    try:
        result = subprocess.run([command, "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu,utilization.memory",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2, check=False)
        if result.returncode:
            return {"available": False, "reason": "nvidia_smi_failed"}
        devices = []
        for line in result.stdout.splitlines():
            values = [part.strip() for part in line.split(",")]
            if len(values) != 7:
                continue
            device = {"index": values[0], "name": values[1], "driver_version": values[2]}
            for name, value in zip(("memory_total_mb", "memory_used_mb", "gpu_utilization_percent", "memory_utilization_percent"), values[3:]):
                try:
                    device[name] = float(value)
                except ValueError:
                    device[name] = None
            devices.append(device)
        return {"available": bool(devices), "devices": devices}
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False, "reason": "nvidia_smi_unavailable"}


def resource_snapshot():
    result = {"captured_at": timestamp(), "gpu": gpu_snapshot()}
    try:
        import psutil
        memory = psutil.virtual_memory()
        processes = []
        for process in psutil.process_iter(["name", "memory_info"]):
            try:
                name = (process.info.get("name") or "").lower()
                if "ollama" in name or "llama_server" in name or "llama-server" in name:
                    info = process.info.get("memory_info")
                    processes.append(info.rss if info else 0)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        result.update(psutil_available=True, cpu_percent=psutil.cpu_percent(interval=None),
                      ram_total_bytes=memory.total, ram_available_bytes=memory.available, ram_used_bytes=memory.used,
                      benchmark_rss_bytes=psutil.Process().memory_info().rss, ollama_process_rss_bytes=sum(processes))
    except (ImportError, OSError):
        result.update(psutil_available=False)
    return result


class ResourceSampler:
    def __init__(self, interval=.5):
        if not .5 <= interval <= 5:
            raise ValueError("resource_sample_interval_out_of_bounds")
        self.interval, self.samples, self.stop_event = interval, [], threading.Event()
        self.thread = None

    def _run(self):
        while not self.stop_event.is_set() and len(self.samples) < 300:
            started = time.monotonic()
            self.samples.append(resource_snapshot())
            self.stop_event.wait(max(.01, self.interval - (time.monotonic() - started)))

    def __enter__(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=3)


def resource_summary(samples):
    def peak(key):
        values = [sample[key] for sample in samples if isinstance(sample.get(key), (float, int))]
        return max(values) if values else None
    devices = [device for sample in samples for device in sample.get("gpu", {}).get("devices", [])]
    memory = [device["memory_used_mb"] for device in devices if device.get("memory_used_mb") is not None]
    utilization = [device["gpu_utilization_percent"] for device in devices if device.get("gpu_utilization_percent") is not None]
    return {"sample_count": len(samples), "requested_sample_interval_ms": 500,
            "peak_cpu_percent": peak("cpu_percent"), "peak_host_ram_used_bytes": peak("ram_used_bytes"),
            "peak_ollama_rss_bytes": peak("ollama_process_rss_bytes"), "peak_benchmark_rss_bytes": peak("benchmark_rss_bytes"),
            "peak_gpu_memory_used_mb": max(memory) if memory else None,
            "mean_gpu_utilization_percent": statistics.mean(utilization) if utilization else None}


def timing_metadata(response):
    result = {}
    for name in ("load_duration", "prompt_eval_duration", "eval_duration", "total_duration"):
        value = response.get(name)
        if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 10 ** 18:
            result[name] = value
            result[name + "_ms"] = value / 1_000_000
    for name in ("prompt_eval_count", "eval_count", "thinking_eval_count", "thinking_count"):
        value = response.get(name)
        if type(value) is int and 0 <= value <= 10 ** 9:
            result[name] = value
    if result.get("eval_duration", 0) > 0 and "eval_count" in result:
        result["tokens_per_second"] = result["eval_count"] / (result["eval_duration"] / 1_000_000_000)
    message = response.get("message", {})
    result["thinking_characters"] = message.get("thinking_characters") if isinstance(message, dict) else None
    result["done"] = response.get("done") is True
    result["done_reason"] = response.get("done_reason") if response.get("done_reason") in ("stop", "length", "load", "unload") else None
    return result


def protected_state(root, generation_id):
    root = Path(root)
    paths = ["config/answer_generation.json", "config/release.json", "config/source_policies.json", "config/curated_sources.json",
             "ipindia_corpus/retrieval.py", "ipindia_corpus/retrieval_legacy.py", "data/model.lock.json",
             "data/generations/ACTIVE", "data/generations/CURRENT", f"data/generations/{generation_id}/manifest.json"]
    return {path: sha256((root / path).read_bytes()) if (root / path).is_file() else None for path in paths}


def code_bindings():
    names = ["ipindia_corpus/grounded_answers.py", "ipindia_corpus/llm_scoring.py",
             "ipindia_corpus/llm_experiment.py", "tools/benchmark_local_llms.py", "tools/freeze_llm_benchmark.py"]
    return {name: sha256((ROOT / name).read_bytes()) for name in names}


def save_artifact(root, relative, value):
    root = Path(root)
    body = canonical(value)
    digest = sha256(body)
    immutable_write(root / "reports/llm/history" / f"{digest}.json", body)
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(body)
    target.with_suffix(target.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def store_receipt(root, receipt):
    body = canonical(receipt)
    path = Path(root) / "data/llm/models" / receipt["model_digest"] / f"{sha256(body)}.json"
    immutable_write(path, body)
    return path.relative_to(root).as_posix()


def _safe_placement(client):
    try:
        return client.placement()
    except ExperimentFailure as exc:
        return {"available": False, "reason": str(exc)}


def cohort_placement(placement, model, *, allow_empty=False):
    """Verify actual resident identity separately from reported GPU allocation."""
    rows = placement.get("models") if isinstance(placement, dict) else None
    names = [row.get("name", row.get("model")) for row in rows] if isinstance(rows, list) and all(isinstance(row, dict) for row in rows) else None
    single = names == [model]
    result = {"single_model_verified": single, "cohort_safe": single or allow_empty and names == [],
              "resident_models": names, "gpu_memory_placement_measured": False}
    if single:
        size, vram = rows[0].get("size"), rows[0].get("size_vram")
        if type(size) is int and size > 0 and type(vram) is int and vram >= 0:
            ratio = vram / size
            result.update(gpu_memory_placement_measured=True, reported_size_bytes=size,
                          reported_size_vram_bytes=vram, reported_vram_ratio=ratio,
                          reported_memory_placement="predominantly_gpu" if ratio >= .95 else "mixed_cpu_gpu" if ratio > 0 else "cpu")
    return result


def prepare_cohort(client, receipt, managed_models):
    """Evict only exact digest-bound experiment tags; refuse unrelated residents."""
    current = receipt["model"]
    owned = {**managed_models, current: receipt["model_digest"]}
    initial = client.placement()
    def is_owned(row):
        if not isinstance(row, dict):
            return False
        name = row.get("name", row.get("model"))
        digest = row.get("digest")
        return name in owned and (digest is None or isinstance(digest, str) and digest.removeprefix("sha256:") == owned[name])
    rows = initial.get("models")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ExperimentFailure("invalid_model_placement")
    for row in rows:
        if not is_owned(row):
            return {"ready": False, "reason": "external_model_resident", "placement_before": initial, "unloaded_owned_models": []}
    unloaded = []
    for row in rows:
        name = row.get("name", row.get("model"))
        # Recheck before each unload: an external process may have loaded a model
        # since the initial check. Never unload it or proceed concurrently.
        current_rows = client.placement().get("models", [])
        if not isinstance(current_rows, list) or any(not is_owned(resident) for resident in current_rows):
            return {"ready": False, "reason": "external_model_resident", "placement_before": initial, "unloaded_owned_models": unloaded}
        client.request("POST", "/api/generate", {"model": name, "keep_alive": 0})
        unloaded.append(name)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        remaining = client.placement()
        if remaining.get("models") == []:
            return {"ready": True, "reason": "exclusive_empty_cohort_confirmed", "placement_before": initial,
                    "placement_after": remaining, "unloaded_owned_models": unloaded}
        remaining_rows = remaining.get("models")
        if not isinstance(remaining_rows, list) or any(not is_owned(row) for row in remaining_rows):
            return {"ready": False, "reason": "external_model_resident", "placement_before": initial, "unloaded_owned_models": unloaded}
        time.sleep(.1)
    return {"ready": False, "reason": "owned_model_unload_not_confirmed", "placement_before": initial, "unloaded_owned_models": unloaded}


def _managed_models(root, baseline):
    """Recover ownership only from the baseline and hash-verified prior run."""
    owned = {baseline["model"]: baseline["model_digest"]} if baseline.get("model") and baseline.get("model_digest") else {}
    previous = Path(root) / "reports/llm/model_comparison.json"
    receipt = previous.with_suffix(".json.sha256")
    if previous.is_file() and receipt.is_file() and sha256(previous.read_bytes()) == receipt.read_text(encoding="ascii").strip():
        recorded = json.loads(previous.read_bytes()).get("benchmark_managed_models", {})
        if isinstance(recorded, dict):
            for name, digest in recorded.items():
                try:
                    validate_model(name)
                except ValueError:
                    continue
                if isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest):
                    owned[name] = digest
    return owned


def observe(client, receipt, case, system, schema, profile, *, root, run_id, phase, pass_number=0, streaming=False,
            enforce_single_model=False, allow_unloaded=False):
    """Exactly one genuine model request, then shared strict scoring; no repairs."""
    from tools.freeze_llm_benchmark import model_input
    started = time.monotonic()
    entry = {"observation_id": uuid.uuid4().hex, "run_id": run_id, "captured_at": timestamp(),
             "model": receipt["model"], "model_digest": receipt["model_digest"], "case_id": case["id"],
             "category": case.get("category"), "review_kind": case.get("review_kind"), "phase": phase,
             "pass": pass_number, "profile": profile, "model_input_sha256": case["model_input_sha256"],
             "evidence_sha256": case["evidence_sha256"], "provider_called": False,
             "backend_guard_credit": False, "ttft_content_ms": None, "ttft_any_generated_ms": None}
    entry["code_sha256"] = code_bindings()
    entry["placement_before"] = _safe_placement(client)
    entry["cohort_before"] = cohort_placement(entry["placement_before"], receipt["model"], allow_empty=allow_unloaded)
    with ResourceSampler() as sampler:
        try:
            if enforce_single_model and not entry["cohort_before"]["cohort_safe"]:
                raise ExperimentFailure("unexpected_model_concurrency")
            identity_started = time.monotonic()
            client.verify_identity(receipt)
            entry["identity_verification_ms"] = (time.monotonic() - identity_started) * 1000
            inputs = model_input(case, system, schema)
            payload = {**inputs, "model": receipt["model"], "stream": streaming,
                       "options": {"temperature": 0, "num_predict": profile["num_predict"], "num_ctx": profile["num_ctx"]},
                       "keep_alive": profile["keep_alive"]}
            if profile.get("think") is False:
                if not receipt.get("thinking_capable"):
                    raise ExperimentFailure("thinking_capability_unverified")
                payload["think"] = False
            entry["provider_called"] = True
            chat_started = time.monotonic()
            try:
                result = client.request("POST", "/api/chat", payload, streaming=streaming)
            finally:
                entry["chat_attempt_wall_ms"] = (time.monotonic() - chat_started) * 1000
            response = result["response"]
            if response.get("model") != receipt["model"]:
                raise ExperimentFailure("model_identity_mismatch")
            raw = response.get("message", {}).get("content")
            reason = response.get("done_reason")
            if response.get("done") is not True or reason not in (None, "stop", "length"):
                reason = "error"
            entry.update(status="completed" if response.get("done") is True and reason in (None, "stop") else "incomplete",
                         response_json=response, response_bytes_sha256=result["response_bytes_sha256"],
                         chat_http_wall_ms=result["http_wall_ms"], ttft_content_ms=result["ttft_content_ms"],
                         ttft_any_generated_ms=result["ttft_any_generated_ms"], streaming=streaming,
                         timings=timing_metadata(response), score=score_case(case, raw, done_reason=reason, provider_called=True))
        except ExperimentFailure as exc:
            reason = str(exc)
            entry.update(status="timeout" if reason == "timeout" else "failed", failure_reason=reason,
                         score=score_case(case, None, done_reason="timeout" if reason == "timeout" else "error", provider_called=True))
        except Exception:
            entry.update(status="failed", failure_reason="experiment_observation_failed",
                         score=score_case(case, None, done_reason="error", provider_called=True))
    entry["placement_after"] = _safe_placement(client)
    entry["cohort_after"] = cohort_placement(entry["placement_after"], receipt["model"])
    if enforce_single_model and not entry["cohort_after"]["cohort_safe"]:
        entry.update(status="failed", failure_reason="unexpected_model_concurrency")
        entry["score"]["passed"] = False
        if "UNEXPECTED_MODEL_CONCURRENCY" not in entry["score"]["failures"]:
            entry["score"]["failures"].append("UNEXPECTED_MODEL_CONCURRENCY")
    entry["observation_wall_ms"] = (time.monotonic() - started) * 1000
    entry["resource_summary"] = resource_summary(sampler.samples)
    entry["resource_samples"] = sampler.samples
    body = canonical(entry)
    path = Path(root) / "reports/llm/observations" / run_id / f"{entry['observation_id']}-{sha256(body)}.json"
    immutable_write(path, body)
    entry["immutable_observation_path"] = path.relative_to(root).as_posix()
    entry["immutable_observation_sha256"] = sha256(body)
    return entry


def percentile(values, probability):
    return sorted(values)[max(0, math.ceil(len(values) * probability) - 1)] if values else None


def summarize(observations):
    complete = [row for row in observations if row["status"] == "completed"]
    latencies = [row["chat_http_wall_ms"] for row in complete]
    tps = [row["timings"]["tokens_per_second"] for row in complete if "tokens_per_second" in row.get("timings", {})]
    errors = Counter(error for row in observations for error in row["score"]["failures"])
    all_latencies = [row.get("chat_attempt_wall_ms", row.get("chat_http_wall_ms", row.get("observation_wall_ms"))) for row in observations]
    all_latencies = [value for value in all_latencies if type(value) in (int, float) and math.isfinite(value)]
    return {"observations": len(observations), "completed_requests": len(complete),
            "timeout_requests": sum(row["status"] == "timeout" for row in observations),
            "request_failure_rate": (len(observations) - len(complete)) / len(observations) if observations else None,
            "latency_scope": "completed_chat_requests_only; timeouts/failures retained separately, never numerical successes",
            "warm_latency_ms": {"p50": percentile(latencies, .5), "median": statistics.median(latencies) if latencies else None,
                                "p90": percentile(latencies, .9), "p95": percentile(latencies, .95)},
            "all_attempt_latency_ms": {"p50": percentile(all_latencies, .5), "median": statistics.median(all_latencies) if all_latencies else None,
                                       "p90": percentile(all_latencies, .9), "p95": percentile(all_latencies, .95)},
            "all_attempt_latency_scope": "generation HTTP wall time including failed/time-limited attempts; pre-generation failures use total observation wall time",
            "attempts_with_latency_measurement": len(all_latencies),
            "median_output_tokens_per_second": statistics.median(tps) if tps else None,
            "raw_model_case_pass_rate": sum(row["score"]["passed"] is True for row in observations) / len(observations) if observations else None,
            "failures": dict(errors), "fabricated_citations": errors["FABRICATED_CITATION"],
            "fabricated_quotes": errors["FABRICATED_QUOTE"],
            "fabrication_assessable_requests": sum(row["score"].get("schema_valid") is True for row in observations),
            "zero_fabricated_citations_gate_passed": bool(observations) and len(complete) == len(observations)
                and all(row["score"].get("source_valid") is True for row in observations)
                and not errors["FABRICATED_CITATION"] and not errors["FABRICATED_QUOTE"],
            "backend_guard_results_separate": True, "backend_guard_success_credited_to_model": False}


def runtime(root=ROOT, *, client=None):
    client = client or OllamaExperimentClient()
    report = {"captured_at": timestamp(), "runtime_location": "native_host_loopback", "docker_gpu_required": False,
              "hardware": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "processor": platform.processor()},
              "resources": resource_snapshot(), "public_answer_route_enabled": False, "model_downloads_performed": False}
    for name, path in (("ollama_version", "/api/version"), ("installed_models", "/api/tags"), ("resident_models", "/api/ps")):
        try:
            report[name] = client.request("GET", path)["response"]
        except ExperimentFailure as exc:
            report[name] = {"available": False, "reason": str(exc)}
    report["gpu_placement_note"] = "Use actual /api/ps size_vram plus sampled NVIDIA utilization; installed GPU alone does not prove offloading."
    save_artifact(root, "reports/llm/runtime_diagnostics.json", report)
    lines = ["# Native local LLM runtime diagnostics", "", f"Captured: {report['captured_at']}", "",
             "Ollama is accessed on native loopback. This benchmark does not require Docker GPU access.", "",
             report["gpu_placement_note"], "", "Runtime inventory, resident-model placement, driver/VRAM/utilization and CPU/RAM measurements are recorded in `runtime_diagnostics.json`. Missing counters remain explicitly unavailable.", "",
             "No source synchronization, retrieval activation, public-answer activation or model download was performed."]
    (Path(root) / "reports/llm/runtime_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _profiles(config):
    return {"baseline": {"num_predict": config["max_output_tokens"], "num_ctx": config.get("num_ctx", 4096), "keep_alive": config.get("keep_alive", "5m")},
            "candidate": {"num_predict": 200, "num_ctx": 4096, "keep_alive": "30m"}}


def profile(root=ROOT, *, model="ministral-3:3b", warm_requests=3, client=None, include_cold=True, include_stream=True):
    from tools.freeze_llm_benchmark import load_frozen
    root = Path(root)
    if type(warm_requests) is not int or not 1 <= warm_requests <= 5:
        raise ValueError("warm_requests_must_be_1_to_5")
    manifest, cases, system, schema = load_frozen(root)
    before = protected_state(root, manifest["generation_id"])
    code_before = code_bindings()
    config = json.loads((root / "config/answer_generation.json").read_text(encoding="utf-8"))
    if model != config["model"]:
        raise ValueError("initial_profile_must_use_current_exact_model")
    client = client or OllamaExperimentClient()
    receipt = client.receipt(model)
    if receipt["model_digest"] != config["model_digest"]:
        raise ExperimentFailure("baseline_pinned_model_digest_mismatch")
    receipt_path = store_receipt(root, receipt)
    run_id = "profile-" + uuid.uuid4().hex
    selected = next(case for case in cases if case["expected"]["policy"] == "quote_targets")
    observations, cold_checks = [], {}
    profiles = _profiles(config)
    for name, parameters in profiles.items():
        immutable_write(root / "data/llm/candidate_profiles" / f"{sha256(canonical(parameters))}.json", canonical(parameters))
        cold = client.unload_for_cold_measurement(model) if include_cold else {"cold_measurement_available": False, "reason": "operator_skipped"}
        cold_checks[name] = cold
        warmed = False
        if cold["cold_measurement_available"]:
            row = observe(client, receipt, selected, system, schema, parameters, root=root, run_id=run_id, phase=name + "_cold",
                          enforce_single_model=True, allow_unloaded=True)
            observations.append(row)
            warmed = row["status"] == "completed"
        if not warmed:
            row = observe(client, receipt, selected, system, schema, parameters, root=root, run_id=run_id,
                          phase=name + "_untimed_warmup", enforce_single_model=True, allow_unloaded=True)
            observations.append(row)
            warmed = row["status"] == "completed"
        if not warmed:
            cold_checks[name]["timed_warm_skipped"] = "warmup_not_completed_or_exclusive"
            continue
        for number in range(warm_requests):
            row = observe(client, receipt, selected, system, schema, parameters, root=root, run_id=run_id, phase=name + "_warm",
                          pass_number=number + 1, enforce_single_model=True)
            observations.append(row)
            if row.get("failure_reason") == "unexpected_model_concurrency":
                cold_checks[name]["timed_warm_aborted"] = "unexpected_model_concurrency"
                break
    if include_stream:
        observations.append(observe(client, receipt, selected, system, schema, profiles["candidate"], root=root, run_id=run_id,
                                    phase="candidate_streaming_ttft", streaming=True, enforce_single_model=True))
    after = protected_state(root, manifest["generation_id"])
    code_after = code_bindings()
    report = {"captured_at": timestamp(), "run_id": run_id, "profiling_completed": True, "model": model, "model_digest": receipt["model_digest"],
              "model_receipt_path": receipt_path, "benchmark_id": manifest["id"], "benchmark_files": manifest["files"],
              "generation_id": manifest["generation_id"], "selected_case_id": selected["id"], "profiles": profiles,
              "experimental_request_timeout_seconds": client.timeout, "cold_checks": cold_checks,
              "summaries": {name: summarize([row for row in observations if row["phase"] == name + "_warm"]) for name in profiles},
              "observations": observations, "protected_state_before": before, "protected_state_after": after,
              "protected_state_unchanged": before == after and code_before == code_after, "public_answer_route_enabled": False,
              "code_sha256_before": code_before, "code_sha256_after": code_after, "code_unchanged": code_before == code_after,
              "production_model_config_changed": before["config/answer_generation.json"] != after["config/answer_generation.json"], "model_downloads_performed": False,
              "embedding_initialization_ms": 0, "retrieval_ms": 0,
              "measurement_scope": "Fixed frozen inputs; no embedding, retrieval, database or public API invocation.",
              "status": "completed" if before == after and code_before == code_after else "failed_protected_state_changed"}
    save_artifact(root, "reports/llm/baseline_performance.json", report)
    analysis = ["# Baseline local model performance", "", f"Model: `{model}`; digest `{receipt['model_digest']}`.", "",
                "Frozen evidence and prompts are reused unchanged. Native host counters and actual Ollama timings separate loading, prompt processing and generation. Retrieval and embedding initialization are excluded (not invoked).", "",
                "| Profile | Completed warm requests | Timeouts | Median ms (completed only) | p95 ms (completed only) |", "|---|---:|---:|---:|---:|"]
    for name, summary in report["summaries"].items():
        analysis.append(f"| {name} | {summary['completed_requests']} | {summary['timeout_requests']} | {summary['warm_latency_ms']['median']} | {summary['warm_latency_ms']['p95']} |")
    analysis += ["", "These few repeated inputs characterize runtime behavior, not full-suite model quality. A timeout or truncation remains a failed observation. Nonstreaming TTFT is null; the separate streaming probe buffers all text and scores it before retaining a result. Raw thinking text is excluded.", "",
                 "GPU offloading conclusions require the retained /api/ps placement and sampled utilization. No model is selected or activated by this report. Source certification and the public answer gate are unchanged."]
    (root / "reports/llm/baseline_analysis.md").write_text("\n".join(analysis) + "\n", encoding="utf-8")
    return report


def compare(root=ROOT, *, models, passes=3, client=None, disable_supported_thinking=False):
    from tools.freeze_llm_benchmark import load_frozen
    root = Path(root)
    if type(passes) is not int or not 1 <= passes <= 3 or not models or len(models) != len(set(models)):
        raise ValueError("bounded_unique_model_comparison_required")
    baseline = json.loads((root / "reports/llm/baseline_performance.json").read_text(encoding="utf-8"))
    if baseline.get("profiling_completed") is not True or not baseline.get("protected_state_unchanged"):
        raise ValueError("complete_baseline_profile_required_before_comparison")
    manifest, cases, system, schema = load_frozen(root)
    before = protected_state(root, manifest["generation_id"])
    code_before = code_bindings()
    client = client or OllamaExperimentClient()
    run_id = "compare-" + uuid.uuid4().hex
    results = []
    parameters = {"num_predict": 200, "num_ctx": 4096, "keep_alive": "30m"}
    managed_models = _managed_models(root, baseline)
    for model in models:
        validate_model(model)
        try:
            receipt = client.receipt(model)
        except ExperimentFailure as exc:
            results.append({"model": model, "status": "not_tested", "reason": str(exc), "observations": []})
            continue
        receipt_path = store_receipt(root, receipt)
        preparation = prepare_cohort(client, receipt, managed_models)
        if not preparation["ready"]:
            results.append({"model": model, "status": "not_tested", "reason": preparation["reason"],
                            "cohort_preparation": preparation, "observations": []})
            continue
        managed_models[model] = receipt["model_digest"]
        profile_parameters = dict(parameters)
        if disable_supported_thinking and receipt["thinking_capable"]:
            profile_parameters["think"] = False
        immutable_write(root / "data/llm/candidate_profiles" / f"{sha256(canonical(profile_parameters))}.json", canonical(profile_parameters))
        rows = []
        # One explicit warmup is retained separately and never counted as a pass.
        warmup = observe(client, receipt, cases[0], system, schema, profile_parameters, root=root, run_id=run_id,
                         phase="comparison_warmup", enforce_single_model=True, allow_unloaded=True)
        warmup_placement = cohort_placement(client.placement(), model)
        if warmup["status"] != "completed" or not warmup_placement["single_model_verified"]:
            results.append({"model": model, "model_digest": receipt["model_digest"], "status": "aborted",
                            "reason": "unexpected_model_concurrency" if not warmup_placement["single_model_verified"] else "model_warmup_failed",
                            "cohort_preparation": preparation, "warmup_placement": warmup_placement, "warmup": warmup,
                            "observations": [], "summary": summarize([])})
            continue
        aborted = False
        for number in range(1, passes + 1):
            for case in cases:
                row = observe(client, receipt, case, system, schema, profile_parameters,
                              root=root, run_id=run_id, phase="comparison_warm", pass_number=number, enforce_single_model=True)
                rows.append(row)
                if row.get("failure_reason") == "unexpected_model_concurrency":
                    aborted = True
                    break
            if aborted:
                break
        results.append({"model": model, "model_digest": receipt["model_digest"], "status": "aborted" if aborted else "measured",
                        "model_receipt_path": receipt_path, "profile": profile_parameters, "warmup": warmup,
                        "cohort_preparation": preparation, "warmup_placement": warmup_placement,
                        "summary": summarize(rows), "category_summaries": {category: summarize([row for row in rows if row["category"] == category]) for category in sorted({row["category"] for row in rows})},
                        "observations": rows})
    after = protected_state(root, manifest["generation_id"])
    code_after = code_bindings()
    report = {"captured_at": timestamp(), "run_id": run_id, "benchmark_id": manifest["id"], "benchmark_files": manifest["files"],
              "generation_id": manifest["generation_id"], "cases_per_pass": len(cases), "passes": passes,
              "models": results, "protected_state_before": before, "protected_state_after": after,
              "protected_state_unchanged": before == after and code_before == code_after, "public_answer_route_enabled": False,
              "code_sha256_before": code_before, "code_sha256_after": code_after, "code_unchanged": code_before == code_after,
              "winner": None, "activation_performed": False, "model_downloads_performed": False,
              "status": "completed" if before == after and code_before == code_after else "failed_protected_state_changed",
              "selection_policy": "No forced winner. Require adequate target coverage, zero fabricated citations/quotes, safety and latency evidence before any separate operator decision.",
              "raw_model_and_backend_guards_separate": True}
    report["benchmark_managed_models"] = managed_models
    save_artifact(root, "reports/llm/model_comparison.json", report)
    return report
