"""Local isolated database, preview and checkpointed Milestone 2 experiments."""
import argparse
import json
import os
import time
from urllib.parse import urlsplit, urlunsplit
import uuid

import requests

from ipindia_corpus.llm_experiment import ROOT, canonical, percentile, sha256, timestamp
from ipindia_corpus.paths import immutable_write
from ipindia_corpus.reliability_candidate import Experiment, answer, normalize_query, pack
from tools.benchmark_reliability import raw_score, summarize, system_score
from tools.reliability_v2 import DATA, DEST, GENERATION, cases, coverage, write

STATE = ROOT / DATA / "database.json"
VARIANTS = {"baseline": Experiment(), **{f"depth{k}": Experiment(f"depth{k}", k) for k in (8, 10, 15)},
            "coverage": Experiment("coverage", 5, coverage=True),
            "normalized": Experiment("normalized", 5, normalize=True),
            "multiquery": Experiment("multiquery", 5, expand=True)}


def score_result(case, result):
    score = system_score(case, result)
    if result.get("evaluation_citation_audit", {}).get("passed") is False:
        score = {**score, "passed": False, "failures": score["failures"] + ["CITATION_INTEGRITY_FAILURE"]}
    return score


def audit_citations(result, verifier):
    count = 0
    try:
        for claim in result.get("claims", []):
            for citation in claim.get("citations", []):
                hit = verifier.chunks[citation["chunk_id"]]
                title = hit.get("document_title") or verifier.docs[hit["document_id"]]["title"]
                verifier({**hit, "title": title, "status": "unknown"}, GENERATION)
                if (citation.get("title") != title or citation["quote"] not in hit["text"]
                        or any(citation.get(k) != hit.get(k) for k in ("document_id", "source_url", "raw_sha256", "section", "rule", "page_start", "page_end"))):
                    raise ValueError("citation_integrity_failed")
                count += 1
        return {"passed": True, "verified_citations": count}
    except (ValueError, KeyError, TypeError, OSError) as exc:
        # Preserve the actual private response for inspection, rather than
        # replacing a bad citation with an apparently empty legal abstention.
        return {"passed": False, "verified_citations_before_failure": count, "error_kind": type(exc).__name__}


def session():
    result = requests.Session()
    result.trust_env = False
    return result


def database_dsn(name):
    parts = urlsplit(os.environ["DATABASE_URL"])
    if not name.startswith("ayurip_preview_v2_") or not all(c in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name) or name == parts.path.lstrip("/"):
        raise ValueError("unsafe_experiment_database")
    return urlunsplit((parts.scheme, parts.netloc, "/" + name, parts.query, parts.fragment))


def prepare():
    import psycopg
    from psycopg import sql
    from ipindia_corpus.indexer import SCHEMA
    if STATE.exists():
        state = json.loads(STATE.read_bytes())
        with psycopg.connect(database_dsn(state["database"])) as connection:
            if dict(connection.execute("SELECT key,value FROM app_state").fetchall()) != state["production_state"]:
                raise ValueError("existing_candidate_state_changed")
        return state
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as production:
        before = dict(production.execute("SELECT key,value FROM app_state").fetchall())
        if before.get("active_generation") != GENERATION or before.get("ranking_version") != "v2":
            raise ValueError("unexpected_production_binding")
        name = "ayurip_preview_v2_" + uuid.uuid4().hex
        production.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        # Clone only known index/cache columns. Generated lexical columns rebuild
        # inside the disposable database; no source files or embeddings change.
        tables = {"corpus_documents": "generation_id,document_id,metadata",
                  "corpus_chunks": "generation_id,chunk_id,document_id,content_hash,text,metadata",
                  "embedding_cache": "content_hash,model_revision,embedding",
                  "index_generations": "generation_id,model_revision,documents,chunks,manifest_sha256,activated_at",
                  "app_state": "key,value"}
        with psycopg.connect(database_dsn(name), autocommit=True) as candidate:
            candidate.execute(SCHEMA)
            counts = {}
            for table, columns in tables.items():
                with production.cursor().copy(f"COPY {table} ({columns}) TO STDOUT WITH (FORMAT BINARY)") as source:
                    with candidate.cursor().copy(f"COPY {table} ({columns}) FROM STDIN WITH (FORMAT BINARY)") as destination:
                        for block in source:
                            destination.write(block)
                counts[table] = candidate.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            candidate.execute("ANALYZE")
        after = dict(production.execute("SELECT key,value FROM app_state").fetchall())
        if before != after:
            raise ValueError("production_binding_changed")
        state = {"database": name, "generation_id": GENERATION, "production_state": before,
                 "production_unchanged": True, "counts": counts, "at": timestamp()}
        immutable_write(STATE, canonical(state))
        write(ROOT, DEST + "/isolated_database.json", state)
        return state


def serve_search():
    import uvicorn
    state = json.loads(STATE.read_bytes())
    os.environ["DATABASE_URL"] = database_dsn(state["database"])
    os.environ["AYURIP_RATE_LIMIT_ENABLED"] = "0"
    from ipindia_corpus.search_api import app, _ensure_ready
    from ipindia_corpus.ranking_diagnostics import parameters
    from ipindia_corpus.retrieval import diagnostic_rows, query_parameters
    from ipindia_corpus.indexer import _connect
    from fastapi import Query

    @app.get("/_candidate_ranks", include_in_schema=False)
    def ranks(q: str = Query(min_length=1, max_length=512)):
        _ensure_ready()
        args = parameters(q, GENERATION)
        args.update(query_parameters(q))
        with _connect() as connection:
            rows = diagnostic_rows(connection, args)
        return {"origin": "new_diagnostic_replay", "generation_id": GENERATION,
                "rows": [{"chunk_id": r[0], "lexical_rank": r[3], "vector_rank": r[4],
                          "provision_rank": r[5], "fused_rank": r[6], "score": float(r[7]) if r[7] is not None else None}
                         for r in rows]}

    uvicorn.run(app, host="0.0.0.0", port=8002, proxy_headers=False)


def serve_answer(variant):
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from ipindia_corpus.http_security import configure_security
    from ipindia_corpus.private_preview import ArchiveVerifier
    from ipindia_corpus.search_api import AnswerRequest
    from ipindia_corpus.reliable_answers import SpanProvider
    os.environ["AYURIP_RATE_LIMIT_ENABLED"] = "0"
    experiment = VARIANTS[variant]
    app = FastAPI(docs_url=None, redoc_url=None)
    configure_security(app)
    verifier, provider, client = ArchiveVerifier(ROOT), SpanProvider(), session()
    frozen = freeze(variant)

    def search(q, top_k=5):
        response = client.get("http://127.0.0.1:8002/search", params={"q": q, "top_k": top_k}, timeout=(3, 15))
        response.raise_for_status()
        return response.json()

    @app.get("/health")
    def health():
        try:
            response = client.get("http://127.0.0.1:8002/health", timeout=(2, 3))
            return JSONResponse(status_code=response.status_code, content={"candidate_sha256": frozen["candidate_sha256"], "experiment": experiment.receipt()})
        except requests.RequestException:
            return JSONResponse(status_code=503, content={"reason": "preview_dependency_unavailable"})

    @app.post("/answer")
    def respond(request: AnswerRequest):
        if request.top_k != 5 or request.source is not None or request.domain is not None:
            return JSONResponse(status_code=400, content={"reason": "frozen_experiment_required"})
        if frozen["candidate_sha256"] != fingerprint(variant)["candidate_sha256"]:
            return JSONResponse(status_code=503, content={"reason": "candidate_source_changed"})
        result = answer(request.question, search, verifier, experiment=experiment, provider=provider)
        result["candidate_sha256"] = frozen["candidate_sha256"]
        return JSONResponse(status_code=503 if result["status"] == "unavailable" else 200, content=result)

    uvicorn.run(app, host="127.0.0.1", port=8001, proxy_headers=False)


def fingerprint(variant):
    from tools.benchmark_reliability import binding, FILES
    extra = ["ipindia_corpus/reliability_candidate.py", "tools/reliability_v2.py", "tools/reliability_v2_runtime.py"]
    extra += [p.relative_to(ROOT).as_posix() for p in (ROOT / "ipindia_corpus").glob("*.py")]
    extra += ["tools/reliability_v2_matrix.py", "tests/test_reliability_v2.py"]
    hashes = {name: sha256((ROOT / name).read_bytes()) for name in sorted(set(FILES + extra))}
    payload = {"experiment": VARIANTS[variant].receipt(), "sources": hashes, "baseline_binding": binding(ROOT),
               "review_manifest_sha256": sha256((ROOT / DATA / "sealed_review_v1/manifest.json").read_bytes())}
    return {"candidate_sha256": sha256(canonical(payload)), **payload}


def freeze(variant):
    payload = fingerprint(variant)
    folder = ROOT / DATA / "candidates" / payload["candidate_sha256"]
    immutable_write(folder / "candidate.json", canonical(payload))
    for name in payload["sources"]:
        immutable_write(folder / "sources" / name, (ROOT / name).read_bytes())
    return payload


def depth():
    client = session()
    items = [c for c in cases()["development"] if c["expected"]["policy"] == "quote_targets"]
    fingerprint_value = fingerprint("baseline")
    raw_rows = []
    for c in items:
        response = client.get("http://127.0.0.1:8002/_candidate_ranks", params={"q": c["question"]}, timeout=(3, 30))
        response.raise_for_status()
        ranks = {r["chunk_id"]: r for r in response.json()["rows"]}
        target_ranks = []
        for group in c["expected"]["target_groups"]:
            target_ranks.append([{**ranks.get(cid, {"chunk_id": cid, "fused_rank": None}),
                                  "eligibility": "eligible" if cid in ranks else "ineligible_or_absent"} for cid in group["required_chunk_ids"]])
        for k in (5, 8, 10, 15):
            started = time.monotonic()
            result = client.get("http://127.0.0.1:8002/search", params={"q": c["question"], "top_k": k}, timeout=(3, 20))
            result.raise_for_status()
            elapsed = (time.monotonic() - started) * 1000
            hits = result.json()["hits"]
            groups = c["expected"]["target_groups"]
            found = coverage(groups, hits)
            packed = pack(c["question"], [{**h, "provenance": "verified_immutable_archive"} for h in hits])
            raw_rows.append({"origin": "new_diagnostic_replay", "case_id": c["id"], "depth": k, "ms": elapsed,
                             "target_group_recall": sum(found) / len(found), "all_groups_present": all(found),
                             "recall_at_5": sum(coverage(groups, hits[:5])) / len(groups),
                             "recall_at_10": sum(coverage(groups, hits[:10])) / len(groups) if k >= 10 else None,
                             "packed_group_recall": sum(coverage(groups, packed)) / len(groups),
                             "packed_ids": [h["chunk_id"] for h in packed], "hit_ids": [h["chunk_id"] for h in hits],
                             "required_target_ranks": target_ranks})
    summary = {}
    for k in (5, 8, 10, 15):
        selected = [r for r in raw_rows if r["depth"] == k]
        summary[str(k)] = {"questions": len(selected), "target_group_recall": sum(r["target_group_recall"] for r in selected) / len(selected),
                           "complete_questions": sum(r["all_groups_present"] for r in selected),
                           "packed_group_recall": sum(r["packed_group_recall"] for r in selected) / len(selected),
                           "p95_ms": percentile([r["ms"] for r in selected], .95)}
    best = min((5, 8, 10, 15), key=lambda k: (-summary[str(k)]["target_group_recall"], k))
    payload = {"at": timestamp(), "candidate": fingerprint_value, "summary": summary, "best_depth": best,
               "population": "15 development answerable queries; no inference", "MRR_nDCG": "not computed: graded relevance labels unavailable",
               "context_tokens": "not measured in retrieval-only sweep; real model prompt counts retained in inference observations", "rows": raw_rows}
    write(ROOT, DEST + "/retrieval_depth.json", payload)
    return {"summary": summary, "best_depth": best}


def language_probe():
    # Prespecified paired provision questions; targets are reused, never independent certification.
    from copy import deepcopy
    positives = [c for c in cases()["development"] if c["expected"]["policy"] == "quote_targets"]
    subjects = [("10", "complete specification", "पूरी specification", "poori specification"),
                ("2", "new product or process definition", "नए product या process की परिभाषा", "naye product ya process ki paribhasha"),
                ("66", "public interest revocation", "लोकहित में revocation", "lokhit mein revocation")]
    queries = []
    for section, english, hindi, roman in subjects:
        base = next(c for c in positives if f"section {section}" in normalize_query(c["question"]).casefold())
        for language, question in (("English", f"Quote archived Patents Act section {section} on {english}."),
                                   ("Hindi", f"पेटेंट अधिनियम धारा {section} में {hindi} का archived मूल उद्धरण दें।"),
                                   ("romanized", f"Patents Act dhara {section} mein {roman} ka archived quote do."),
                                   ("mixed", f"Patents Act धारा {section}: {english} का archived extract दें।")):
            queries.append({"id": f"language-{section}-{language}", "question": question, "language": language, "targets": deepcopy(base["expected"]["target_groups"])})
    immutable_write(ROOT / DATA / "language_probe.json", canonical(queries))
    client, rows = session(), []
    for item in queries:
        for variant in ("original", "normalized"):
            q = item["question"] if variant == "original" else normalize_query(item["question"])
            started = time.monotonic()
            response = client.get("http://127.0.0.1:8002/search", params={"q": q, "top_k": 5}, timeout=(3, 20))
            response.raise_for_status()
            hits = response.json()["hits"]
            found = coverage(item["targets"], hits)
            rows.append({**item, "variant": variant, "retrieval_query": q, "hit_ids": [h["chunk_id"] for h in hits],
                         "target_group_recall": sum(found) / len(found), "ms": (time.monotonic() - started) * 1000})
    payload = {"rows": rows, "certification": False, "scope": "12 diagnostic paired queries, reused development targets"}
    write(ROOT, DEST + "/multilingual_retrieval.json", payload)
    return [{k: r[k] for k in ("id", "variant", "target_group_recall")} for r in rows]


def evaluate(variant, mode, split, passes):
    from ipindia_corpus.private_preview import ArchiveVerifier
    from ipindia_corpus.reliable_answers import SpanProvider
    from ipindia_corpus.llm_experiment import prepare_cohort, ResourceSampler, resource_summary
    from ipindia_corpus.locking import file_lock
    from tools.report_llm_results import verified_artifact
    frozen = freeze(variant)
    selected = cases()[split]
    folder = ROOT / DATA / "observations" / frozen["candidate_sha256"] / f"{mode}-{split}-{passes}passes"
    immutable_write(folder / "plan.json", canonical({"candidate": frozen, "mode": mode, "split": split, "passes": passes,
                                                   "cases_sha256": sha256(canonical(selected))}))
    client, provider, verifier = session(), SpanProvider(), ArchiveVerifier(ROOT)
    experiment = VARIANTS[variant]

    def execute(case):
        started = time.monotonic()
        if mode == "http":
            response = client.post("http://127.0.0.1:8001/answer", json={"question": case["question"]}, timeout=(3, 120))
            if response.status_code not in (200, 503):
                raise ValueError("invalid_preview_response")
            result = response.json()
            if result.get("candidate_sha256") != frozen["candidate_sha256"]:
                raise ValueError("preview_binding_mismatch")
        else:
            result = answer(case["question"], lambda *a, **kw: {"generation_id": GENERATION, "hits": case["evidence"]},
                            verifier, experiment=experiment, provider=provider, controlled_evidence=case["evidence"])
        elapsed = (time.monotonic() - started) * 1000
        audit_started = time.monotonic()
        result["evaluation_citation_audit"] = audit_citations(result, verifier)
        result["evaluation_postresponse_citation_audit_ms"] = (time.monotonic() - audit_started) * 1000
        return result, elapsed

    slots = [(n, c) for n in range(1, passes + 1) for c in selected]
    pending = [(n, c) for n, c in slots if not (folder / "intents" / f"{n}-{c['id']}.json").exists()]
    with file_lock(ROOT / "data/llm/citation_selection.lock"):
        if pending:
            managed = verified_artifact(ROOT / "reports/llm/model_comparison.json")["benchmark_managed_models"]
            preparation = prepare_cohort(provider.client, provider.client.receipt(baseline_model()), managed)
            if not preparation["ready"]:
                raise ValueError(preparation["reason"])
            warm_case = next(c for c in selected if c["expected"]["policy"] == "quote_targets")
            warm, elapsed = execute(warm_case)
            immutable_write(folder / "warmups" / (uuid.uuid4().hex + ".json"), canonical({"result": warm, "ms": elapsed, "preparation": preparation}))
            if not warm.get("model_observations"):
                raise ValueError("warmup_model_not_called")
            with ResourceSampler(interval=1) as sampler:
                for number, case in pending:
                    slot = f"{number}-{case['id']}"
                    intent = {"case_id": case["id"], "pass": number, "candidate_sha256": frozen["candidate_sha256"],
                              "case_sha256": sha256(canonical(case)), "at": timestamp()}
                    immutable_write(folder / "intents" / (slot + ".json"), canonical(intent))
                    started = time.monotonic()
                    try:
                        result, elapsed = execute(case)
                    except Exception as exc:
                        result, elapsed = {"status": "unavailable", "reason": "evaluation_request_failed", "claims": [],
                                           "failure_kind": type(exc).__name__, "model_observations": []}, (time.monotonic() - started) * 1000
                    row = {**intent, "result": result, "http_or_pipeline_ms": elapsed, "system_score": score_result(case, result),
                           "model_called": bool(result.get("model_calls_attempted")),
                           "raw_selection_score": raw_score(case, result) if mode == "controlled" else None}
                    raw = canonical(row)
                    immutable_write(folder / "rows" / slot / (sha256(raw) + ".json"), raw)
                    print(json.dumps({"variant": variant, "mode": mode, "split": split, "pass": number, "id": case["id"], "passed": row["system_score"]["passed"]}), flush=True)
            immutable_write(folder / "resources" / (uuid.uuid4().hex + ".json"), canonical(resource_summary(sampler.samples)))
        rows = []
        for number, case in slots:
            slot = f"{number}-{case['id']}"
            intent = json.loads((folder / "intents" / (slot + ".json")).read_bytes())
            if intent["case_sha256"] != sha256(canonical(case)) or intent["candidate_sha256"] != frozen["candidate_sha256"]:
                raise ValueError("intent_binding_changed")
            found = list((folder / "rows" / slot).glob("*.json"))
            if not found:
                result = {"status": "unavailable", "reason": "interrupted_attempt_not_retried", "claims": []}
                row = {**intent, "result": result, "http_or_pipeline_ms": None, "model_called": False, "system_score": system_score(case, result)}
                raw = canonical(row)
                immutable_write(folder / "rows" / slot / (sha256(raw) + ".json"), raw)
                found = list((folder / "rows" / slot).glob("*.json"))
            if len(found) != 1 or sha256(found[0].read_bytes()) != found[0].stem:
                raise ValueError("corrupt_observation")
            row = json.loads(found[0].read_bytes())
            if row["system_score"] != score_result(case, row["result"]):
                raise ValueError("score_changed")
            rows.append(row)
        if frozen != fingerprint(variant):
            raise ValueError("candidate_changed_during_evaluation")
        summary = summarize(rows, selected)
        summary["measurement_scope"] = "HTTP request or controlled pipeline; postresponse evaluator citation audit excluded"
        summary["unique_questions"] = len(selected)
        positives = {c["id"] for c in selected if c["expected"]["policy"] == "quote_targets"}
        for label, subset in (("answerable", [r for r in rows if r["case_id"] in positives]),
                              ("model_invoking", [r for r in rows if r.get("model_called")])):
            values = [r["http_or_pipeline_ms"] for r in subset if r.get("http_or_pipeline_ms") is not None]
            summary[label + "_latency_ms"] = {"p50": percentile(values, .5), "p95": percentile(values, .95)}
        summary["stages_ms"] = {stage: {"p50": percentile(values, .5), "p95": percentile(values, .95)} for stage, values in
                                 ((stage, [r["result"].get("timings_ms", {}).get(stage) for r in rows if stage in r["result"].get("timings_ms", {})])
                                  for stage in sorted({k for r in rows for k in r["result"].get("timings_ms", {})}))}
        prompts = [o["telemetry"].get("prompt_eval_count") for r in rows for o in r["result"].get("model_observations", []) if o.get("telemetry", {}).get("prompt_eval_count") is not None]
        summary["prompt_tokens"] = {"max": max(prompts) if prompts else None, "p95": percentile(prompts, .95)}
        summary["extra_citation_attempts"] = sum("DISALLOWED_CLAIM" in r["system_score"]["failures"] for r in rows)
        summary["inappropriate_abstention_attempts"] = sum("INAPPROPRIATE_ABSTENTION" in r["system_score"]["failures"] for r in rows)
        payload = {"at": timestamp(), "variant": variant, "mode": mode, "split": split, "passes": passes,
                   "candidate": frozen, "summary": summary, "rows": rows,
                   "independent_certification": False, "public_answers_enabled": False}
        write(ROOT, f"{DEST}/{variant}-{mode}-{split}-{passes}passes.json", payload)
        return summary


def baseline_model():
    from ipindia_corpus.reliable_answers import MODEL
    return MODEL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "serve-search", "serve-answer", "depth", "language", "freeze", "evaluate"))
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="baseline")
    parser.add_argument("--mode", choices=("http", "controlled"), default="http")
    parser.add_argument("--split", choices=("development", "historical", "heldout_regression"), default="development")
    parser.add_argument("--passes", type=int, choices=(1, 3), default=1)
    args = parser.parse_args()
    if args.action == "serve-search":
        serve_search()
        return
    if args.action == "serve-answer":
        serve_answer(args.variant)
        return
    actions = {"prepare": prepare, "depth": depth, "language": language_probe, "freeze": lambda: freeze(args.variant),
               "evaluate": lambda: evaluate(args.variant, args.mode, args.split, args.passes)}
    print(json.dumps(actions[args.action](), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
