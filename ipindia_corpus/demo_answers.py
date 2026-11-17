"""Explicit IP India classroom demo; no release certification is implied."""
from copy import deepcopy
import time
import re

from . import reliable_answers
from .grounded_answers import NOTICE
from .inference import make_provider, provider_binding_valid, provider_identity, read_setting
from .passport.failover import FALLBACK_REASONS, RequestBudget
from .request_policy import classify
from .demo_queries import retrieval_query, VERSION as QUERY_VERSION

DEMO_NOTICE = "Classroom research demo: IP India archives only; not certified legal advice. " + NOTICE


def enabled():
    return read_setting("AYURIP_DEMO_ANSWERS", "0") == "1"


def response(reason, *, status="abstained", generation_id=None):
    return {"status": status, "reason": reason, "answer_text": "", "claims": [],
            "notice": DEMO_NOTICE, "generation_id": generation_id, "mode": "verified_extracts",
            "demo": True, "scope": "ipindia_only", "provider": {}}


def answer(question, search_fn, verify_fn, binding_check, *, primary=None, fallback=None):
    if re.search(r"\b(?:ABS|TKDL|India Code|biological diversity|access[ -]and[ -]benefit[ -]sharing|benefit[ -]sharing)\b", question, re.I):
        return response("unsupported_domain_in_demo")
    policy = classify(question)
    if not policy["allowed"]:
        return response(policy["reason"])
    try:
        setting = read_setting("AYURIP_FALLBACK_PROVIDER", "gemini")
        if setting not in {"gemini", "none"}:
            raise ValueError("unsupported_demo_fallback")
        primary = primary or make_provider()
        if fallback is None and setting == "gemini":
            fallback = make_provider("gemini")
        binding_check()
    except Exception:
        return response("answer_dependency_unavailable", status="unavailable")
    from .passport.service import CountingProvider
    budget = RequestBudget(max_calls=2)
    cached = {}

    def cached_search(query, top_k=5):
        key = (query, top_k)
        if key not in cached:
            cached[key] = search_fn(retrieval_query(query), top_k=top_k)
        return deepcopy(cached[key])

    attempts = []
    used_fallback = False
    provider = primary
    for index, candidate in enumerate([primary] + ([fallback] if fallback else [])):
        provider = candidate
        counted = CountingProvider(candidate, budget=RequestBudget(max_calls=2, deadline=budget.deadline),
                                   deadline=min(budget.deadline, budget.started + 30) if index == 0 and fallback else budget.deadline)
        result = reliable_answers.answer(question, cached_search, verify_fn, provider=counted)
        attempts.append({"provider": provider_identity(candidate)["provider"],
                         "status": result["status"], "reason": result.get("reason"),
                         "calls_attempted": counted.attempted})
        used_fallback = index == 1
        if index == 0 and fallback and result["status"] == "unavailable" and result.get("reason") in FALLBACK_REASONS:
            try:
                binding_check()
                if not provider_binding_valid(primary):
                    raise ValueError("provider_binding_changed")
            except Exception:
                return response("active_binding_changed", status="unavailable")
            if time.monotonic() < budget.deadline:
                continue
        break
    try:
        binding_check()
        if not provider_binding_valid(primary) or not provider_binding_valid(provider):
            raise ValueError("provider_binding_changed")
        if time.monotonic() >= budget.deadline:
            raise ValueError("answer_deadline_exceeded")
    except Exception:
        return response("active_binding_changed_or_timeout", status="unavailable")
    # Never expose raw completions, model-facing prompts or internal file paths.
    public = {key: result[key] for key in ("status", "reason", "answer_text", "claims", "generation_id", "provider", "timings_ms") if key in result}
    public.update(notice=DEMO_NOTICE, mode="verified_extracts", demo=True, scope="ipindia_only",
                  routing={"fallback_used": used_fallback, "attempts": attempts, "query_adapter": QUERY_VERSION})
    return public
