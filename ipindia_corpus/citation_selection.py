"""Candidate-only chunk-ID grammar constraints; no production configuration changes."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re

from .corpus_certification import official_public_url
from .grounded_answers import SCHEMA

VARIANT = "citation-chunk-id-enum-v1"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def eligible_chunk_ids(evidence):
    """All verified-metadata input IDs, including decoys; never consult targets.

    This pure function checks metadata structure, not archived bytes. The caller
    must verify the frozen generation and source hashes before a model run.
    """
    if not isinstance(evidence, list):
        raise ValueError("evidence_list_required")
    identifiers, seen = [], set()
    for hit in evidence:
        if not isinstance(hit, dict) or not isinstance(hit.get("chunk_id"), str) or not hit["chunk_id"].strip():
            raise ValueError("nonempty_evidence_chunk_id_required")
        identifier = hit["chunk_id"]
        if identifier in seen:
            raise ValueError("ambiguous_evidence_chunk_id")
        seen.add(identifier)
        if (not isinstance(hit.get("document_id"), str) or not hit["document_id"].strip()
                or not isinstance(hit.get("text"), str) or not hit["text"].strip()
                or hit.get("evidence_role") == "institutional_background"
                or hit.get("provenance") != "verified_immutable_archive"
                or not isinstance(hit.get("source"), str)
                or not isinstance(hit.get("source_url"), str)
                or not isinstance(hit.get("raw_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", hit["raw_sha256"])):
            continue
        if hit.get("source") not in {"ipindia", "indiacode", "tkdl"} or not official_public_url(hit["source"], hit.get("source_url", "")):
            continue
        identifiers.append(identifier)
    return identifiers


def citation_enum_schema(evidence):
    """Deep-clone the existing quote schema, constraining only legal chunk IDs."""
    schema = deepcopy(SCHEMA)
    identifiers = eligible_chunk_ids(evidence)
    if identifiers:
        schema["properties"]["claims"]["items"]["properties"]["chunk_id"]["enum"] = identifiers
    else:
        # JSON Schema enum must be nonempty; an empty claims array is the only
        # meaningful output when prompt data has no eligible official citation.
        schema["properties"]["claims"]["maxItems"] = 0
    return schema


def derive_case(case, system, base_schema):
    """Keep question/evidence/order unchanged and bind the actual derived input."""
    from tools.freeze_llm_benchmark import model_input
    if canonical(base_schema) != canonical(SCHEMA):
        raise ValueError("frozen_base_schema_not_current_quote_schema")
    original = hashlib.sha256(canonical(model_input(case, system, base_schema))).hexdigest()
    if original != case.get("model_input_sha256"):
        raise ValueError("original_frozen_model_input_hash_mismatch")
    if hashlib.sha256(canonical(case["evidence"])).hexdigest() != case.get("evidence_sha256"):
        raise ValueError("original_frozen_evidence_hash_mismatch")
    schema = citation_enum_schema(case["evidence"])
    candidate = deepcopy(case)
    candidate["original_frozen_model_input_sha256"] = original
    candidate["candidate_variant"] = VARIANT
    candidate["derived_schema_sha256"] = hashlib.sha256(canonical(schema)).hexdigest()
    candidate["model_input_sha256"] = hashlib.sha256(canonical(model_input(candidate, system, schema))).hexdigest()
    binding = {"case_id": case["id"], "variant": VARIANT,
               "original_frozen_model_input_sha256": original,
               "derived_model_input_sha256": candidate["model_input_sha256"],
               "derived_schema_sha256": candidate["derived_schema_sha256"],
               "evidence_sha256": case["evidence_sha256"], "eligible_chunk_ids": eligible_chunk_ids(case["evidence"]),
               "forced_empty_selection": schema["properties"]["claims"]["maxItems"] == 0}
    return candidate, schema, binding
