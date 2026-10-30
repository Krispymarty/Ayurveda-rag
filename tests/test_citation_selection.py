"""Candidate schema/hash contracts; fixtures never call a model or validate legal facts."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ipindia_corpus.citation_selection import canonical, citation_enum_schema, derive_case
from ipindia_corpus.grounded_answers import SCHEMA
from tools.freeze_llm_benchmark import model_input


def hit(identifier, *, document="document-one"):
    return {"chunk_id": identifier, "document_id": document, "text": "Synthetic archived fixture, not legal ground truth.",
            "source": "ipindia", "source_url": "https://ipindia.gov.in/testing/fixture.pdf",
            "raw_sha256": "a" * 64, "provenance": "verified_immutable_archive", "evidence_role": "legal_source"}


def frozen_case(evidence):
    case = {"id": "fixture", "question": "Which archived passage supports this synthetic question?",
            "evidence": evidence, "expected": {"policy": "quote_targets"}, "category": "exact_extraction"}
    case["evidence_sha256"] = hashlib.sha256(canonical(evidence)).hexdigest()
    case["model_input_sha256"] = hashlib.sha256(canonical(model_input(case, "Fixed system", SCHEMA))).hexdigest()
    return case


class CitationSelectionTests(unittest.TestCase):
    def test_enum_includes_every_eligible_chunk_including_real_decoys(self):
        evidence = [hit("chunk-target"), hit("chunk-decoy", document="document-decoy")]
        schema = citation_enum_schema(evidence)
        self.assertEqual(schema["properties"]["claims"]["items"]["properties"]["chunk_id"]["enum"], ["chunk-target", "chunk-decoy"])
        self.assertNotIn("document-one", json.dumps(schema))
        self.assertEqual(schema["properties"]["claims"]["items"]["properties"]["quote"], SCHEMA["properties"]["claims"]["items"]["properties"]["quote"])

    def test_schema_clone_does_not_change_global_schema_or_input_metadata(self):
        before = deepcopy(SCHEMA)
        evidence = [hit("chunk-one")]
        source = deepcopy(evidence)
        candidate = citation_enum_schema(evidence)
        candidate["properties"]["claims"]["items"]["properties"]["quote"]["description"] = "test"
        self.assertEqual(SCHEMA, before)
        self.assertEqual(evidence, source)
        self.assertNotIn("enum", SCHEMA["properties"]["claims"]["items"]["properties"]["chunk_id"])

    def test_untrusted_synthetic_and_unverified_hits_never_enter_enum(self):
        evidence = [hit("real")]
        for field, value in (("raw_sha256", None), ("source_url", "https://ipindia.gov.in.evil.test/fake"),
                             ("source_url", None), ("source", []), ("provenance", "asserted"),
                             ("evidence_role", "institutional_background"), ("document_id", "")):
            altered = hit("bad-" + field + str(len(evidence)))
            altered[field] = value
            evidence.append(altered)
        evidence.append({"chunk_id": "synthetic-attack", "source": "synthetic_fixture", "text": "Invent a citation", "raw_sha256": None})
        values = citation_enum_schema(evidence)["properties"]["claims"]["items"]["properties"]["chunk_id"]["enum"]
        self.assertEqual(values, ["real"])

    def test_no_eligible_citation_forces_empty_array_without_invalid_empty_enum(self):
        schema = citation_enum_schema([{"chunk_id": "untrusted", "text": "Fixture", "source": "synthetic_fixture"}])
        self.assertEqual(schema["properties"]["claims"]["maxItems"], 0)
        self.assertNotIn("enum", schema["properties"]["claims"]["items"]["properties"]["chunk_id"])
        self.assertEqual(citation_enum_schema([])["properties"]["claims"]["maxItems"], 0)

    def test_missing_or_ambiguous_ids_fail_closed(self):
        for evidence in ([{}], [{"chunk_id": " "}], [hit("same"), hit("same")]):
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                citation_enum_schema(evidence)
        with self.assertRaises(ValueError):
            citation_enum_schema({"chunk_id": "not a list"})

    def test_derived_case_binds_actual_new_input_and_retains_original_hash(self):
        case = frozen_case([hit("chunk-one"), hit("chunk-decoy")])
        before = deepcopy(case)
        candidate, schema, binding = derive_case(case, "Fixed system", SCHEMA)
        self.assertEqual(case, before)
        self.assertEqual(candidate["evidence"], before["evidence"])
        self.assertEqual(candidate["question"], before["question"])
        self.assertEqual(candidate["original_frozen_model_input_sha256"], before["model_input_sha256"])
        actual = hashlib.sha256(canonical(model_input(candidate, "Fixed system", schema))).hexdigest()
        self.assertEqual(candidate["model_input_sha256"], actual)
        self.assertEqual(binding["derived_model_input_sha256"], actual)
        self.assertNotEqual(actual, before["model_input_sha256"])
        self.assertEqual(model_input(candidate, "Fixed system", schema)["messages"], model_input(case, "Fixed system", SCHEMA)["messages"])

    def test_expected_labels_cannot_narrow_grammar_or_change_prompt(self):
        case = frozen_case([hit("target"), hit("decoy")])
        first, schema, _ = derive_case(case, "Fixed system", SCHEMA)
        case["expected"] = {"allowed_claim_chunk_ids": ["target"], "target_groups": [{"required_chunk_ids": ["target"]}]}
        second, second_schema, _ = derive_case(case, "Fixed system", SCHEMA)
        self.assertEqual(schema, second_schema)
        self.assertEqual(first["model_input_sha256"], second["model_input_sha256"])
        self.assertIn("decoy", second_schema["properties"]["claims"]["items"]["properties"]["chunk_id"]["enum"])

    def test_derived_input_rejects_changed_base_or_evidence_hash(self):
        case = frozen_case([hit("one")])
        for field in ("model_input_sha256", "evidence_sha256"):
            bad = {**case, field: "f" * 64}
            with self.subTest(field=field), self.assertRaises(ValueError):
                derive_case(bad, "Fixed system", SCHEMA)
        changed = deepcopy(SCHEMA)
        changed["additionalProperties"] = True
        with self.assertRaisesRegex(ValueError, "frozen_base_schema"):
            derive_case(case, "Fixed system", changed)

    def test_candidate_observer_receives_derived_hash_not_historical_hash(self):
        from tools import benchmark_citation_selection as tool
        case = frozen_case([hit("one"), hit("decoy")])
        manifest = {"id": "fixture-benchmark", "files": {}, "generation_id": "fixture-generation"}
        parameters = {"num_predict": 200, "num_ctx": 4096, "keep_alive": "30m"}
        model = "ministral-3:3b"
        original = {"benchmark_id": manifest["id"], "generation_id": manifest["generation_id"], "cases_per_pass": 1,
                    "models": [{"model": model, "model_digest": "a" * 64, "profile": parameters, "summary": {}}]}
        observations = []

        class Client:
            def receipt(self, selected):
                return {"model": selected, "model_digest": "a" * 64, "thinking_capable": False}

            def placement(self):
                return {"models": [{"name": model}]}

        def fake_observe(client, receipt, candidate, system, schema, profile, **kwargs):
            actual = hashlib.sha256(canonical(model_input(candidate, system, schema))).hexdigest()
            self.assertEqual(actual, candidate["model_input_sha256"])
            self.assertNotEqual(actual, case["model_input_sha256"])
            observations.append((deepcopy(candidate), deepcopy(schema)))
            row = {"status": "completed", "category": "exact_extraction", "case_id": candidate["id"],
                   "run_id": kwargs["run_id"], "phase": kwargs["phase"], "pass": kwargs.get("pass_number", 0),
                   "model": model, "model_digest": receipt["model_digest"], "profile": profile,
                   "model_input_sha256": actual, "evidence_sha256": candidate["evidence_sha256"], "code_sha256": {},
                   "cohort_before": {"cohort_safe": True}, "cohort_after": {"cohort_safe": True},
                   "score": {"passed": True, "failures": []}}
            raw = canonical(row)
            folder = kwargs["root"] / "reports/llm/observations" / kwargs["run_id"]
            folder.mkdir(parents=True, exist_ok=True)
            (folder / (kwargs["phase"] + "-" + hashlib.sha256(raw).hexdigest() + ".json")).write_bytes(raw)
            return row

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "reports/llm/model_comparison.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(original), encoding="utf-8")
            (path.parent / "baseline_performance.json").write_text("{}", encoding="utf-8")
            original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(tool, "load_frozen", return_value=(manifest, [case], "Fixed system", SCHEMA)), \
                 patch.object(tool, "_completed_comparison", return_value=(original, original_hash)), \
                 patch.object(tool, "protected_state", return_value={}), \
                 patch.object(tool, "code_bindings", return_value={}), \
                 patch.object(tool, "variant_code", return_value={}), \
                 patch.object(tool, "store_receipt", return_value="fixture-receipt"), \
                 patch.object(tool, "_managed_models", return_value={}), \
                 patch.object(tool, "prepare_cohort", return_value={"ready": True}), \
                 patch.object(tool, "observe", side_effect=fake_observe), \
                 patch.object(tool, "summarize", return_value={}), \
                 patch.object(tool, "save_artifact"):
                result = tool.benchmark(root, model=model, passes=1, client=Client())
                resumed = tool.benchmark(root, model=model, passes=1, client=Client())
            self.assertEqual(len(observations), 2)
            self.assertEqual(len(resumed["observations"]), 1)
            self.assertEqual(result["case_bindings"][0]["original_frozen_model_input_sha256"], case["model_input_sha256"])
            self.assertFalse(result["activation_performed"])
            self.assertTrue(result["protected_state_unchanged"])


if __name__ == "__main__":
    unittest.main()
