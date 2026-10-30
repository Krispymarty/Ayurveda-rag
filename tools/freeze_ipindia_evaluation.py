"""Freeze additional archive-selected passage labels before retrieval tuning."""
import hashlib
import json
from datetime import datetime, timezone

from ipindia_corpus.indexer import generation
from ipindia_corpus.sync_runner import ROOT


def main():
    target = ROOT / "tests/ipindia_retrieval_expanded.json"
    if target.exists():
        raise ValueError("expanded_labels_already_frozen")
    original = ROOT / "tests/retrieval_queries.json"
    dataset = json.loads(original.read_text(encoding="utf-8"))
    active = (ROOT / "data/generations/ACTIVE").read_text().strip()
    _, documents, chunks = generation(generation_id=active)
    specs = [
        ("Patents Act section 48 exclusive rights of patentees", [("section-48", "exclusive right")]),
        ("Patents Act section 53 term of patent", [("section-53", "twenty years")]),
        ("Patents Act section 64 revocation of patents", [("section-64", "may be revoked")]),
        ("Patents Act section 57 amendment of specification", [("section-57", "amend")]),
        ("Patents Act section 39 filing outside India resident", [("section-39", "resident in India")]),
        ("Patents Act section 60 restoration of lapsed patent", [("section-60", "ceased to have effect")]),
        ("Patents Act section 66 revocation public interest", [("section-66", "mischievous")]),
        ("Patents Act section 72 register inspection", [("section-72", "open to inspection")]),
        ("Patents Act exclusive rights of a patentee", [("section-48", "exclusive right")]),
        ("Patents Act duration and term of patent", [("section-53", "twenty years")]),
        ("Patent Rules rule 80 renewal fees", [("rule-80", "expiration of the second year")]),
        ("Patent Rules rule 71 permission to apply outside India", [("rule-71", "Form 25")]),
        ("Patent Rules publication of application eighteen months", [("rule-24", "eighteen months")]),
        ("Patent Rules rule 90 registration of title and interest", [("rule-90", "Form 16")]),
        ("Patent Rules rule 16 models and samples", [("rule-16", "Models or samples")]),
        ("Patents Rules 2003 drawngs rule 15", [("rule-15", "Drawings")]),
        ("Tradmark Rules 2017 rule 23 application form", [("1_69_1_Trade_Marks_Rules_2017.pdf", "TM-A")]),
        ("Designs Act section 5 registration prescribed fee", [("1_58_1_design_act_1_.pdf", "prescribed fee")]),
        ("Patents Act sections 48 and 53 rights and term", [("section-48", "exclusive right"), ("section-53", "twenty years")]),
        ("Patent Rules rules 15 and 16 drawings and models", [("rule-15", "Drawings"), ("rule-16", "Models or samples")]),
    ]
    queries = [item for item in dataset["queries"] if item["source"] == "ipindia" and item["reviewed"]]
    by_id = {doc["document_id"]: doc for doc in documents}
    for number, (query, selectors) in enumerate(specs, 11):
        targets = []
        for suffix, phrase in selectors:
            candidates = [c for c in chunks if c["source_url"].endswith("/" + suffix)
                          and phrase.casefold() in c["text"].casefold()]
            if not candidates:
                raise ValueError(f"archive_passage_not_found:{suffix}:{phrase}")
            chunk = candidates[0]
            doc = by_id[chunk["document_id"]]
            unit = {"source": "ipindia", "urls": [chunk["source_url"]], "contains": phrase,
                    "raw_sha256": [doc["sha256"]]}
            for key in ("section", "rule"):
                if chunk.get(key):
                    unit[key] = chunk[key]
            if chunk.get("page_start"):
                unit["page"] = chunk["page_start"]
            targets.append(unit)
            print(json.dumps({"query": query, "source_url": chunk["source_url"],
                              "passage": chunk["text"][:250]}, ensure_ascii=False))
        queries.append({"id": f"ipindia-{number:02}", "source": "ipindia", "query": query,
                        "reviewed": True, "reviewer": "Codex archive passage inspection",
                        "review_note": "Selected from archived original passage before ranking changes; no legal-status conclusion.",
                        "targets": targets})
    frozen = {"schema_version": 2, "frozen": True, "scope": "ipindia_only",
              "frozen_at": datetime.now(timezone.utc).isoformat(), "baseline_generation": active,
              "original_label_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
              "annotation_method": "Archive-selected before tuning; first ten labels copied verbatim; no search output used.",
              "queries": queries}
    target.write_text(json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8")
    target.with_suffix(".sha256").write_text(hashlib.sha256(target.read_bytes()).hexdigest(), encoding="ascii")
    print(json.dumps({"path": str(target), "queries": len(queries),
                      "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
