"""Local ranking explanations; never exposed by the public search endpoint."""
import argparse
import json
from pathlib import Path

from . import retrieval_legacy
from .indexer import _connect, model_revision
from .passage_evaluation import matches
from .search_api import _model
from .sync_runner import ROOT


def parameters(query, generation_id, source=None, domain=None):
    from .cli_readiness import initialize_for_cli
    initialize_for_cli()
    import re
    vector = _model().encode(["query: " + query], normalize_embeddings=True)[0]
    provision = re.search(r"\b(section|rule)\s+(\d+[a-z]?)\b", query, re.I)
    return {"generation_id": generation_id, "query": query, "source": source, "domain": domain,
            "revision": model_revision(), "embedding": "[" + ",".join(str(float(v)) for v in vector) + "]",
            "section": provision.group(2).lower() if provision and provision.group(1).lower() == "section" else None,
            "rule": provision.group(2).lower() if provision and provision.group(1).lower() == "rule" else None,
            "limit": 5, **retrieval_legacy.query_parameters(query)}


def legacy_rows(connection, args):
    prefix = retrieval_legacy.SEARCH_SQL.split("SELECT c.chunk_id,c.text,c.metadata,f.score")[0].rstrip()
    sql = prefix + """, scored AS (
      SELECT c.chunk_id,f.score
       + CASE WHEN lower(c.metadata->>'section')=%(section)s::text OR lower(c.metadata->>'rule')=%(rule)s::text THEN .05 ELSE 0 END
       - CASE WHEN length(c.text)<140 AND c.text ~ '^(SECTION|RULE|CHAPTER) [0-9A-Za-z]+[. ]' AND c.text !~ '\\([0-9a-z]\\)' THEN .05 ELSE 0 END final_score
      FROM fused f JOIN eligible c USING(chunk_id)
    ), overall AS (
      SELECT *,row_number() OVER(ORDER BY final_score DESC,chunk_id) fused_rank FROM scored
    ), full_lexical AS (
      SELECT chunk_id,row_number() OVER(ORDER BY ts_rank_cd(lexical,websearch_to_tsquery('simple',%(lexical_query)s),32) DESC,chunk_id) rank
      FROM eligible WHERE lexical @@ websearch_to_tsquery('simple',%(lexical_query)s)
    ), full_semantic AS (
      SELECT c.chunk_id,row_number() OVER(ORDER BY e.embedding <=> %(embedding)s::vector,c.chunk_id) rank
      FROM eligible c JOIN embedding_cache e ON e.content_hash=c.content_hash AND e.model_revision=%(revision)s
    )
    SELECT c.chunk_id,c.text,c.metadata,l.rank,s.rank,p.rank,o.fused_rank,o.final_score,
      coalesce(1.0/(60+lc.rank),0),coalesce(1.0/(60+sc.rank),0),coalesce(1.0/(60+p.rank),0)
    FROM eligible c LEFT JOIN full_lexical l USING(chunk_id) LEFT JOIN full_semantic s USING(chunk_id)
    LEFT JOIN provisions p USING(chunk_id) LEFT JOIN overall o USING(chunk_id)
    LEFT JOIN lexical lc USING(chunk_id) LEFT JOIN semantic sc USING(chunk_id)
    ORDER BY o.fused_rank NULLS LAST,c.chunk_id
    """
    return connection.execute(sql, args).fetchall()


def diagnose(query, *, generation_id=None, targets=(), version="v1", top_k=5):
    connection = _connect()
    try:
        generation_id = generation_id or connection.execute("SELECT value FROM app_state WHERE key='active_generation'").fetchone()[0]
        args = parameters(query, generation_id)
        if version == "v1":
            rows = legacy_rows(connection, args)
        else:
            from .retrieval import diagnostic_rows, query_parameters
            args.update(query_parameters(query))
            rows = diagnostic_rows(connection, args)
        details = []
        for row in rows:
            chunk_id, text, metadata, lexical, semantic, provision, fused, score, lc, sc, pc = row
            details.append({"chunk_id": chunk_id, "document_id": metadata["document_id"],
                            "source_url": metadata["source_url"], "section": metadata.get("section"),
                            "rule": metadata.get("rule"), "page_start": metadata.get("page_start"),
                            "page_end": metadata.get("page_end"), "text": text, "eligible": True,
                            "lexical_rank": lexical, "vector_rank": semantic, "provision_rank": provision,
                            "fused_rank": fused, "score": float(score) if score is not None else None,
                            "contributions": {"lexical": float(lc or 0), "vector": float(sc or 0), "provision": float(pc or 0),
                                              "heading_and_adjustments": float(score or 0) - float(lc or 0) - float(sc or 0) - float(pc or 0)},
                            "candidate_missing": fused is None})
        target_details = []
        from .indexer import generation
        _, _, archived = generation(generation_id=generation_id)
        for target in targets:
            ids = {c["chunk_id"] for c in archived if matches(c, target)}
            found = [d for d in details if d["chunk_id"] in ids]
            target_details.append({"target": target, "archived_chunk_ids": sorted(ids),
                                   "eligible": bool(found), "candidates": found,
                                   "failure": "annotated_passage_missing" if not ids else "ineligible" if not found else "candidate_missing" if all(d["candidate_missing"] for d in found)
                                   else "rank_below_top_k" if all(d["fused_rank"] is None or d["fused_rank"] > top_k for d in found) else None})
        return {"query": query, "generation_id": generation_id, "ranking_version": version,
                "parameters": {k: v for k, v in args.items() if k != "embedding"},
                "targets": target_details, "top_candidates": [d for d in details if d["fused_rank"] and d["fused_rank"] <= top_k]}
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query")
    parser.add_argument("--labels", type=Path, default=ROOT / "tests/retrieval_queries.json")
    parser.add_argument("--ids", nargs="*", default=["ipindia-08", "ipindia-09", "ipindia-10"])
    parser.add_argument("--version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--generation-id")
    parser.add_argument("--preview", action="store_true", help="Use the separately prepared preview database")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.preview:
        import os
        from .retrieval_preview import STATE, preview_dsn
        os.environ["DATABASE_URL"] = preview_dsn(json.loads(STATE.read_text())["database"])
    items = [{"query": args.query, "targets": []}] if args.query else [q for q in json.loads(args.labels.read_text(encoding="utf-8"))["queries"] if q["id"] in args.ids]
    report = [diagnose(item["query"], targets=item["targets"], generation_id=args.generation_id, version=args.version) for item in items]
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized, encoding="utf-8")
    summary = []
    for report_item in report:
        target_summary = [{"failure": target["failure"], "ranks": [
            {key: candidate[key] for key in ("lexical_rank", "vector_rank", "fused_rank")}
            for candidate in target["candidates"]]} for target in report_item["targets"]]
        summary.append({"query": report_item["query"], "targets": target_summary})
    print(serialized if args.query else json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
