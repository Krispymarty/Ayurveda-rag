"""Trusted query parsing and hybrid ranking over a single active generation."""
import re


def query_parameters(query):
    value = query.casefold()
    act, rules = bool(re.search(r"\bact\b", value)), bool(re.search(r"\brules?\b", value))
    kind = "Act" if act and not rules else "Rules" if rules and not act else None
    domain = None
    for pattern, label in ((r"\bpatents?\b", "Patent"), (r"trade\s*marks?|trademarks?", "Trademark"),
                           (r"\bdesigns?\b", "Design"), (r"biological diversity|biodiversity", "Biodiversity / ABS"),
                           (r"geographical indications?", "GI"), (r"drugs? and cosmetics?", "AYUSH regulation")):
        if re.search(pattern, value):
            domain = label
            break
    years = re.findall(r"\b(?:19|20)\d{2}\b", value)
    tokens = re.findall(r"\w+", value, re.UNICODE)
    lexical = " OR ".join(dict.fromkeys(token for token in tokens if len(token) > 1)) or query
    return {"requested_type": kind, "requested_domain": domain if kind else None,
            "requested_year": years[0] if len(set(years)) == 1 and kind else None, "lexical_query": lexical}


SEARCH_SQL = """
WITH eligible AS (
  SELECT c.* FROM corpus_chunks c
  JOIN corpus_documents d USING(generation_id,document_id)
  WHERE c.generation_id=%(generation_id)s
    AND (%(source)s::text IS NULL OR c.metadata->>'source'=%(source)s::text)
    AND (%(domain)s::text IS NULL OR c.metadata->>'legal_domain'=%(domain)s::text)
    AND (%(requested_type)s::text IS NULL OR d.metadata->>'document_type'=%(requested_type)s::text)
    AND (%(requested_domain)s::text IS NULL OR c.metadata->>'legal_domain'=%(requested_domain)s::text)
    AND (%(requested_year)s::text IS NULL OR position(%(requested_year)s::text IN coalesce(d.metadata->>'title','') || ' ' || coalesce(d.metadata->>'source_url',''))>0)
), lexical AS (
  SELECT chunk_id,row_number() OVER (ORDER BY ts_rank_cd(lexical,websearch_to_tsquery('simple',%(lexical_query)s),32) DESC,chunk_id) rank
  FROM eligible WHERE lexical @@ websearch_to_tsquery('simple',%(lexical_query)s)
  ORDER BY ts_rank_cd(lexical,websearch_to_tsquery('simple',%(lexical_query)s),32) DESC,chunk_id LIMIT 50
), semantic AS (
  SELECT c.chunk_id,row_number() OVER (ORDER BY e.embedding <=> %(embedding)s::vector,c.chunk_id) rank
  FROM eligible c JOIN embedding_cache e ON e.content_hash=c.content_hash AND e.model_revision=%(revision)s
  ORDER BY e.embedding <=> %(embedding)s::vector,c.chunk_id LIMIT 50
), provisions AS (
  SELECT c.chunk_id,row_number() OVER (ORDER BY e.embedding <=> %(embedding)s::vector,c.chunk_id) rank
  FROM eligible c JOIN embedding_cache e ON e.content_hash=c.content_hash AND e.model_revision=%(revision)s
  WHERE (%(section)s::text IS NOT NULL AND lower(c.metadata->>'section')=%(section)s::text)
     OR (%(rule)s::text IS NOT NULL AND lower(c.metadata->>'rule')=%(rule)s::text)
  ORDER BY e.embedding <=> %(embedding)s::vector,c.chunk_id LIMIT 50
), fused AS (
  SELECT chunk_id,sum(score) score FROM (
    SELECT chunk_id,1.0/(60+rank) score FROM lexical
    UNION ALL SELECT chunk_id,1.0/(60+rank) FROM semantic
    UNION ALL SELECT chunk_id,1.0/(60+rank) FROM provisions
  ) candidates GROUP BY chunk_id
)
SELECT c.chunk_id,c.text,c.metadata,f.score
  + CASE WHEN (%(section)s::text IS NOT NULL AND lower(c.metadata->>'section')=%(section)s::text)
           OR (%(rule)s::text IS NOT NULL AND lower(c.metadata->>'rule')=%(rule)s::text) THEN .05 ELSE 0 END
  - CASE WHEN length(c.text)<140 AND c.text ~ '^(SECTION|RULE|CHAPTER) [0-9A-Za-z]+[. ]'
           AND c.text !~ '\\([0-9a-z]\\)' THEN .05 ELSE 0 END final_score
FROM fused f JOIN eligible c USING(chunk_id)
ORDER BY final_score DESC,c.chunk_id LIMIT %(limit)s
"""
