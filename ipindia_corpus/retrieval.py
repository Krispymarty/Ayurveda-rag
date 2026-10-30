"""Versioned passage retrieval; document identity is separate from topic intent."""
import re

RANKING_VERSION = "v2"
CANDIDATE_LIMIT = 100
RRF_CONSTANT = 60
DOMAINS = ((r"\bpatents?\b", "Patent"), (r"\btrade\s*marks?|\btrademarks?", "Trademark"),
           (r"\bdesigns?\b", "Design"), (r"biological diversity|biodiversity", "Biodiversity / ABS"),
           (r"geographical indications?", "GI"), (r"drugs? and cosmetics?", "AYUSH regulation"))
STOPWORDS = {"a", "an", "the", "of", "and", "or", "in", "on", "for", "to", "under", "with", "by", "is", "are"}


def provision_ids(query):
    result = {"sections": [], "rules": []}
    for match in re.finditer(r"\b(sections?|rules?)\s+(\d+[a-z]?(?:\s*(?:,|and|&)\s*\d+[a-z]?){0,11})\b", query, re.I):
        key = "sections" if match[1].lower().startswith("section") else "rules"
        result[key].extend(re.findall(r"\d+[a-z]?", match[2].lower()))
    return {key: list(dict.fromkeys(values)) for key, values in result.items()}


def query_parameters(query):
    value = query.casefold()
    act, rules = bool(re.search(r"\bacts?\b", value)), bool(re.search(r"\brules?\b", value))
    kind = "Act" if act and not rules else "Rules" if rules and not act else None
    domains = [(pattern, label) for pattern, label in DOMAINS if re.search(pattern, value)]
    domain = domains[0][1] if len(domains) == 1 else None
    years = re.findall(r"\b(?:19|20)\d{2}\b", value)
    intent = value
    if kind and domain:
        intent = re.sub(domains[0][0], " ", intent)
    if kind:
        intent = re.sub(r"\b(?:acts?|rules?|(?:19|20)\d{2})\b", " ", intent)
    intent = re.sub(r"\bsections?\s+\d+[a-z]?(?:\s*(?:,|and|&)\s*\d+[a-z]?)*", " ", intent)
    intent = re.sub(r"\b\d+[a-z]?\b", " ", intent)
    tokens = list(dict.fromkeys(token for token in re.findall(r"\w+", intent, re.UNICODE)
                                if len(token) > 1 and token not in STOPWORDS))
    lexical = " OR ".join(tokens) or query
    return {"requested_type": kind, "requested_domain": domain if kind else None,
            "requested_year": years[0] if len(set(years)) == 1 and kind else None,
            "lexical_query": lexical, "intent_query": " ".join(tokens) or query,
            **provision_ids(query)}


# Full stream ranks are retained for diagnostics; only bounded heads enter RRF.
CTES = """
WITH eligible AS (
  SELECT c.* FROM corpus_chunks c JOIN corpus_documents d USING(generation_id,document_id)
  WHERE c.generation_id=%(generation_id)s
    AND (%(source)s::text IS NULL OR c.metadata->>'source'=%(source)s::text)
    AND (%(domain)s::text IS NULL OR c.metadata->>'legal_domain'=%(domain)s::text)
    AND (%(requested_type)s::text IS NULL OR d.metadata->>'document_type'=%(requested_type)s::text)
    AND (%(requested_domain)s::text IS NULL OR c.metadata->>'legal_domain'=%(requested_domain)s::text)
    AND (%(requested_year)s::text IS NULL OR position(%(requested_year)s::text IN coalesce(d.metadata->>'title',''))>0)
), features AS (
  SELECT c.*,e.embedding <=> %(embedding)s::vector distance,
    ts_rank_cd(c.body_lexical,websearch_to_tsquery('english',%(lexical_query)s),34) body_score,
    (ts_rank_cd(c.heading_lexical,websearch_to_tsquery('english',%(lexical_query)s),34)
     + 2*ts_rank_cd(to_tsvector('english',coalesce(c.metadata->>'heading','')),websearch_to_tsquery('english',%(lexical_query)s),34)) heading_score,
    (lower(coalesce(c.metadata->>'section',''))=ANY(%(sections)s::text[]) OR lower(coalesce(c.metadata->>'rule',''))=ANY(%(rules)s::text[])) exact_provision
  FROM eligible c JOIN embedding_cache e ON e.content_hash=c.content_hash AND e.model_revision=%(revision)s
), lexical_all AS (
  SELECT chunk_id,row_number() OVER(ORDER BY body_score DESC,chunk_id) rank FROM features WHERE body_score>0
), semantic_all AS (
  SELECT chunk_id,row_number() OVER(ORDER BY distance,chunk_id) rank FROM features
), heading_all AS (
  SELECT chunk_id,row_number() OVER(ORDER BY heading_score DESC,chunk_id) rank FROM features WHERE heading_score>0
), provision_all AS (
  SELECT chunk_id,row_number() OVER(ORDER BY unit_rank,distance,chunk_id) rank FROM (
    SELECT *,row_number() OVER(PARTITION BY metadata->>'section',metadata->>'rule' ORDER BY distance,chunk_id) unit_rank
    FROM features WHERE exact_provision
  ) units
), fused AS (
  SELECT chunk_id,sum(score) score FROM (
    SELECT chunk_id,1.0/(60+rank) score FROM lexical_all WHERE rank<=100
    UNION ALL SELECT chunk_id,1.0/(60+rank) FROM semantic_all WHERE rank<=100
    UNION ALL SELECT chunk_id,1.0/(60+rank) FROM heading_all WHERE rank<=100
    UNION ALL SELECT chunk_id,1.0/(60+rank) FROM provision_all WHERE rank<=100
  ) candidates GROUP BY chunk_id
), scored AS (
  SELECT f.chunk_id,f.text,f.metadata,f.content_hash,f.document_id,x.score
    + CASE WHEN f.exact_provision THEN .05 ELSE 0 END
    - CASE WHEN length(f.text)<140 AND f.text ~ '^(SECTION|RULE|CHAPTER) [0-9A-Za-z]+[. ]' AND f.text !~ '\\([0-9a-z]\\)' THEN .05 ELSE 0 END final_score
  FROM fused x JOIN features f USING(chunk_id)
), deduplicated AS (
  SELECT *,row_number() OVER(PARTITION BY document_id,content_hash ORDER BY final_score DESC,chunk_id) duplicate_rank FROM scored
), ranked AS (
  SELECT *,row_number() OVER(ORDER BY final_score DESC,chunk_id) fused_rank FROM deduplicated WHERE duplicate_rank=1
)
"""
SEARCH_SQL = CTES + "SELECT chunk_id,text,metadata,final_score FROM ranked ORDER BY fused_rank LIMIT %(limit)s"


def diagnostic_rows(connection, args):
    return connection.execute(CTES + """
      SELECT f.chunk_id,f.text,f.metadata,l.rank,s.rank,p.rank,r.fused_rank,r.final_score,
        CASE WHEN l.rank<=100 THEN 1.0/(60+l.rank) ELSE 0 END,
        CASE WHEN s.rank<=100 THEN 1.0/(60+s.rank) ELSE 0 END,
        coalesce(CASE WHEN p.rank<=100 THEN 1.0/(60+p.rank) ELSE 0 END,0)
      FROM features f LEFT JOIN lexical_all l USING(chunk_id) LEFT JOIN semantic_all s USING(chunk_id)
      LEFT JOIN provision_all p USING(chunk_id) LEFT JOIN ranked r USING(chunk_id)
      ORDER BY r.fused_rank NULLS LAST,f.chunk_id
    """, args).fetchall()
