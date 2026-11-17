# AyurIP evidence service

**14 November 2026 release preparation:** the public research UI and isolated CPU container are verified locally; cloud deployment was explicitly deferred by the owner. The protected corpus and ranker remain unchanged. The unchanged 30-query regression retained 96.67% Recall@5 with 235 ms warm p95 on the constrained release container. Public answers remain blocked and the private Passport remains loopback-only. Submission documents and verified archives are prepared separately; historical report links below refer to the original local workspace.

Active workspace: **`E:\Ayurveda Rag`**. The older `E:\SIH` checkout is a preserved backup.

The local UI is available at `http://127.0.0.1:8000/` when Compose is running. This is an internal evidence-search service. IP India is indexed; staged India Code and TKDL files remain excluded because original capture and passage-fidelity evidence is incomplete. Legal status remains unknown.

Current implementation and release evidence: [milestone report](reports/current_milestone.md), [source certification](reports/corpus_certification.md), [unified readiness](reports/unified_readiness.md), [local model evaluation](reports/llm/model_selection.md), and [Docker recovery](reports/docker_recovery.md).

Latest: [Reliability Milestone 2](reports/reliability_v2/milestone.md), measured 20 October 2026 locally. The isolated experiments retained 630 final evaluation requests and 150 development ablations. Unchanged search measured 96.67% Recall@5 and 538 ms warm p95; all 658 emitted final citations matched the archive. Private HTTP full-case success was 65.4%, with 3.52 s pooled warm p95. The answer candidate failed the 95% quality gates and was not activated. The new 60-case seed package remains unreviewed; both previews are stopped.

Previous [readiness and private answer reliability results](reports/reliability/milestone.md): 96.67% Recall@5 / 594 ms search p95, and 3.96 s private HTTP p95. Those historical scores and frozen labels remain unchanged. Public answers and fresh synchronization remain disabled; all legal statuses remain unknown.

The separate [backend-copied span experiment](reports/llm/span_selection.md) passed 126/135 frozen extraction cases with 1.11 s warm model p95 and exact archived quotations. It remains isolated because current-law abstention regressed; this result does not certify legal answers or end-to-end latency.

The retained active generation is `20261014T204940358650Z`, ranker v2: **307 documents and 8,880 chunks**. Source synchronization is paused and public answers remain disabled. The model experiments have separate configuration, immutable observations and resumable checkpoints; [reproduction instructions](reports/llm/README.md) describe their scope.

From this directory, start the retained database and API:

```powershell
docker compose up -d db api
```

The API now initializes archive validation and the embedding model before admitting searches. `/live` reports process liveness; `/health` returns HTTP 503 with a preparation/failure reason until initialization finishes, then HTTP 200. The UI shows “Preparing evidence search…” and keeps the question in place. Measured initialization on this machine took approximately 21–27 seconds; that work no longer runs unexpectedly inside the first admitted search. A failed initialization requires resolving its local-log cause and restarting only the API.

The separate answer reliability pipeline is an **experimental, opt-in local evaluation service**. It uses the pinned Qwen model, backend request policy, opaque span IDs, a second relevance-verification pass and backend-copied citations. It does not enable the main `/answer` endpoint. Its evaluation is complete and the preview was stopped afterward; source synchronization remains paused.

The optional candidate-only worker preserves the active index and skips sources without approved access evidence:

```powershell
docker compose --profile candidate up -d candidate-worker
docker compose exec -T api python -m ipindia_corpus.candidate_sync status
```

Its defaults are daily conditional checks and weekly discovery once a source is permitted. The separate legacy `sync` profile remains disabled. Public answers are blocked until certification matches an active three-source generation. Local Ollama selection and exact-quotation validation are implemented for developer verification; no hosted answer key is required.

## Original ingestion workflow

A bounded crawler for public IP India Acts, Rules, Manuals, Guidelines, forms, and filing guidance. It produces source linked JSONL documents and legal-structure chunks for a later RAG index. It does not query application records, accounts, CAPTCHA pages, or other hosts.

## Install and run

In PowerShell, copy only the command lines below. The triple backticks and the word `powershell` are Markdown formatting, not commands.

```powershell
python -m pip install -r requirements.txt
python -m ipindia_corpus.pipeline preflight --out corpus
```

Inspect `corpus/audit.jsonl` and any website terms/policy pages it finds. The preflight intentionally exits with code 2 and `manual_review_required` until an operator reviews the site's terms and records that review:

```powershell
python -m ipindia_corpus.pipeline crawl --out corpus --terms-reviewed --max-pages 100 --max-depth 3 --delay 1.5
```

If you already have a crawl, apply extractor and filter fixes to cached files without any network request:

```powershell
python -m ipindia_corpus.pipeline rebuild --out corpus
```

`--terms-reviewed` is an operator attestation, not an automatic legal conclusion. Each run checks `robots.txt`, checks the public homepage, and tries policy links. An HTTP 404 for `robots.txt` is recorded as no published rules; network, access, or server errors fail closed. For live use, record the reviewed policy URL, date, and reviewer in your project records before setting the flag.

The crawler starts from the homepage and official legal resource indexes, then discovers internal links and sitemap entries. Resource indexes are used for discovery but excluded from the RAG export. It prioritizes relevant URLs, excludes individual application records and unrelated site areas, and caps pages, depth, and response size. It waits at least one second between network requests (1.5 seconds by default), backs off on transient failures, and never follows a redirect outside `ipindia.gov.in` or `www.ipindia.gov.in`. Other `.gov.in` links found on pages are only listed in `external_official_links.jsonl`. A 401/403, challenge, restricted resource, image-only PDF, unsupported format, or unverifiable policy goes to `audit.jsonl` as `manual_review_required`.

Cached downloads live under `corpus/cache`; per-document metadata and chunks live under `corpus/documents/<document_id>/`. `corpus/documents.jsonl` and `corpus/chunks.jsonl` are rebuilt at the end of a crawl or with `python -m ipindia_corpus.pipeline export --out corpus`. Reruns reuse cached bytes and overwrite per-document outputs, so the aggregate exports do not duplicate documents. Preserve raw files and audit records for provenance.

`corpus/review_queue.jsonl` contains the latest unresolved manual-review items. The exports deduplicate identical source bytes served under multiple official URLs and list alternate download URLs in document metadata.

## Audited canonical export

The repository contains an audited IP India archive plus staged India Code and public TKDL candidates. India Code/TKDL originals and capture provenance remain incomplete. The first audit and repair plan are under `reports/`. Earlier derived exports remain at the workspace root and under `data/documents/` and `data/chunks/`; the accepted searchable exports are bound to the active immutable generation under `data/generations/`. Content-addressed raw snapshots are under `data/raw/ipindia/`; `data/manifests/manifest.sqlite` holds the version ledger. TKDL candidates are staged in `data/documents/tkdl_candidates.jsonl` and excluded from legal retrieval until their original HTML and retrieval dates can be verified.

To rebuild the canonical export entirely from archived IP India bytes, run these command lines from `E:\Ayurveda Rag` (do not type the Markdown fence lines):

```powershell
python tools\build_derived.py
python tools\validate_derived.py
python -m unittest discover -s tests -q
```

The existing `corpus/` cache is retained. The canonical builder preserves raw hashes, fixes document citation URLs, tracks PDF pages, marks unverified legal status `unknown`, and does not make network requests. The version ledger supports immutable snapshots and change records for future source connectors; unattended sync is disabled in `sources.csv` pending source-specific policy and legal metadata review. `reports/ipindia_ingestion_report.md` and `reports/sync_architecture.md` describe the current state and gaps.

HTML uses BeautifulSoup; PDFs use PyMuPDF text extraction with no automatic OCR; DOCX uses python-docx; XLSX uses openpyxl; XLS uses xlrd; CSV uses the standard library. Legacy DOC uses `antiword` if installed, otherwise it is marked for manual review. Complex layouts and tables should be reviewed before indexing; Docling may be added as a separate quality pass.

Chunks begin at detected chapters, sections, rules, and, for Patents Act Section 3, individual clauses. Oversized units are split at line boundaries while retaining their legal metadata. `publication_date`, `effective_from`, `effective_to`, amendment relationships, version, and language remain unset when they cannot be established confidently from the source; downstream legal review should fill those fields. This crawler does not claim that every public page is a current consolidated law.

Run offline checks with:

```powershell
python -m unittest discover -s tests -v
```

If PowerShell prints `` `powershell is not recognized``, a Markdown fence was pasted into the shell. Enter only the lines beginning with `python`. The `RequestsDependencyWarning` seen in the system Python installation comes from its globally installed `chardet 7.4.3` with `requests 2.32.3`; it did not stop the verified crawl or tests. A project virtual environment avoids that unrelated global package combination.

## Three-source local evidence search

The new local service uses the curated [source manifest](config/curated_sources.json), the existing SQLite version ledger, immutable source snapshots, validated [corpus generations](ipindia_corpus/generations.py), and PostgreSQL full-text plus pgvector search. It returns evidence passages only. Every result states `status: unknown`; public TKDL pages are tagged `institutional_background`, not legal authority. The protected TKDL formulation database is excluded.

The active snapshot contains **307 IP India documents and 8,880 searchable chunks**, including 994 retained short units. All 82 PDFs passed archived-original text and page validation; source hashes are verified separately. India Code and TKDL require permitted live recapture before entering retrieval. All three policies remain unapproved in [source_policies.json](config/source_policies.json). Recorded access evidence is in `reports/access/`: India Code robots returned HTTP 500, TKDL connections reset, and IP India's linked privacy policy leaves automated access unresolved. A previous `--terms-reviewed` flag is not a persistent review record.

India Code now uses the verified `indiacode.gov.in` public REST service. The connector follows bounded official metadata links, verifies central instrument identity, and archives metadata alongside PDF bytes. Historical `nic.in` URLs remain allowed. Missing linked subordinate legislation produces a review item and blocks release readiness. TKDL discovery covers only curated public institutional and Ayurveda information pages.

From PowerShell in `E:\Ayurveda Rag`, enter each command as a single line, without Markdown fence markers:

1. If `.env` is absent, copy `.env.example` to `.env` and replace `AYURIP_DB_PASSWORD` with a private alphanumeric password. Start Docker Desktop from the Windows Start menu, then run `docker compose up -d db`.
2. Start the API against the retained index with `docker compose up -d api`.
3. Inspect `http://127.0.0.1:8000/health` and `/sources` for the active searchable generation. The existing embedding revision and cache are pinned in `data/model.lock.json`.
4. Open `http://127.0.0.1:8000/docs`. Example: `http://127.0.0.1:8000/search?q=traditional%20knowledge&top_k=5`. New corpus/ranking candidates use the isolated preview and gated activation commands below.
5. Run recorded-response and static checks with `python -m tools.verify_offline`; run PostgreSQL failure checks in an automatically created disposable database with `docker compose exec -T api python -m tools.verify_postgres`. Validate passage preservation with `python -m tools.validate_passages`.

Capture current access evidence with `docker compose run --rm api python -m ipindia_corpus.access_review --source SOURCE`; optional `--policy-url` values must identify actual official access terms. The capture command does not approve access. After inspecting evidence that permits retrieval, record the conclusion with `docker compose run --rm api python -m ipindia_corpus.sync_runner record-review --source SOURCE --reviewer "REVIEWER" --policy-url "https://OFFICIAL-SITE/POLICY" --notes "EVIDENCE AND CONCLUSION" --permitted-prefix "/PUBLIC-PATH/"`. Repeat the prefix option for each permitted path. Approval requires archived receipts, matching hashes, a successful policy response, and verified robots evidence. The configured crawler contact is `https://github.com/Krispymarty/Ip-Shakti--Rag`.

For an approved source, run `docker compose run --rm api python -m ipindia_corpus.sync_runner sync --source SOURCE --discover`, followed by two successful checks with the same command without `--discover`. Run workers through Docker. Source and publication locks prevent overlapping workers; checkpoints survive interruptions. A partial capture remains outside accepted snapshots. Capture, publication, and indexing failures are recorded separately. Long `Retry-After` delays are persisted and respected. Failed candidate indexing retains the database's active generation; embeddings are reused by content hash and model revision.

The evaluation set has 30 draft queries, with ten reviewed IP India passage targets and twenty pending fresh-source review. Targets require section/rule, page, or passage text and bind to source hashes. The reviewed IP India subset was fixed before ranking changes. Review pending targets with `python -m ipindia_corpus.evaluation_labels review --generation-id GENERATION --query-id QUERY --url OFFICIAL_URL --section SECTION --contains "SOURCE TEXT" --reviewer "REVIEWER" --note "PASSAGE EVIDENCE"` inside the API container. Use `--rule` or `--page` when appropriate. Once all targets are reviewed, use the same module's `freeze` command with generation, reviewer, and note arguments. Freeze checks source and passage availability; it does not run retrieval.

Record hardware with `tools\capture_hardware.ps1`, then run `docker compose run --rm api python -m ipindia_corpus.evaluate`. Primary evaluation uses unfiltered HTTP search over three warm passes; source-filtered results are diagnostic. `--partial` measures only reviewed queries and cannot pass the three-source gate. The initial partial result was **70% Recall@5 and approximately 0.4 seconds warm p95**. The active reviewed IP India results are **90% / 714 ms** on the original ten queries and **96.67% / 646 ms** on the expanded thirty; see the [milestone report](reports/current_milestone.md). Document-only hits do not count as correct passage retrieval.

Run `docker compose run --rm api python -m ipindia_corpus.release_gates` to inspect readiness. Only after access reviews, fresh three-source coverage, discovery plus two incremental cycles, passage and database checks, frozen labels, and the ≥90% Recall@5 / ≤2-second p95 evaluation pass, run `docker compose run --rm api python -m ipindia_corpus.sync_runner enable-scheduler`. Then start the loop with `docker compose --profile sync up -d scheduler`. Daily checks and weekly discovery have independently persisted due times. The scheduler remains disabled now.

The API is bound to loopback in [compose.yaml](compose.yaml). The local database password comes from `.env`, which is ignored by Git. The source archives and review records remain on disk under `data/`; preserve them when backing up or moving this workspace. See [first release status](reports/first_release_status.md) for the currently unverified gates.

`/sources` reports the database's active searchable generation and source publication/access status. `/health` returns stable readiness reasons without database exception details. Every returned search hit is checked against an archived source hash and its document identity. Scraped text is untrusted data and cannot change connector settings or scheduler controls.

## Offline retrieval work and workspace migration

The active workspace is **`E:\Ayurveda Rag`**. Its original `section_output`, `tkdl_corpus_final`, and regulatory PDFs are preserved. They remain outside the searchable generation pending provenance and passage review. The previous folder is retained as a backup. `reports/workspace_migration.json` records the transferred archive checks. The Docker project name remains `sih` to reuse the existing `sih_pgdata` database volume; the API's `/app` mount points to the new folder.

Fresh synchronization is deferred and scheduling remains disabled. Historical frozen labels and reports are preserved. A separate factual-review bundle, `data/benchmarks/ipindia-reviewed-20261015-v1`, corrects the Designs Act fee target from physical PDF page 2 to page 3, with source hash and quoted page evidence. This is archived-source factual review, not qualified legal review. The reviewed bundle was used for the accepted retrieval release.

Offline candidate and diagnostic commands (run from this folder):

```powershell
python -m ipindia_corpus.generations --repair-pdf --candidate
docker compose exec -T -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 api python -m ipindia_corpus.retrieval_preview prepare --generation-id GENERATION
docker compose exec -T -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 api python -m ipindia_corpus.retrieval_preview evaluate
docker compose exec -T api python -m ipindia_corpus.ranking_diagnostics --version v1 --out reports/ranking_baseline.json
docker compose exec -T api python -m ipindia_corpus.retrieval_release check
```

The preview uses a separate PostgreSQL database and a temporary internal API on port 8001. Its server stops after evaluation. Indexing reuses cached embeddings and never changes the production selection. `retrieval_release activate` requires validated frozen labels, both quality evaluations, passage preservation, and verification evidence. `retrieval_release rollback` restores the recorded generation/ranker pair. `retrieval_preview cleanup` removes only its disposable database after review.

The earlier uncorrected-label candidate measured **90% Recall@5 / 468 ms p95** and **93.3% / 363 ms**; those historical results remain in [offline retrieval status](reports/offline_retrieval_status.md) and [annotation evidence](reports/annotation_review.json). After the separate factual review and release checks, generation `20261014T204940358650Z` with ranker v2 was activated, measuring **90% / 714 ms** and **96.67% / 646 ms** on the reviewed sets. The previous generation/ranker remain available for rollback. API embedding loading uses the verified local cache in offline mode; initial installation on another machine requires explicitly enabling model downloads.
