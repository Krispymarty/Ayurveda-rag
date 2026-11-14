# AyurIP v1 — public research release candidate

Prepared in `E:\Ayurveda Rag` on 14 November 2026. The owner explicitly deferred cloud deployment. **There is no public deployment URL or remote AI preview.** Local service verification and cloud deployment are different release stages.

AyurIP searches archived official IP India passages using PostgreSQL full-text retrieval, multilingual E5 vectors and hybrid ranker v2. The existing FastAPI service serves the responsive HTML/CSS/JavaScript interface from the same origin. Results show copied source text, official citation links, document/chunk IDs, page/provision anchors, capture time and original SHA-256. Legal status remains unknown.

The prepared corpus contains 307 IP India documents and 8,880 active chunks in generation `20261014T204940358650Z`. Rollback generation `20261014T190100599320Z` retains 7,755 chunks and ranker v1. India Code, TKDL, NBA and Gazette candidates are not admitted by this release. Public `/answer` remains HTTP 503. The local private Passport still uses the existing pinned Qwen3.5 4B; it is not part of the internet-facing image's routes.

Start here:

1. `FINAL_STATUS.md`: measured state, limitations and remaining release decisions.
2. `TESTING.md`: executed checks and their exact scope.
3. `DEPLOYMENT.md`: approved future deployment procedure and rollback.
4. `DEMO.md`: representative research workflow.

The submission ZIP contains sanitized application source, controlled tests, deployment configuration, documentation and selected verification evidence. It contains **no credentials, database volume, user inputs or model weights**. The separate state archive contains the verified database export and original evidence needed for provisioning. Model files must be supplied from the pinned, hash-bound existing cache. The local prepared build context and tested OCI image are separate assets; this ZIP alone is not a running service.

Local development continues with `docker compose up -d db api` from the original workspace. Do not start scheduler/candidate profiles. Private Passport remains loopback-only at port 8003 with its existing startup procedure. See the deployment guide for the isolated production container, which uses a separate database and must never overwrite the original `sih_pgdata` volume.
