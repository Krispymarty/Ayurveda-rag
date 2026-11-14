# Executed release verification — 14 November 2026

These are **local release results**, not external cloud, legal-relevance or public security certification. Raw observations and failed attempts remain under `E:\Ayurveda Rag\data\cloud_release_20261114`; selected sanitized evidence accompanies the submission.

| Verification | Actual outcome |
|---|---|
| Relevant offline suites | 135 tests; zero failures, errors or skips |
| Existing PostgreSQL integration suite | 2 tests passed in a uniquely named disposable database on the isolated cluster; removed only that test DB afterwards |
| Ruff | All application, test and tool checks passed |
| Database export/restore | 614 document-version rows, 16,635 chunk rows, 14,081 embeddings, 2 index-generation records, 3 release-state keys; full table digests match |
| Transport/privileges | TLS 1.3 with certificate verification; separate SELECT-only role; actual UPDATE rejected |
| Source originals | Both immutable generations validate; 307 unique raw originals, 142,769,087 bytes |
| Published citations | Every hit in all 90 timed requests matches published text, identity, official URL and archived raw digest |
| Retrieval | 30 unchanged regression questions, 3 timed passes after warmup: 96.67% macro Recall@5; 235.27 ms warm p95 |
| Baseline equivalence | All 90 requests returned the same ordered top-five chunk IDs as the original service's corresponding query |
| Final actual browser | 14 checks passed; three real searches, desktop 1440 px/mobile 390 px, quotations/provenance/source panel/filters/keyboard recovery/icon; zero JS/console errors or failed requests |
| Public boundaries | `/answer` 503 `three_source_not_certified`; private routes/docs/archive traversal unavailable; bad Host/Origin rejected; oversized query rejected; real 429 rate limiting observed |
| Process/database recovery | Exact final image restarts ready; database-down health 503 while liveness stays 200; persistent data and search recover |
| Rollback | Actual isolated v1 rollback search verified; binding changes fail closed; original v2 binding restored exactly |
| Restore safety | Actual populated target rejected; original source endpoint fingerprint rejected before connection |

The historical regression set is `ipindia-reviewed-20261015-v1/expanded.json`, SHA-256 `acb31a0a3208b6899efd7381ac2adeebf0a4ee97bee5719cdd86b784bd4b7c25`. Its annotations are existing developer archive checks, not independent human legal review. The same `ipindia-10` miss remains; labels and scores were not relaxed. Thirty repeated questions provide thirty distinct questions, not ninety independent cases. Historical `archived-20` and all frozen artifacts remain unchanged.

Retrieval was benchmarked on the first passing production image. The final image changes only `web/index.html` and adds `web/favicon.svg`; manifest comparison proves identical backend, model, database and archive bytes. Final-image browser and restart checks passed after this static-only correction. No ranker or model-input tuning was performed.

Hardware: Intel Core i7-13620H, 10 physical/16 logical cores. The app test container was limited to **1 CPU/2 GiB** and the database to **0.5 CPU/1 GiB**, with one app process and four inference threads. Torch is `2.14.1+cpu`; no CUDA runtime or new model is required by this image. Full dependency and embedding file locks are supplied. The app ran as UID 10001 with a read-only filesystem, dropped capabilities and no-new-privileges. `/health` became ready 15.71 seconds after final-image startup; a subsequent final-image process restart became ready at 17.69 seconds. Internal initialization and individual observations are retained. These are fresh-process measurements with filesystem caches potentially warm, not a reliable startup p95 or cold-filesystem claim.

The real HTTP regression invocation was:

```text
python -B -m tools.release_smoke --base-url http://127.0.0.1:18000 --baseline-url http://127.0.0.1:8000 --allow-local --output data/cloud_release_20261114/http_smoke.json
```

The harness saves every request attempt separately and paces outside measured request durations to respect the public quota. Its HTTPS mode is ready for a future real domain; it has not been executed externally. The exact fifteen offline test patterns/results are in `evidence/offline_tests.json`; the original SQL suite covers failed indexing retention and passage/version duplicate handling.

Retained failures include a package-download timeout, invalid cache-layout assumptions rejected by the packager, certificate/IP-hostname mismatch, Docker export failure when C: became full, the initial missing icon and the CSP rejection of an inline icon. None was replaced with a claimed passing observation. Model files are now bound by explicit file digests; TLS uses the verified DNS name; the icon is same-origin with CSP unchanged. Disk recovery removed only 19.7 GB of disposable pip download cache, preserving installed packages, model caches and volumes. The earlier browser fixture cleanup failure and all subsequent actual-browser observations remain retained.

**Not executed/certified:** real cloud HTTPS, DNS/provider TLS redirects, external unauthorized access, managed-cloud restart/redeploy, provider logs/billing/resource measurement, public load/security certification, independent legal review, three-source certification, remote Passport authentication/inference, 95% answer-quality gates or five-second complete-answer latency. These remain release dependencies, not implied passes.
