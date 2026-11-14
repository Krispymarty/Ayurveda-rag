# AyurIP release status — 14 November 2026

**Deployment preparation completed locally. Cloud deployment was deferred by the owner.** No public URL, paid service, external artifact transfer, remote Git push or cloud AI preview was created.

- Working original search/UI: `http://127.0.0.1:8000`.
- Isolated production-image demonstration: `http://127.0.0.1:18000` (loopback only).
- Existing private Passport: `http://127.0.0.1:8003` (loopback only, unchanged serial Qwen workflow).
- Public cloud URL: **none**. Protected cloud AI URL: **none**.

The final local image is `ayurip-release:20261114-ready`, image ID `sha256:ba5b7d301371a0cb895189a154db9634ce991291b4a9e3c4c2b56dc35118feef`. Its immutable context is `data/cloud_release_20261114/runtime-v6`, release-manifest SHA-256 `3b1922d978a00beb503ea5d17ca235be74cbb6dece6c82a25ed79e9ae6600204`. All **384** manifest files verify. Model weights are supplied through the prepared image/context, not the submission ZIP.

The original production database and volume retain generation `20261014T204940358650Z`, ranker `v2`, rollback `20261014T190100599320Z`/`v1`, original archives, benchmarks, source policies and model locks. The final preservation checks pass. Only the existing API/database Compose services run; synchronization/candidate workers are stopped. Legal statuses remain unknown. The NBA candidate and pending human judgments are not admitted.

The new empty test database restored both published generations exactly over verified TLS; application credentials have SELECT-only access. Restart, database outage/recovery, source-target rejection and actual v1/v2 rollback checks passed on isolated resources. The local release retained **96.67% Recall@5, 235 ms warm p95**, exact baseline hit identities and citation fidelity. **135 offline tests, 2 existing PostgreSQL integration tests, Ruff and 14 final actual-browser checks passed.** Desktop/mobile screenshots and exact observations are included. The final image's fresh startup was 15.71 seconds and final process restart 17.69 seconds; these are not cold-filesystem or cloud p95 measurements.

Git checkpoint `d25901a3a757e3d6811024745fb3d23be78c2eae` preserved the initial verified release implementation. The submission's generated `MANIFEST.json` records the final exact source commit and payload hashes; ZIP sidecars record verified archive SHA-256. No remote was added or pushed.

Prepared deliverables:

```text
E:\Ayurveda Rag\submission\AyurIP-v1.zip
E:\Ayurveda Rag\submission\AyurIP-v1.zip.sha256
E:\Ayurveda Rag\submission\AyurIP-v1-state.zip
E:\Ayurveda Rag\submission\AyurIP-v1-state.zip.sha256
```

The first archive contains sanitized source, deployment settings, these seven documents, tests and selected screenshots/evidence. The separate state archive contains the scoped database export and accepted original/generation evidence. Neither contains credentials, private keys, PostgreSQL volumes, model weights, personal research observations or review decisions. Provisioning also requires the separately supplied pinned model/image and a persistent database; the ZIP alone is not a deployed service.

Remaining external actions are explicit: resume deployment with approved budget/data transfer, authenticate Render and a private image registry, provision managed pgvector with restricted ingress/read-only app role, set trusted TLS roots/exact HTTPS origin/snapshot digest, then execute real external smoke/browser/security/log/restart tests. Estimated proposed compute/storage is **US$44.30/month before registry, excess usage and taxes**, not purchased or approved. Remote AI stays disabled until its separate authentication, privacy, inference and quality requirements pass. The 95% answer-quality and five-second answer-latency gates remain unchanged and unpassed by this search-only milestone.

The local TLS test certificate is a short-lived synthetic fixture expiring after seven days; it must not become a production certificate. Loopback test resources are separate from the original service and can be stopped by their explicit names. Keep adequate C: headroom for Docker image storage; the recovered download cache is rebuildable and was not a model or database cache. Historical failures are retained alongside the passing evidence.
