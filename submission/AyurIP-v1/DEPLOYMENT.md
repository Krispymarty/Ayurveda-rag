# Selected deployment: Render image-backed service and managed Postgres

Deployment is deferred by the owner's 14 November instruction. No cloud service, paid resource, registry upload, remote push or public URL was created. Do not treat this guide as spending/transfer approval.

The single selected architecture is an always-on Render web service serving both FastAPI and the existing frontend, plus managed PostgreSQL 16 with pgvector in the same region. The app carries a verified read-only evidence/model snapshot; it does not require a writable archive disk. Use private image storage and an immutable image digest. [Render supports private prebuilt images](https://render.com/docs/deploying-an-image), [managed pgvector](https://render.com/docs/postgresql-extensions) and [health checks](https://render.com/docs/health-checks).

## Provisioning assets

- Source/configuration: this submission's `source/` and `deploy/` directories.
- Prepared local build context: `E:\Ayurveda Rag\data\cloud_release_20261114\runtime-v6`.
- Read-only database export: `E:\Ayurveda Rag\data\cloud_release_20261114\database`.
- Separate state archive: `E:\Ayurveda Rag\submission\AyurIP-v1-state.zip` (no model weights).
- Model: `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`; exact observed file hashes in `deploy/embedding-files.lock.json`. The production wheel is Torch `2.14.1+cpu`, matching the baseline Torch version without CUDA libraries. Existing Qwen weights are not shipped.

The state archive supplies `data/raw`, both exact generation folders, `data/model.lock.json`, and the database export. It does not include unrelated history, user observations or private decisions. Reconstruct a new Linux staging root from sanitized source and this state. Supply the existing pinned HF cache separately; do not infer model bytes from cache blob filenames. The packager dereferences only contained, explicitly hash-bound model files. It refuses existing output directories and does not alter the original workspace.

In the current Windows workspace, model links are Linux cache links. Use the existing Linux API container for snapshot preparation, as verified:

```powershell
docker compose exec -T api python -B -m tools.prepare_release --output /app/data/cloud_release_20261114/runtime-new
docker build --tag ayurip-release:NEW_TAG --file deploy/Dockerfile data/cloud_release_20261114/runtime-new
```

On a fully provisioned Linux checkout, run `python -B -m tools.prepare_release --output /separate/new/context`. Never use an unchecked whole-workspace build context. The root development Dockerfile is retained for local compatibility; the production build uses `deploy/Dockerfile`, pinned CPU dependencies/base digest, a nonroot user and read-only runtime settings. Preserve the emitted release-manifest SHA-256.

## Database migration

Read credentials through environment variables/provider secret management. Never place real DSNs in Git, command history or documentation. Remote transfers require TLS `verify-full`; configure a trusted CA file or supported `sslrootcert=system`.

```text
python -B -m tools.cloud_snapshot export --output NEW_EMPTY_EXPORT_DIRECTORY
python -B -m tools.cloud_snapshot verify --snapshot EXPORT_DIRECTORY
python -B -m tools.cloud_snapshot restore --snapshot EXPORT_DIRECTORY --allow-empty-target
```

Export uses `SOURCE_DATABASE_URL`; restore uses `TARGET_DATABASE_URL`. `--root` can identify the provisioned evidence root. Restore must target a new empty database, not the source or any existing application's tables. It creates pgvector/schema/indexes and imports the exact protected active/rollback releases plus needed embeddings. Every table digest, vector dimension, model revision, original hash and passage binding must verify before commit. No destructive reset is part of this procedure.

Create a separate login for the application. Grant database CONNECT, public-schema USAGE and SELECT on `corpus_documents`, `corpus_chunks`, `index_generations`, `embedding_cache`, `app_state`; no write/DDL privileges. Keep migration-owner credentials out of the web service. Restrict database ingress to approved application/operator clients. The isolated local release uses such a read-only role over verified TLS.

## Provider settings and activation

Use `deploy/render-settings.json` as operator input, not an already-submitted API request. The initial proposed sizes are one `1c-2g` web instance and one `0.5c-1g` database, with 1 GB storage and autoscaling off. [Official compute IDs](https://render.com/docs/compute-plans) and [storage options](https://render.com/docs/postgresql-creating-connecting) were checked on 14 November 2026. Configure the provider's exact real HTTPS domain in `AYURIP_PUBLIC_ORIGIN`, the emitted digest in `AYURIP_SNAPSHOT_SHA256`, and the read-only TLS DSN in provider secrets. Follow `deploy/cloud.env.example`. No wildcard host/origin allowance is supported.

The image honors provider `PORT`, binds one app worker and disables access logging. Its health probe sends the configured public Host even on a nondefault port. Set provider health path `/health`; `/live` alone must not admit traffic. Keep edge HTTP-to-HTTPS redirect enabled and proxy-header trust disabled in the application. Keep the public AI/Passport and all synchronization/candidate services disabled. Expose neither database nor Ollama management to the internet.

After explicit approval, upload only the verified image/export/evidence to the approved private services. Verify the deployed image digest and snapshot hash. Run:

```text
python -B -m tools.release_smoke --base-url https://REAL_DOMAIN --output NEW_CLOUD_RESULT.json
```

Then run real desktop/mobile browser searches, unauthorized-route checks, certificate/redirect checks, provider-log/resource inspection and restart/redeploy persistence. Preserve failed attempts. This local package does not establish those external results or the five-second AI target.

## Rollback and costs

Retain the previous OCI image digest, source snapshot, database backup and three release-state values before changing a deployed service. Source/data updates require a separately verified candidate and explicit promotion. For a retrieval rollback, an authorized migration operator may restore the preserved generation/ranker binding `20261014T190100599320Z`/`v1`; restart and verify its manifest/model/archives before routing traffic. A binding change invalidates the running app. Restore `20261014T204940358650Z`/`v2` and restart to return. Never edit or drop the original local database/volume. See the local rollback evidence in `TESTING.md`; a managed-cloud rollback still needs external execution.

Proposed recurring cost is approximately **US$44.30/month**: $25 web, $19 database, $0.30 for 1 GB storage, before registry, excess bandwidth, taxes or plan changes. No cloud inference is included. This is an unpurchased estimate based on [Render pricing](https://render.com/pricing), not a bill or approved budget. Confirm all charges in the provider console before provisioning.
