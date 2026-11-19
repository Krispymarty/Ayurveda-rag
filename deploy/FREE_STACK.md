# AyurIP free-tier deployment

The joint public release requires both the search and AI-extract gates. Use an authenticated Vercel preview for staging. The production domain stays unpublished until approval. Use only existing no-cost entitlements, with overages disabled or unavailable.

## Build artifacts

Use a clean export of the candidate commit in a separate directory. Supply the verified active and rollback generation folders and referenced raw IP India originals in that staging root. Prepare the Render context with:

    python -B -m tools.prepare_release --root STAGED_SOURCE --output RENDER_CONTEXT --source-commit FULL_GIT_SHA --embedding-backend remote_e5
    docker build --platform linux/amd64 -f deploy/Dockerfile.render-free -t AYURIP_CANDIDATE RENDER_CONTEXT

Save the emitted snapshot hash and local image identity. Upload an approved image to the selected private Docker Hub Personal repository and deploy to Render by registry digest. Give Render a pull-only credential. Apply deploy/free-stack.env.example using provider secrets. Use an explicit Render host in AYURIP_ALLOWED_HOSTS and the exact Vercel preview origin in AYURIP_PUBLIC_ORIGIN. Register /health as the Render health-check path. Keep one worker.

## Existing Hugging Face Space

Confirm write access and rebuild eligibility for Yashmit/legal-rag on CPU Basic. Preserve the existing revision for rollback. Inspect the public .env locally as the owner. If it contains live credentials, revoke and rotate them. Clean the public history with owner approval and configure only replacement secrets.

Over the verified Space commit, copy deploy/hf_space/ayurip_e5.py and deploy/embedding-files.lock.json into app. Import ayurip_e5.router from app and include it after the FastAPI app is created. Preserve the existing routes and dependency settings. Test model package compatibility before publishing.

At build time copy ayurip_e5.py, embedding-files.lock.json, and deploy/hf_space/preload.py together into a build directory. After the current model dependencies have installed, run:

    python /tmp/ayurip-e5-build/preload.py /opt/ayurip-e5

Ensure the Space runtime user can read the resulting directory. Set one 256-bit random hex AYURIP_E5_API_KEY secret on Hugging Face and Render. Run a real authorized E5 parity check before pointing Render at the Space. The Space accepts JSON text, prepends the E5 query prefix, and returns the exact configured model ID and revision alongside the normalized vector. Existing MiniLM routes remain available.

## Neon and Vercel

Restore the protected export into a new PostgreSQL 16 Neon Free database with pgvector. Use the existing tools.cloud_snapshot restore command and a TLS-verifying owner DSN. Provide the verified evidence root to the restore tool. Grant the runtime role CONNECT, schema USAGE, and SELECT on the five corpus state tables. Use deploy/answer-quota.sql for the dedicated quota role. Verify role separation and table hashes. Save the export and retain the protected rollback binding.

    python -B -m tools.prepare_vercel --output VERCEL_CONTEXT --render-origin https://RENDER_HOST.onrender.com

Set Vercel ORIGIN_SECRET and Render AYURIP_ORIGIN_PROXY_SECRET to the same independent 256-bit hex value. Enable Vercel Authentication with Standard Protection and deploy as preview. Verify unauthenticated access is blocked and the direct Render service rejects non-probe requests. Static files come from the committed frontend. Search, readiness, source status and answers go to the exact Render origin. Provider accounts must use verified free quota. The quota role and provider keys are needed only for the protected answer evaluation until joint approval.

## Verify and promote

Check the candidate on Render-sized resources locally, then from the protected external HTTPS preview. Run the frozen 30-search benchmark for three passes and verify unchanged ordered results, provenance, Recall@5, and warm p95. Check failed dependencies, rollback, request limits, consent, direct access rejection, provider routing, and desktop/mobile Hindi fallback. Keep the retrieval generation and local service intact.

Complete independent review on the existing 64-case packet. Freeze the deployment and score three timed answer passes at the existing limits, pacing over UTC days as needed. Keep every timeout and rejection. Require 95% full-case and answerable correctness, no safety-critical failures, exact quotations and citations, and 5-second complete HTTP p95. Record provider-specific times. The earlier development latency does not meet this gate.

Wait for both gates, artwork permission, and explicit approval of the exact artifact transfers. Do not attach a public production alias while either gate is open. If cloud staging needs spend or cannot be kept private, retain the local candidate. For an approved production activation, bind the final exact origin and recheck external HTTPS and restrictions. Roll back using the protected image digest, the recorded Space revision, the prior Vercel deployment, and the verified database backup. Check readiness and hashes before restoring traffic.
