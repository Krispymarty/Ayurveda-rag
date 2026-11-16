# Gemini inference for the private Evidence Passport

The private Passport now defaults to the Gemini API. Public evidence search still
uses the accepted PostgreSQL/pgvector index and local pinned E5 embedding model.
Public `/answer` remains blocked. Historical local-model experiments and the
pinned Qwen adapter remain available for explicit rollback. Private Passport
requests may also use one bounded Qwen fallback attempt for temporary Gemini
availability failures.

## Configuration

Add settings to the existing ignored `E:\Ayurveda Rag\.env`; preserve its current
database settings. Never put an API key in JavaScript, Git, a screenshot or chat.

```dotenv
AYURIP_LLM_PROVIDER=gemini
AYURIP_GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_API_KEY=your_private_key
AYURIP_QWEN_FALLBACK=1
```

Process environment settings take precedence over the project `.env`. The key
may also use the documented `GOOGLE_API_KEY` name. The loader reads only these
documented inference settings and never expands values or changes the environment.

Start the existing search/database services, then the private Passport:

```powershell
docker compose up -d --no-build db api
$env:AYURIP_PASSPORT_DEV='1'
python -B -m ipindia_corpus.passport.api --serve
```

Open `http://127.0.0.1:8003/`. This remains a loopback-only research preview;
internet-facing authentication and public legal-answer release are separate work.

The page identifies Google Gemini processing and requires a separate consent
checkbox before sending a non-confidential intake and retrieved public passages.
The backend independently enforces that acknowledgment. Ordinary request logging
does not retain user intake text, passages or model completions. Content sent to
Google remains subject to the API account's applicable terms and settings.

## Evidence and inference controls

- Existing planner, serial topic retrieval, span selection and subset verifier.
- Maximum three topics and four selected spans per pass. Each provider attempt
  has at most seven model calls; Gemini's external-call allowance is unchanged.
  One fresh local fallback may add up to seven calls (14 total maximum), under
  the same original 90-second acceptance deadline.
- Original prompts, source verification, copied quotations and official citations.
- Application context budget 4,096 tokens: Gemini `countTokens` checks input
  at 3,800 tokens; generation reserves 200 output tokens and a 96-token margin.
  Returned token usage is checked again. No silent evidence truncation.
- Current Gemini 3.5-family API does not accept custom sampling parameters.
  The adapter requests minimal thinking and does not pretend this is temperature
  zero. The explicit 2.5-family option retains temperature zero and no thinking.
- Only the official HTTPS Gemini endpoint; key in a header, no redirects,
  environment proxies, tool execution or retries inside the provider adapter.
- Hosted model IDs and returned `modelVersion` are recorded. There is no invented
  weights digest. A version change latches failure, invalidates readiness and
  removes all evidence from a partly built Passport.
- Token-count/generation calls share the remaining request budget. Late results
  are rejected; Requests socket timeouts are not a hard process-level deadline.

The adapter uses the existing `requests` dependency; no AI framework or model
download is needed. See `ipindia_corpus/inference.py`.

## Automatic private fallback

`AYURIP_QWEN_FALLBACK=1` enables request-level fallback for Gemini timeouts,
rate limits, connection failures and temporary server failures. It is enabled
by default for the connected private Passport, and ignored for explicit local
operation. Set `AYURIP_QWEN_FALLBACK=0` and restart only the private service to
disable it. Injected evaluation providers do not acquire fallback implicitly.

The Gemini attempt has a 30-second acceptance subdeadline when fallback is
configured. Only one fresh Qwen attempt is allowed; it restarts planning,
selection and verification and receives the remainder of the original 90-second
deadline. Partial Gemini output is discarded. Repeated topic searches reuse only
the same request's copied retrieval snapshot; archive checks run again. Both
providers retain the original prompts, context, output caps and pinned bindings.

Authentication/configuration errors, TLS/redirect rejection, safety blocks,
malformed output, source/model binding changes, missing evidence and legitimate
abstention do not trigger fallback. Deterministic post-selection checks reject
missing explicitly requested multiple sections/rules without using benchmark
labels. All legal statuses remain unknown.

After three consecutive transient Gemini failures, new primary attempts pause
for 30 seconds; a rate limit pauses them for 60 seconds. A later successful
primary attempt resets the circuit. These controls are process-local and do not
bypass provider quotas. Consent and intake policy still apply while paused.

The installed Ollama service must be reachable at its validated loopback endpoint
with the exact pinned `qwen3.5:4b` digest. Its receipt is verified on each model
call. Missing or changed local models fail closed; nothing is downloaded. Health
reports fallback as configured and checked on use, rather than claiming it is
continuously available.

Responses identify the actual model, sanitized routing attempts and whether
Gemini processing was attempted before local fallback. A local final result does
not imply the entire request stayed local. Ordinary requests do not retain intake
or completion logs. Late results, including late final archive checks, are
discarded. Synchronous socket/filesystem work is not hard process cancellation;
the five-second complete-answer target remains unproven for fallback.

This remains an experimental private research flow. No new public-answer,
independent legal-accuracy or three-source certification is implied.

## Performance and regression checks

```powershell
python -B -m unittest discover -s tests -p 'test_gemini*.py' -v
python -B -m unittest discover -s tests -p 'test_passport*.py' -v
$env:AYURIP_RUN_BROWSER_TESTS='1'
python -B -m unittest discover -s tests -p test_gemini_readiness.py -v
```

The opt-in benchmark uses only six retained synthetic, legally unreviewed cases,
actual hybrid retrieval, backend archive verification and real loopback HTTP on
isolated port 18003. It never registers an answer endpoint on the public service.

```powershell
python -B -m tools.benchmark_gemini --live --provider both --passes 3 --output data/gemini_20261116/a_new_run_name
```

Choose a new output directory for every run. All failed/interrupted attempts and
separate warmups are retained. Reports append to
`reports/gemini_20261116/performance.md`. Timing and structural/provenance results
do not establish legal relevance or meet the independent 95% quality gate.
The complete-answer p95 target remains five seconds; no target is weakened.

## Explicit local rollback

Stop only the private Passport process, then set the provider and restart it:

```powershell
$env:AYURIP_LLM_PROVIDER='ollama'
$env:AYURIP_PASSPORT_DEV='1'
python -B -m ipindia_corpus.passport.api --serve
```

The installed Ollama server and existing `qwen3.5:4b` model must be available for
that rollback. The model is not downloaded or removed by this change. Clear the
process override or set it to `gemini` to return to the Gemini configuration.
Do not change the corpus generation, ranker, rollback generation, source access
decisions or synchronization state when changing inference providers.

Official API references: [generateContent](https://ai.google.dev/api/generate-content),
[structured outputs](https://ai.google.dev/gemini-api/docs/structured-output),
[Gemini 3.5 Flash-Lite](https://aistudio.google.com/docs/models/gemini-3.5-flash-lite).
