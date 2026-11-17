# Classroom research demo

The local project is enabled with `AYURIP_DEMO_ANSWERS=1` in the ignored `.env`. The tracked example defaults to `0`. Open http://127.0.0.1:8000/ and use **AI-selected, verified extracts**. Confirm that the question is non-confidential and consent to Groq processing with Google Gemini fallback, then choose **Ask AyurIP**.

The demo uses the top five IP India passages from ranker v2, two-pass ID selection, backend-copied quotations and pre/post-inference archive verification. It uses the configured Groq Qwen primary; only transient availability failures permit a fresh Gemini attempt. Raw model completions and internal diagnostics are not returned. Existing request policy still abstains on personal advice, unsafe requests and claims about current law. India Code, TKDL and ABS coverage remain unavailable.

Suggested question: “What does the archived Patents Act say about traditional knowledge in section 3?” Evidence search remains available through the response-mode selector and does not call a hosted model.

With the existing database volume and local archive provisioned, start Docker Desktop normally, then run `docker compose up -d --no-build db api`. Do not start the sync profiles. No database migration or corpus promotion is needed. Disable the demo by setting the flag to `0`; restart only the API. The existing cloud `public_research` mode remains restricted separately: a classroom demo does not certify a public legal-answer release.

For a clean default-configuration regression run in PowerShell, set `$env:AYURIP_DEMO_ANSWERS='0'` in the test shell before unittest discovery. New demo tests explicitly enable the path under synthetic fixtures; full-corpus tests use the existing local data. Never upload credentials or ordinary invention inputs as test observations.

## Hindi provision questions and known limits

The demo maps a bounded Hindi legal vocabulary and Devanagari digits into retrieval terms, preserving provision references. The original question still reaches both model passes unchanged; original source text and citations are not translated. Unknown Hindi stays on the existing semantic path. English retrieval queries are unchanged. The active corpus and ranker v2 are not replaced.

Request-policy v3 distinguishes English “mere admixture” from romanized possessive usage, and archived treatment-provision questions from personal/clinical instructions. A section number alone does not authorize medical advice. Policy abstention, insufficient evidence and provider failure have different UI explanations.

A retained ten-question HTTP smoke test returned extracts for all five Hindi examples and three of five English examples. All five Hindi requests used Gemini after Groq rate-limit failures; this is not a Qwen-only language comparison. One English request was rejected by Groq and one lacked the requested traditional-knowledge opposition passage in the top five. Those attempts remain failures/abstentions, not fabricated answers or replaced scores. Delivered extracts are not independently certified complete legal answers. Raw observations remain excluded under `data/hindi_demo_repair_20261117/`.
