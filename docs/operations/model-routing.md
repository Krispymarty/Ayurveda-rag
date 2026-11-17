# Model routing and privacy

Configure the ignored project `.env`:

```dotenv
AYURIP_LLM_PROVIDER=groq
AYURIP_GROQ_MODEL=qwen/qwen3.8-27b
AYURIP_FALLBACK_PROVIDER=gemini
AYURIP_GEMINI_MODEL=gemini-3.5-flash-lite
GROQ_API_KEY=your_private_key
GEMINI_API_KEY=your_private_key
```

The private Passport requires non-confidential input and server-enforced consent covering both Groq and Google Gemini. The same checks apply to every language; no unvalidated blanket language router is installed.

A temporary primary timeout, rate limit or server/network failure may trigger one fresh Gemini attempt. Partial primary evidence is discarded; selection and verification stay with one provider per attempt. Authentication errors, rejected requests, malformed outputs, safety blocks, legitimate abstention and integrity failures do not trigger fallback. Both attempts share the original 90-second acceptance deadline, with seven calls maximum per attempt. A primary subdeadline reserves time for fallback. Circuit cooldowns prevent repeated primary calls during observed outages.

Both providers use backend-controlled citations, fixed official endpoints, no tool execution, bounded output and response validation. Source text cannot change settings. Groq runtime input has a 48KB transport bound and exact returned-token acceptance checks (3800 input, 200 completion, 4096 total); this is not exact preflight tokenization. No tokenizer or model-weight download occurs in production. Hosted model names are not immutable weight digests.

Set `AYURIP_FALLBACK_PROVIDER=none` to disable fallback. For explicit Gemini-only rollback, set `AYURIP_LLM_PROVIDER=gemini` and `AYURIP_FALLBACK_PROVIDER=none`; restart only the private service. Existing pinned local Ollama support is retained for manual operation with fallback disabled. Preserve archives, model locks, corpus generation and database volume.

This remains an internal research preview. Remote authentication and a public answer release require their separate gates; changing the model does not approve either. API failures must remain distinct from evidence insufficiency. Operational logs must not retain user intake or model completion text.
