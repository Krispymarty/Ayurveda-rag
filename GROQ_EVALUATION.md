# Isolated Groq comparison

Groq is an evaluation candidate, not an active Passport provider. The existing
Gemini primary and pinned Qwen fallback configuration is unchanged. No public
answer route is enabled by adding a Groq key.

Put the key only in the existing ignored project `.env` or server environment:

```dotenv
GROQ_API_KEY=your_private_key
AYURIP_GROQ_MODEL=openai/gpt-oss-120b
```

Do not put credentials in frontend code, Git, screenshots or model inputs.
The adapter reads only these two Groq settings, uses the official fixed HTTPS
endpoint and sends the key in an authorization header. It disables proxies,
redirects, tools and retries. Ordinary application provider selection does not
accept `groq`; the opt-in evaluation driver constructs the candidate explicitly.

Run offline checks before making real API calls:

```powershell
python -B -m unittest discover -s tests -p test_groq_candidate.py -v
python -B -m unittest discover -s tests -p test_groq_benchmark.py -v
python -m ruff check ipindia_corpus/groq_candidate.py tools/benchmark_evidence_quality.py tests/test_groq_candidate.py tests/test_groq_benchmark.py
```

The local evaluator uses the already-installed optional `tiktoken` package and
`o200k_base` ordinary-text tokenization. Its first use may fetch small tokenizer
BPE data; it does not download model weights. The preflight estimate includes
256 tokens of framing reserve. Exact returned usage must still satisfy the
3,800 input, 200 completion and 4,096 total-token limits. Hidden reasoning counts
toward completion usage. The estimate is not a claim of exact provider chat
tokenization.

The provider wire schema omits only `uniqueItems`, which Groq rejects. The
original schema and model messages remain unchanged, and backend checks still
reject duplicate IDs, unknown IDs, extra fields and altered quotations.

For a new explicitly authorized comparison, choose a new output directory:

```powershell
python -u -B -m tools.benchmark_evidence_quality --bundle data/evidence_quality_20261116/bundle-final.json --out data/groq_20261117/NEW-UNUSED-RUN --providers ollama gemini groq --gemini-interval 15 --groq-interval 20 --warmup --run
```

This command makes real external API calls with frozen public archived evidence
and separately identified synthetic inputs. It never uses ordinary user intakes.
Do not rerun failed questions to replace their scores. The separate synthetic
warmup is unscored. Rate-limit, credential, model-unavailable or binding failures
stop further questions for that provider; unexecuted questions remain visible.
Raw observations and private reports stay excluded under the current Git policy.

Compare provision selection, qualification retention and completeness on the
same completed questions, and report delivery failures separately. Per-call
timings exclude deliberate pacing; they do not measure complete HTTP Passport
latency. A short regression set cannot establish legal accuracy, public readiness
or sustained service availability. Hosted model aliases are not immutable weight
digests. Keep the existing release gates and legal-status restrictions.

The retained November 17 observations are under `data/groq_20261117/`; the report
is `reports/groq_20261117/comparison.md`. These local artifacts are separate from
the tracked evaluation source and sanitized setup instructions.
