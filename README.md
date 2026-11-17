# AyurIP

Ayurveda intellectual-property research through traceable government-source evidence.

## Current configuration

- **Primary:** Groq `qwen/qwen3.8-27b`.
- **Fallback:** Google Gemini `gemini-3.5-flash-lite`, for transient availability failures.
- **Retrieval:** PostgreSQL/pgvector, full-text search and pinned multilingual E5 embeddings, ranker v2.
- **Interface:** existing responsive HTML/CSS/JavaScript and FastAPI.
- **Evidence:** verified archived IP India corpus. India Code, public TKDL and NBA material are not certified for admission.

The model selects evidence IDs; backend code copies quotations and constructs citations. Legal status stays unknown. Public `/answer` remains blocked. The Evidence Passport remains a loopback-only research preview, not a public legal-advice system. Deployment is not claimed.

## Setup

Install Python dependencies with `python -m pip install -r requirements-service.txt`. Copy `.env.example` to `.env` and configure your private database password, Groq key and Gemini key. Never commit `.env`.

This repository deliberately excludes corpus originals, database volumes, model weights, user observations and review decisions. **A fresh clone cannot serve search until a separately obtained verified evidence snapshot, embedding cache and database index have been provisioned.** Preserve existing archives and the existing PostgreSQL volume when working with the local project.

With those prerequisites available:

```powershell
docker compose up -d --no-build db api
$env:AYURIP_PASSPORT_DEV='1'
python -B -m ipindia_corpus.passport.api --serve
```

Search: http://127.0.0.1:8000/ — private Passport: http://127.0.0.1:8003/.
Do not enable the optional scheduler/candidate profiles without resolved access reviews. Do not delete or recreate the existing database volume.

## Documentation

- [Model routing, consent and rollback](docs/operations/model-routing.md)
- [Repository map](docs/REPOSITORY.md)
- [Evaluation instructions](GROQ_EVALUATION.md)
- [Earlier operational workflows](docs/archive/README-before-provider-finalization.md) — historical, not current release claims.
- [Deployment configuration](deploy/) — configuration only; hosting requires separate provisioning and approval.

Raw local reports are excluded. Recorded model comparisons are engineering regression observations, not independently reviewed legal accuracy. Native-speaker review, three-source certification, integrated quality gates and the five-second complete-answer target remain outstanding.

## Verification

```powershell
python -B -m unittest discover -s tests -v
python -m ruff check ipindia_corpus tools tests
```

Some browser, database and model checks are explicitly opt-in. Unit fixtures are synthetic and do not establish legal relevance. Never use ordinary confidential invention descriptions in benchmarks.
