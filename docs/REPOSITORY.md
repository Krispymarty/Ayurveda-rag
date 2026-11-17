# Repository map

| Path | Responsibility |
|---|---|
| `ipindia_corpus/` | Source capture, parsing, immutable generations, retrieval and API |
| `ipindia_corpus/passport/` | Private research planning, extract selection, failover and UI |
| `web/` | Public research frontend |
| `config/` | Safe versioned configuration and sanitized policy examples |
| `tests/` | Offline synthetic tests and explicitly gated integrations |
| `tools/` | Operator and evaluation CLIs; read command help before execution |
| `deploy/` | Deployment templates and migration helpers |
| `docs/operations/` | Current operating instructions |
| `docs/archive/` | Historical documentation, not current readiness claims |
| `submission/` | Submission documentation; generated bundles excluded |

Local-only data, corpus captures, user-supplied originals, benchmark observations, decisions, logs, database volumes and secrets remain outside version control. Original filesystem paths are preserved to avoid breaking citation provenance. Do not move source archives merely to tidy the working directory.

Historical reports referenced by archived documents are local artifacts and may not exist in a fresh clone. New operators should follow the root README and model-routing guide.
