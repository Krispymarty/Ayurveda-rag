# Project summary

AyurIP is an evidence-search platform for Ayurveda-related intellectual-property research. This release concentrates on traceable archived IP India material, with source-controlled quotations and citations rather than open-ended public legal answers.

The implementation reuses FastAPI, PostgreSQL/pgvector, the existing E5 embedding model, hybrid ranker v2, HTML/CSS/JavaScript UI and immutable-generation design. Cloud preparation adds strict configurable hosts/origin, a public-research-only mode, a manifest-verified CPU container, exact database snapshot/restore tools and a sanitized submission package. The local Qwen Passport workflow is retained behind its existing development boundary.

Source coverage is 307 archived IP India documents. The active generation has 8,880 chunks; a separately preserved rollback has 7,755. Every document's legal status is unknown. No India Code, TKDL, NBA or Gazette certification is claimed. The NBA text candidate and pending human reviews stay quarantined.

Public deployment and protected cloud AI are deferred. The next external release requires owner approval of infrastructure/data transfer, an authenticated provider account, verified managed database provisioning, exact HTTPS origin, and successful real endpoint/browser/security/restart measurements. The 95% answer-quality gate and five-second complete-answer latency target are unchanged and are not certified by research-search tests.
