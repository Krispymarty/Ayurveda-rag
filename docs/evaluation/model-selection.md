# Model selection evidence

The configured primary is Groq Qwen 3.8 27B, with Gemini 3.5 Flash-Lite as a transient-availability fallback. This configuration is not independent legal certification.

A 26-case engineering regression measured 24/26 completed, strictly correct outputs for Groq Qwen; 19/19 qualification fragments were retained in completed cases. On nine matched correctly answered cases, median selection+verification model work was 0.832s, versus earlier Gemini 3.425s, Groq GPT-OSS 1.315s and local Qwen 2.360s. These runs occurred in different windows and exclude planning, retrieval, archive verification and full HTTP response time.

A subsequent 42-case multilingual pilot used 14 language/script variants of three repeated tasks, with English source quotations. Gemini delivered 39/42 and Groq Qwen 40/42 correct outputs; all completed cases passed their engineering labels. Hindi was 3/3 on both. Translation/native-speaker and legal-relevance review remain pending. This does not establish translated-answer fluency or universal regional-language support.

Raw private observations, corpus snapshots and review decisions are deliberately excluded. Local manifests preserve inputs, source hashes, failures and model settings. The 95% integrated-quality and five-second full-answer gates have not been newly certified; public answers remain blocked. Models select IDs, while backend code enforces source identity and copies quotations.
