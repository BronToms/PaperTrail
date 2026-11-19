# Research plan and interview narrative

## Current deliverable

Reproducible lexical retrieval, optional dense/RRF implementations, an in-memory
API, exact excerpts, an Ollama adapter, corpus import, provenance, experiment
reports, regression tests and CI. Neural and generation integrations are not
automatically evidence of model quality: benchmark them on the target hardware.

## Next milestones

| Milestone | Acceptance criterion |
|---|---|
| Corpus | 100–300 licensed ML articles; extraction audit; immutable manifest |
| Labels | 150+ independently authored questions; adjudicated evidence labels |
| Retrieval | Dense and hybrid runs; dev-only tuning; frozen test and uncertainty |
| Reranking | Cross-encoder ablation with candidate depth and latency tradeoff |
| Answers | Human claim-level grounding audit; refusal and injection challenge sets |
| Serving | Measured concurrency, persistent index, limits and structured telemetry |
| Showcase | Recorded demo; English case study; failures and reproducible commands |

## What to explain in an interview

- Why BM25 is a useful baseline for scientific terms, acronyms and identifiers.
- Why fusion uses ranks instead of adding incomparable BM25/cosine scores.
- Why paper-level deduplication matters when papers have many chunks.
- Why retrieval relevance, answerability and answer faithfulness are different.
- Why a small authored corpus gives optimistic ranking results.
- Why the current lexical rejection rule fails on plausible in-domain requests.
- How you would choose a cross-encoder using quality versus latency experiments.

The project should ultimately be described with measured results from that study,
not an invented uplift or a claim that a framework wrapper constitutes research.
