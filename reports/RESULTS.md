# Demo experiment results

Measured locally on 2026-11-19. See the JSON reports for corpus/query hashes, package versions and every prediction.

**Scope:** ten authored research notes, twenty answerable queries and six unsupported queries. This is a smoke evaluation with strong authoring bias, not a held-out benchmark. All documents fit in one chunk under the default settings.

| Method | Recall@3 | MRR@3 | nDCG@3 | Unsupported abstention | Supported answer | p50 / p95 retrieval ms |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 1.000 | 1.000 | 1.000 | 0.667 | 1.000 | 0.065 / 0.076 |
| tfidf | 1.000 | 1.000 | 1.000 | 0.667 | 1.000 | 0.418 / 0.577 |

## What these results mean

Both lexical baselines solve this small, vocabulary-overlapping retrieval task. The result does not demonstrate that either method generalizes or that a neural model would improve it. The difference in one-run latency is descriptive only; no concurrent-load study was performed.

The lexical answerability gate incorrectly returns excerpts for these unsupported questions:

- **q24**: What is the exact production deployment cost of the Transformer on modern hardware?
- **q25**: Does QLoRA guarantee memory savings for arbitrary workloads?

These failures distinguish document relevance from answer sufficiency. A chunk sharing terms with the question is not enough to answer it. The baseline remains visible rather than being tuned to these six known examples.

## Validation coverage

BM25, TF-IDF, corpus import, chunk offsets, metric arithmetic, API behavior, excerpt provenance, citation-ID handling and RRF logic are tested. Neural model loading and live Ollama generation have not been run. RRF and the generator adapter have controlled unit tests; those tests do not establish model quality. Docker configuration and GitHub Actions are supplied but have not been executed here.

## Next experiment

Create an independent question set over original article text before adding ranking complexity. Follow [the protocol](../docs/EVALUATION.md); compare dense and hybrid only after labels and splits are frozen.
