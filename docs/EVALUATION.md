# Evaluation protocol

## Implemented demonstration

The CLI runs every query against the same corpus and emits per-query JSON.
Each paper is counted once after ranking its highest-scoring chunk. Recall@k
uses the total count of labeled relevant papers as denominator; MRR@k uses
the first relevant paper; nDCG@k uses binary gains and an ideal ranking capped
at k. Unsupported queries have no ranking score, rather than silently receiving 0.

The answer gate requires two distinct non-stopword query terms in a retrieved
chunk. Report supported-answer rate and unsupported-abstention rate separately:
either alone rewards degenerate policies. These measure the extractive gate,
not LLM answer correctness. A relevant passage may still fail to answer a question.

Latency measures one warm retrieval per query, including distinct-document
candidate retrieval, excluding initialization, answer generation and network.
The JSON records p50/p95 but this tiny run is not a throughput benchmark.
Corpora and questions are hashed; package versions and OS are recorded.

## Independent study required for portfolio claims

1. Freeze a rights-cleared corpus of at least 100–300 articles, retaining metadata,
   version, section and page provenance. Verify extraction on equations and tables.
2. Have a person author 150+ realistic questions without viewing retriever outputs.
   Include paraphrases, comparisons, methods, numeric results and unanswerable cases.
3. Split queries by topic/paper family into development and frozen test sets.
   Inspect overlaps; relevant documents remain indexed in both search evaluations.
4. Pool candidates from all retrievers, annotate relevance and supporting passages,
   and adjudicate a second annotator's disagreements. Record annotation guidelines.
5. Tune chunk sizes, retrieval parameters and rejection thresholds on dev only.
   Compare BM25, TF-IDF, dense and hybrid under the same corpus and k.
6. Evaluate the locked test once; report bootstrap confidence intervals over queries,
   per-category failures and latency under repeated, specified workloads.
7. For generation, annotate answer correctness, claim-level evidence support,
   citation precision/coverage and appropriate refusal separately. Keep model,
   prompt, temperature and retrieved context in versioned experiment metadata.

Do not claim that a valid citation ID proves faithfulness. Do not report the small
demo set as a held-out test or tune the heuristic until its known failures vanish.
