import hashlib
import importlib.metadata
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

from papertrail.answers import answer
from papertrail.corpus import fingerprint
from papertrail.retrieval import Retriever


def ranking_metrics(predicted: list[str], relevant: set[str], k: int) -> dict:
    if not relevant:
        raise ValueError("Ranking metrics require at least one relevant document")
    unique = list(dict.fromkeys(predicted))[:k]
    gains = [int(p in relevant) for p in unique]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return {
        "recall": sum(gains) / len(relevant),
        "mrr": next((1 / (i + 1) for i, g in enumerate(gains) if g), 0),
        "ndcg": dcg / ideal,
    }


def evaluate(retriever: Retriever, queries_path: Path, k: int = 3) -> dict:
    queries = [json.loads(line) for line in queries_path.read_text().splitlines() if line.strip()]
    if not queries:
        raise ValueError("Query set is empty")
    known = {c.paper_id for c in retriever.chunks}
    if len({q["id"] for q in queries}) != len(queries):
        raise ValueError("Duplicate query IDs")
    for q in queries:
        if not set(q["relevant"]) <= known:
            raise ValueError(f"Unknown relevance IDs in {q['id']}")
    retriever.search(queries[0]["query"], k)  # warmup excluded from timings
    rows = []
    for q in queries:
        start = time.perf_counter()
        # Retrieve all chunks to produce a true top-k distinct-paper ranking.
        hits = retriever.search(q["query"], len(retriever.chunks))
        elapsed = (time.perf_counter() - start) * 1000
        papers = list(dict.fromkeys(h.chunk.paper_id for h in hits))[:k]
        metrics = ranking_metrics(papers, set(q["relevant"]), k) if q["relevant"] else None
        abstained = answer(retriever, q["query"], k)["abstained"]
        rows.append(
            {
                **q,
                "predicted": papers,
                "metrics": metrics,
                "retrieval_ms": elapsed,
                "abstained": abstained,
            }
        )
    supported = [r for r in rows if r["metrics"] is not None]
    unsupported = [r for r in rows if r["metrics"] is None]
    summary = {
        f"{key}@{k}": float(np.mean([r["metrics"][key] for r in supported])) if supported else None
        for key in ("recall", "mrr", "ndcg")
    }
    summary.update(
        {
            "unsupported_abstention_rate": float(np.mean([r["abstained"] for r in unsupported]))
            if unsupported
            else None,
            "supported_answer_rate": float(np.mean([not r["abstained"] for r in supported]))
            if supported
            else None,
            "retrieval_p50_ms": float(np.percentile([r["retrieval_ms"] for r in rows], 50)),
            "retrieval_p95_ms": float(np.percentile([r["retrieval_ms"] for r in rows], 95)),
        }
    )
    return {
        "scope": "Authored demo queries over authored research notes; not a held-out benchmark",
        "method": retriever.method,
        "model": retriever.model_name if retriever.method in {"dense", "hybrid"} else None,
        "corpus_sha256": fingerprint(retriever.chunks),
        "queries_sha256": hashlib.sha256(queries_path.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                p: importlib.metadata.version(p) for p in ("numpy", "scikit-learn", "fastapi")
            },
        },
        "chunks": len(retriever.chunks),
        "queries": len(rows),
        "summary": summary,
        "per_query": rows,
    }
