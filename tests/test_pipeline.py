import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from papertrail.answers import answer
from papertrail.api import create_app
from papertrail.corpus import fingerprint, import_directory, load_corpus
from papertrail.evaluation import evaluate, ranking_metrics
from papertrail.retrieval import Retriever

CORPUS = Path(__file__).resolve().parents[1] / "data/papers.jsonl"


@pytest.fixture
def retriever():
    return Retriever(load_corpus(CORPUS))


def test_chunk_offsets_and_stability():
    rows = {r["id"]: r for r in map(json.loads, CORPUS.read_text().splitlines())}
    chunks = load_corpus(CORPUS, size=20, overlap=5)
    assert len(chunks) > len(rows)
    assert len({c.id for c in chunks}) == len(chunks)
    for chunk in chunks:
        assert rows[chunk.paper_id]["text"][chunk.start : chunk.end] == chunk.text
    assert fingerprint(chunks) == fingerprint(load_corpus(CORPUS, 20, 5))
    assert fingerprint(chunks) != fingerprint(load_corpus(CORPUS))


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_chunk_config(size, overlap):
    with pytest.raises(ValueError):
        load_corpus(CORPUS, size, overlap)


def test_empty_and_duplicate_corpus(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_corpus(path)
    line = CORPUS.read_text().splitlines()[0]
    path.write_text(line + "\n" + line)
    with pytest.raises(ValueError, match="Duplicate"):
        load_corpus(path)


@pytest.mark.parametrize("method", ["bm25", "tfidf"])
def test_rank_and_out_of_vocabulary(method):
    r = Retriever(load_corpus(CORPUS), method)
    assert r.search("double quantization paged optimizers")[0].chunk.paper_id == "qlora"
    assert r.search("xyzzzzzzzzz") == []
    assert r.search("the and of") == []
    with pytest.raises(ValueError):
        r.search("retrieval", 0)


def test_metrics_deduplicate_and_use_total_relevant():
    metrics = ranking_metrics(["a", "a", "b", "c"], {"b", "c"}, 2)
    assert metrics["recall"] == 0.5
    assert metrics["mrr"] == 0.5
    assert 0 < metrics["ndcg"] < 1
    assert ranking_metrics([], {"a"}, 3)["ndcg"] == 0


def test_excerpts_are_exact_and_cited(retriever):
    result = answer(retriever, "low rank adaptation frozen weights")
    assert not result["abstained"]
    assert result["citation_validation"] == "exact_source_excerpts"
    for source in result["sources"]:
        assert f"[{source['citation']}] {source['text']}" in result["answer"]
    assert answer(retriever, "mushroom risotto recipe")["abstained"]


@pytest.mark.parametrize(
    "text,abstained",
    [
        ("Uses adapters [S1].", False),
        ("An invented claim [S99].", True),
        ("Uncited answer", True),
        ("INSUFFICIENT_EVIDENCE", True),
    ],
)
def test_generator_citation_validation(monkeypatch, retriever, text, abstained):
    def post(self, url, json):
        assert json["stream"] is False
        assert json["messages"][0]["role"] == "system"
        return httpx.Response(
            200, json={"message": {"content": text}}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(httpx.Client, "post", post)
    result = answer(retriever, "low rank frozen weights", model="test-model")
    assert result["abstained"] is abstained


def test_api_validation_and_search():
    with TestClient(create_app(CORPUS)) as client:
        assert client.get("/health").json()["chunks"] == 10
        assert client.post("/search", json={"query": "  "}).status_code == 422
        assert client.post("/search", json={"query": "test", "k": 0}).status_code == 422
        result = client.post("/search", json={"query": "late interaction"})
        assert result.json()["hits"][0]["paper_id"] == "colbert"
        assert client.post("/answer", json={"query": "risotto"}).json()["abstained"]


def test_import_preserves_unicode_and_nested_identity(tmp_path):
    directory = tmp_path / "inputs"
    (directory / "nested").mkdir(parents=True)
    (directory / "paper.md").write_text("# Модель\nОписание адаптеров.")
    (directory / "nested/paper.md").write_text("Different paper.")
    output = tmp_path / "corpus.jsonl"
    assert import_directory(directory, output) == 2
    chunks = load_corpus(output)
    assert len({c.paper_id for c in chunks}) == 2
    assert any("Модель" in c.text for c in chunks)


def test_evaluation_rejects_unknown_labels(retriever, tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text(json.dumps({"id": "q", "query": "test", "relevant": ["missing"]}))
    with pytest.raises(ValueError, match="Unknown"):
        evaluate(retriever, path)


def test_hybrid_rrf_without_model_download(retriever):
    class Encoder:
        def encode_query(self, texts, normalize_embeddings):
            return np.array([[1.0, 0.0]])

    retriever.method = "hybrid"
    retriever.encoder = Encoder()
    retriever.embeddings = np.array([[1.0, 0.0]] + [[0.0, 1.0]] * 9)
    # Only dense has a positive hit: zero lexical scores must not add arbitrary votes.
    result = retriever.search("xyzzzzzzzzz")
    assert len(result) == 1
    assert result[0].score == pytest.approx(1 / 61)
