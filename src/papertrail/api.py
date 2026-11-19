import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from papertrail.answers import answer
from papertrail.corpus import fingerprint, load_corpus
from papertrail.retrieval import Retriever


class Query(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=3, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("Query cannot be blank")
        return value.strip()


def create_app(corpus: Path | None = None, method: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        chunks = load_corpus(corpus or Path(os.getenv("PAPERTRAIL_CORPUS", "data/papers.jsonl")))
        app.state.retriever = Retriever(chunks, method or os.getenv("PAPERTRAIL_METHOD", "bm25"))
        app.state.fingerprint = fingerprint(chunks)
        yield

    app = FastAPI(
        title="PaperTrail",
        version="0.1.0",
        lifespan=lifespan,
        description="Search ML research notes; return evidence with explicit provenance.",
    )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "chunks": len(app.state.retriever.chunks),
            "method": app.state.retriever.method,
            "corpus_sha256": app.state.fingerprint,
        }

    @app.post("/search")
    def search(q: Query):
        return {"hits": [h.to_dict() for h in app.state.retriever.search(q.query, q.k)]}

    @app.post("/answer")
    def get_answer(q: Query):
        try:
            return answer(
                app.state.retriever,
                q.query,
                q.k,
                model=os.getenv("PAPERTRAIL_OLLAMA_MODEL"),
                ollama_url=os.getenv("PAPERTRAIL_OLLAMA_URL", "http://localhost:11434"),
            )
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise HTTPException(
                502, "Generation backend failed; try /search for evidence."
            ) from exc

    return app


app = create_app()
