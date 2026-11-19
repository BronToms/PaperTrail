import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str
    paper_id: str
    title: str
    source: str
    text: str
    start: int
    end: int
    kind: str


def load_corpus(path: Path, size: int = 180, overlap: int = 30) -> list[Chunk]:
    """Chunk by whitespace tokens, retaining exact character offsets into source text."""
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("Require size > overlap >= 0")
    chunks, seen = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("id", "title", "source", "text", "kind"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing or empty string field: {key}")
        if row["id"] in seen:
            raise ValueError(f"Duplicate paper ID: {row['id']}")
        seen.add(row["id"])
        tokens = list(re.finditer(r"\S+", row["text"]))
        for i in range(0, len(tokens), size - overlap):
            start, end = tokens[i].start(), tokens[min(i + size, len(tokens)) - 1].end()
            text = row["text"][start:end]
            digest = hashlib.sha256(text.encode()).hexdigest()[:12]
            chunks.append(
                Chunk(
                    f"{row['id']}:{start}:{digest}",
                    row["id"],
                    row["title"],
                    row["source"],
                    text,
                    start,
                    end,
                    row["kind"],
                )
            )
            if i + size >= len(tokens):
                break
    if not chunks:
        raise ValueError("Corpus is empty")
    return chunks


def fingerprint(chunks: list[Chunk]) -> str:
    payload = json.dumps([asdict(c) for c in chunks], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def import_directory(directory: Path, output: Path) -> int:
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            relative = path.relative_to(directory).as_posix()
            content = path.read_text(encoding="utf-8").strip()
            if content:
                rows.append(
                    {
                        "id": relative,
                        "title": path.stem,
                        "source": relative,
                        "text": content,
                        "kind": "user_document",
                    }
                )
    if not rows:
        raise ValueError("No nonempty Markdown/TXT documents found")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return len(rows)
