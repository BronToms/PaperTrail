import argparse
import json
from pathlib import Path

from papertrail.answers import answer
from papertrail.corpus import import_directory, load_corpus
from papertrail.evaluation import evaluate
from papertrail.retrieval import METHODS, Retriever


def main():
    parser = argparse.ArgumentParser(description="PaperTrail research retrieval")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("search", "answer", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--corpus", type=Path, default=Path("data/papers.jsonl"))
        sub.add_argument("--method", choices=METHODS, default="bm25")
        sub.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
        sub.add_argument("--chunk-size", type=int, default=180)
        sub.add_argument("--overlap", type=int, default=30)
        sub.add_argument("-k", type=int, default=3)
        if command == "evaluate":
            sub.add_argument("--queries", type=Path, default=Path("data/queries.jsonl"))
            sub.add_argument("--output", type=Path, required=True)
        else:
            sub.add_argument("query")
        if command == "answer":
            sub.add_argument("--ollama-model")
    sub = commands.add_parser("import")
    sub.add_argument("directory", type=Path)
    sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "import":
        print(json.dumps({"documents": import_directory(args.directory, args.output)}))
        return
    chunks = load_corpus(args.corpus, args.chunk_size, args.overlap)
    retriever = Retriever(chunks, args.method, args.model)
    if args.command == "search":
        result = [h.to_dict() for h in retriever.search(args.query, args.k)]
    elif args.command == "answer":
        result = answer(retriever, args.query, args.k, model=args.ollama_model)
    else:
        result = evaluate(retriever, args.queries, args.k)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        result = result["summary"]
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
