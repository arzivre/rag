import argparse

from cli.lib.semantic_search import (
    chunk_text,
    embed_query_text,
    embed_text,
    search,
    verify_embeddings,
    verify_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    chunk_parser = subparsers.add_parser(
        "chunk", help="Chunk a document"
    )
    chunk_parser.add_argument("text", type=str, help="Document to be chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="Number of words in each fixed size chunk"
    )

    search_parser = subparsers.add_parser("search", help="Search for relevan movies")
    search_parser.add_argument("query", type=str, help="User query to search based on")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of result return"
    )

    embed_query_parser = subparsers.add_parser(
        "embed_query", help="Encode query with embedding model"
    )
    embed_query_parser.add_argument("query", type=str, help="User query to be encoded")

    subparsers.add_parser(
        "verify_embeddings", help="Verify the embeding models loads properly"
    )

    embed_parser = subparsers.add_parser(
        "embed_text", help="Encode text with embedding model"
    )
    embed_parser.add_argument("text", type=str, help="text to be encoded")

    subparsers.add_parser("verify", help="Verify the embeding models loads properly")

    args = parser.parse_args()

    match args.command:
        case "chunk":
            chunk_text(args.text, args.chunk_size)

        case "search":
            search(args.query, args.limit)

        case "embed_query":
            embed_query_text(args.query)

        case "verify_embeddings":
            verify_embeddings()

        case "embed_text":
            embed_text(args.text)

        case "verify":
            verify_model()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
