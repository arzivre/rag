import argparse

from cli.lib.semantic_search import embed_text, verify_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the embeding models loads properly")

    embed_parser = subparsers.add_parser(
        "embed_text", help="Encode text with embedding model"
    )
    embed_parser.add_argument("text", type=str, help="text to be encoded")

    args = parser.parse_args()

    match args.command:
        case "embed_text":
            embed_text(args.text)

        case "verify":
            verify_model()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
