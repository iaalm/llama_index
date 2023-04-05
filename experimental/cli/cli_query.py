from .configuration import load_index, save_index
from llama_index import SimpleDirectoryReader


def query_cli(args):
    index = load_index()
    print(index.query(args.query))


def register_query_cli(subparsers):
    parser = subparsers.add_parser("query")
    parser.add_argument(
        "query",
        help="Query",
    )

    parser.set_defaults(func=query_cli)

