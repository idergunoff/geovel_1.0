"""JSON command-line interface for the pure core-description parser."""

from __future__ import annotations

import argparse
import json
import sys

from .parser import CoreDescriptionParseError, parse_core_document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", nargs="+", help="DOC or DOCX files")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation")
    args = parser.parse_args()
    results = []
    failed = False
    for document in args.documents:
        try:
            results.append(parse_core_document(document).to_dict())
        except CoreDescriptionParseError as error:
            failed = True
            results.append({"source_path": document, "error": str(error)})
    print(json.dumps(results, ensure_ascii=False, indent=args.indent))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
