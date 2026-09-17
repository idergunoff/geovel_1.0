"""Repeatable smoke benchmark for the core-description parser."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "core_description"
sys.path.insert(0, str(ROOT))

from core_description.parser import parse_core_document  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")

    subprocess.run(
        [sys.executable, str(FIXTURES / "generate_fixtures.py")],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    sources = sorted((FIXTURES / "generated").glob("*.docx"))
    started = perf_counter()
    interval_count = 0
    for _ in range(args.rounds):
        for source in sources:
            interval_count += len(parse_core_document(source).intervals)
    duration = perf_counter() - started
    document_count = args.rounds * len(sources)
    print(json.dumps({
        "rounds": args.rounds,
        "documents": document_count,
        "intervals": interval_count,
        "seconds": round(duration, 6),
        "documents_per_second": round(document_count / duration, 2),
    }, ensure_ascii=False))
    return int(args.max_seconds is not None and duration > args.max_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
