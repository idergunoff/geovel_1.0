"""Batch orchestration for core-description documents.

This module deliberately has no Qt or database dependency.  The GUI can run it in
a worker thread, while tests and other clients can use the same deterministic
file discovery and error-isolation behaviour synchronously.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Callable, Iterable

from .models import ParsedCoreDocument
from .parser import parse_core_document


SUPPORTED_SUFFIXES = frozenset({".doc", ".docx"})


@dataclass
class BatchParseItem:
    """Outcome for one source; parsing failures are data, not batch failures."""

    source_path: str
    status: str
    document: ParsedCoreDocument | None = None
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass
class BatchParseResult:
    items: list[BatchParseItem] = field(default_factory=list)
    cancelled: bool = False

    @property
    def processed(self) -> int:
        return len(self.items)

    @property
    def succeeded(self) -> int:
        return sum(item.status == "ready" for item in self.items)

    @property
    def failed(self) -> int:
        return sum(item.status == "error" for item in self.items)


def discover_documents(paths: Iterable[str | Path], *, recursive: bool = False) -> list[Path]:
    """Return unique Word sources in stable, case-insensitive path order."""

    discovered: dict[str, Path] = {}
    for value in paths:
        path = Path(value).expanduser()
        candidates = path.rglob("*") if path.is_dir() and recursive else path.glob("*") if path.is_dir() else (path,)
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.casefold() in SUPPORTED_SUFFIXES:
                resolved = candidate.resolve()
                discovered[str(resolved)] = resolved
    return sorted(discovered.values(), key=lambda item: str(item).casefold())


def parse_batch(
    paths: Iterable[str | Path],
    *,
    recursive: bool = False,
    cancelled: Callable[[], bool] | None = None,
    progress: Callable[[int, int, BatchParseItem], None] | None = None,
    parser: Callable[[str | Path], ParsedCoreDocument] = parse_core_document,
) -> BatchParseResult:
    """Parse sources independently, stopping cleanly before the next file."""

    sources = discover_documents(paths, recursive=recursive)
    result = BatchParseResult()
    for index, source in enumerate(sources, 1):
        if cancelled and cancelled():
            result.cancelled = True
            break
        started = monotonic()
        try:
            document = parser(source)
            item = BatchParseItem(str(source), "ready", document=document)
        except Exception as error:  # a bad file must not abort the remaining batch
            item = BatchParseItem(str(source), "error", error=str(error) or type(error).__name__)
        item.duration_seconds = monotonic() - started
        result.items.append(item)
        if progress:
            progress(index, len(sources), item)
    return result
