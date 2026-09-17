"""Tests for stage 6 batch discovery and isolated processing."""

from pathlib import Path

from core_description.batch import discover_documents, parse_batch
from core_description.models import ParsedCoreDocument


def _document(path: str | Path) -> ParsedCoreDocument:
    return ParsedCoreDocument(str(path), "docx", "a" * 64, "1", "1", None, None, None, None)


def test_discovery_is_recursive_deduplicated_and_stable(tmp_path):
    nested = tmp_path / "nested"; nested.mkdir()
    (tmp_path / "B.docx").write_bytes(b"")
    (tmp_path / "a.DOC").write_bytes(b"")
    (tmp_path / "ignore.txt").write_bytes(b"")
    (nested / "c.docx").write_bytes(b"")

    shallow = discover_documents([tmp_path, tmp_path / "B.docx"])
    recursive = discover_documents([tmp_path], recursive=True)

    assert [item.name for item in shallow] == ["a.DOC", "B.docx"]
    assert [item.name for item in recursive] == ["a.DOC", "B.docx", "c.docx"]


def test_bad_document_does_not_stop_batch(tmp_path):
    for name in ("1.docx", "2.docx", "3.docx"):
        (tmp_path / name).write_bytes(b"")

    def parser(path):
        if Path(path).name == "2.docx":
            raise ValueError("damaged")
        return _document(path)

    progress = []
    result = parse_batch([tmp_path], parser=parser,
                         progress=lambda done, total, item: progress.append((done, total, item.status)))

    assert [item.status for item in result.items] == ["ready", "error", "ready"]
    assert result.succeeded == 2 and result.failed == 1
    assert progress[-1] == (3, 3, "ready")


def test_cancel_stops_before_next_document(tmp_path):
    for name in ("1.docx", "2.docx"):
        (tmp_path / name).write_bytes(b"")
    parsed = 0

    def parser(path):
        nonlocal parsed
        parsed += 1
        return _document(path)

    result = parse_batch([tmp_path], parser=parser, cancelled=lambda: parsed == 1)

    assert result.cancelled is True
    assert result.processed == 1
