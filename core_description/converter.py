"""Safe conversion of legacy binary Word documents to temporary DOCX files."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class DocConversionError(RuntimeError):
    """Raised when a legacy DOC cannot be converted."""


@contextmanager
def docx_source(path: Path) -> Iterator[Path]:
    """Yield *path* as DOCX, converting a DOC in an isolated temporary directory."""

    path = path.expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".docx":
        yield path
        return
    if suffix != ".doc":
        raise DocConversionError(f"Unsupported Word file extension: {path.suffix}")

    executable = shutil.which("libreoffice") or shutil.which("soffice")
    if not executable:
        raise DocConversionError("LibreOffice is required to convert .doc files")

    with tempfile.TemporaryDirectory(prefix="geovel-core-") as temporary:
        # Do not pass the user-selected path directly to LibreOffice.  On Windows
        # its command-line converter is unreliable with long and non-ASCII paths
        # (both are common for Russian core-description archives).  Staging the
        # input also gives the output a deterministic name instead of relying on
        # LibreOffice to reproduce the original basename exactly.
        work_dir = Path(temporary)
        source = work_dir / "source.doc"
        output_dir = work_dir / "converted"
        output_dir.mkdir()
        try:
            shutil.copyfile(path, source)
        except OSError as error:
            raise DocConversionError(f"Cannot prepare DOC for conversion: {path}: {error}") from error
        command = [
            executable,
            "--headless",
            "--convert-to",
            "docx",
            "--outdir",
            str(output_dir),
            str(source),
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise DocConversionError(f"DOC conversion failed: {error}") from error
        converted = output_dir / "source.docx"
        if result.returncode != 0 or not converted.is_file():
            details = (result.stderr or result.stdout).strip()
            raise DocConversionError(
                f"LibreOffice did not produce a DOCX (exit {result.returncode}): {details}"
            )
        yield converted
