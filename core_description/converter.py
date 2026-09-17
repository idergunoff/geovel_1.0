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
        output_dir = Path(temporary)
        command = [
            executable,
            "--headless",
            "--convert-to",
            "docx",
            "--outdir",
            str(output_dir),
            str(path),
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
        converted = output_dir / f"{path.stem}.docx"
        if result.returncode != 0 or not converted.is_file():
            details = (result.stderr or result.stdout).strip()
            raise DocConversionError(
                f"LibreOffice did not produce a DOCX (exit {result.returncode}): {details}"
            )
        yield converted
