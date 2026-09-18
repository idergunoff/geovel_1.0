"""Safe conversion of legacy binary Word documents to temporary DOCX files."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class DocConversionError(RuntimeError):
    """Raised when a legacy DOC cannot be converted."""


def _libreoffice_executable() -> str | None:
    """Find LibreOffice both in PATH and in its usual desktop locations.

    GUI applications started from a shortcut do not necessarily inherit the
    user's shell PATH.  This is especially common on Windows, where a normal
    LibreOffice installation therefore used to look unavailable to GeoVel.
    ``LIBREOFFICE_PATH`` also provides an explicit escape hatch for portable or
    centrally managed installations.
    """

    configured = os.environ.get("LIBREOFFICE_PATH")
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.is_file():
            return str(configured_path)

    for command in ("libreoffice", "soffice"):
        if executable := shutil.which(command):
            return executable

    candidates: list[Path] = []
    for variable in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        if root := os.environ.get(variable):
            candidates.append(Path(root) / "LibreOffice" / "program" / "soffice.exe")
    candidates.extend((
        Path("/Applications/LibreOffice.app/Contents/MacOS/soffice"),
        Path.home() / "Applications/LibreOffice.app/Contents/MacOS/soffice",
    ))
    return next((str(candidate) for candidate in candidates if candidate.is_file()), None)


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

    executable = _libreoffice_executable()
    if not executable:
        raise DocConversionError(
            "LibreOffice is required to convert .doc files. Install LibreOffice "
            "or set LIBREOFFICE_PATH to the soffice executable"
        )

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
