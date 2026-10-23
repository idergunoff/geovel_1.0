"""Safe conversion of legacy binary Word documents to temporary DOCX files."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .docx import DocxExtractionError, extract_docx


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
        Path("/usr/bin/libreoffice"),
        Path("/usr/bin/soffice"),
        Path("/usr/lib/libreoffice/program/soffice"),
        Path("/usr/lib64/libreoffice/program/soffice"),
        Path("/snap/bin/libreoffice"),
        Path.home() / ".local/bin/libreoffice",
        Path.home() / ".local/bin/soffice",
        Path.home() / ".local/lib/libreoffice/program/soffice",
        Path("/Applications/LibreOffice.app/Contents/MacOS/soffice"),
        Path.home() / "Applications/LibreOffice.app/Contents/MacOS/soffice",
    ))
    executable = next((str(candidate) for candidate in candidates if candidate.is_file()), None)
    if executable:
        return executable

    # The archive downloaded from libreoffice.org is commonly unpacked below
    # /opt, with the version embedded in the directory name.
    for root in (Path("/opt"), Path.home() / ".local/opt"):
        for candidate in sorted(root.glob("libreoffice*/program/soffice")):
            if candidate.is_file():
                return str(candidate)
    return None


def _libreoffice_command() -> list[str] | None:
    """Return the command prefix for a native or Flatpak LibreOffice install."""

    if executable := _libreoffice_executable():
        return [executable]

    # IDEs distributed as Flatpak cannot see host executables. Use the host
    # bridge only when Flatpak already permits it; do not change permissions.
    if os.environ.get("FLATPAK_ID") and (spawn := shutil.which("flatpak-spawn")):
        configured = os.environ.get("LIBREOFFICE_PATH")
        candidates = [configured] if configured else ["/usr/bin/libreoffice", "/usr/bin/soffice"]
        for candidate in candidates:
            command = [spawn, "--host", candidate]
            try:
                result = subprocess.run(
                    [*command, "--version"], check=False, capture_output=True,
                    text=True, timeout=10,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            if result.returncode == 0:
                return command

    flatpak = shutil.which("flatpak")
    if not flatpak:
        return None
    application = "org.libreoffice.LibreOffice"
    installations = (
        Path.home() / ".local/share/flatpak/app" / application,
        Path("/var/lib/flatpak/app") / application,
    )
    if any(path.is_dir() for path in installations):
        return [flatpak, "run", application]
    return None


def _powershell_executable() -> str | None:
    """Return Windows PowerShell for Microsoft Word automation, when available."""

    if os.name != "nt":
        return None
    if executable := shutil.which("powershell.exe") or shutil.which("powershell"):
        return executable
    if windows := os.environ.get("SystemRoot"):
        candidate = Path(windows) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def _run_libreoffice(command: list[str], *, timeout: float) -> subprocess.CompletedProcess:
    """Bound the lifetime of the converter, including its Linux child processes."""
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=os.name == "posix",
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except BaseException:
            # Killing just the launcher can leave soffice.bin alive, holding
            # the pipes and temporary profile. Never touch desktop instances.
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                # A host bridge may leave a remote process holding the pipes.
                # Do not wait indefinitely for processes outside our group.
                process.stdout.close()
                process.stderr.close()
                process.wait()
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _run_conversion(command: list[str], converted: Path, converter_name: str) -> None:
    """Run a converter and require it to create the requested DOCX file."""

    try:
        result = _run_libreoffice(command, timeout=120)
    except subprocess.TimeoutExpired as error:
        # Some LibreOffice versions finish writing the document but hang on
        # shutdown. Only accept a complete, readable DOCX after stopping them.
        try:
            extract_docx(converted)
        except DocxExtractionError:
            raise DocConversionError(
                f"{converter_name} conversion timed out after 120 seconds; "
                "no readable DOCX was produced. Open the DOC in LibreOffice, "
                "save it as DOCX and load the saved file"
            ) from error
        return
    except OSError as error:
        raise DocConversionError(f"{converter_name} conversion failed: {error}") from error
    if result.returncode != 0 or not converted.is_file():
        details = (result.stderr or result.stdout).strip()
        raise DocConversionError(
            f"{converter_name} did not produce a DOCX (exit {result.returncode}): {details}"
        )


def _convert_with_word(powershell: str, source: Path, converted: Path) -> None:
    """Convert a staged document using an installed Microsoft Word on Windows."""

    # Paths are supplied via the process environment rather than interpolated
    # into PowerShell code, so quotes and other path characters remain harmless.
    script = """
$ErrorActionPreference = 'Stop'
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $document = $word.Documents.Open($env:GEOVEL_DOC_SOURCE, $false, $true)
    $document.SaveAs2($env:GEOVEL_DOCX_TARGET, 16)
} finally {
    if ($null -ne $document) { $document.Close($false) }
    if ($null -ne $word) { $word.Quit() }
}
""".strip()
    environment = os.environ.copy()
    environment["GEOVEL_DOC_SOURCE"] = str(source)
    environment["GEOVEL_DOCX_TARGET"] = str(converted)
    command = [powershell, "-NoProfile", "-NonInteractive", "-Command", script]
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=120,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise DocConversionError(f"Microsoft Word conversion failed: {error}") from error
    if result.returncode != 0 or not converted.is_file():
        details = (result.stderr or result.stdout).strip()
        raise DocConversionError(
            f"Microsoft Word did not produce a DOCX (exit {result.returncode}): {details}"
        )


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

    libreoffice_command = _libreoffice_command()
    powershell = _powershell_executable() if not libreoffice_command else None
    if not libreoffice_command and not powershell:
        raise DocConversionError(
            "Converting .doc files requires LibreOffice, or Microsoft Word on Windows. "
            "Install one of them or convert the file to .docx"
        )

    temporary_root = None
    if libreoffice_command and libreoffice_command[1:2] == ["--host"]:
        # Flatpak's /tmp is private. Its per-app cache has the same absolute
        # path on the host, allowing both processes to share staged files.
        application = os.environ["FLATPAK_ID"]
        if application in {".", ".."} or "/" in application or "\\" in application:
            raise DocConversionError("Invalid FLATPAK_ID for shared conversion directory")
        temporary_root = Path.home() / ".var" / "app" / application / "cache"
        try:
            temporary_root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise DocConversionError(f"Cannot prepare shared LibreOffice directory: {error}") from error

    with tempfile.TemporaryDirectory(prefix="geovel-core-", dir=temporary_root) as temporary:
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
        converted = output_dir / "source.docx"
        if libreoffice_command:
            command_prefix = libreoffice_command
            if len(command_prefix) >= 3 and command_prefix[1] == "run":
                # Flatpak has a private /tmp. Explicitly expose only this
                # short-lived staging directory, not the user's source path.
                command_prefix = [
                    *command_prefix[:2], f"--filesystem={work_dir}", *command_prefix[2:],
                ]
            command = [
                *command_prefix,
                # A private profile prevents forwarding this request to an
                # already running desktop instance (or another import).
                f"-env:UserInstallation={(work_dir / 'profile').as_uri()}",
                "--headless",
                "--norestore",
                "--nodefault",
                "--nofirststartwizard",
                "--convert-to",
                "docx",
                "--outdir",
                str(output_dir),
                str(source),
            ]
            _run_conversion(command, converted, "LibreOffice")
        else:
            assert powershell is not None
            _convert_with_word(powershell, source, converted)
        yield converted
