"""Utilities for exporting profile data slices."""

from __future__ import annotations

import csv
from typing import Any, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

from func import *

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as SessionType


def _load_json_sequence(raw: Optional[str], field_name: str) -> Sequence:
    """Safely decode JSON data stored in the database."""
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Can't decode JSON from '{field_name}' field: {exc}") from exc
    if not isinstance(data, (list, tuple)):
        raise ValueError(f"Field '{field_name}' is expected to be a JSON array")
    return data


def export_profile_signals(
    research_id: int,
    bottom_limit: int,
    output_dir: str,
    session: Optional["SessionType"] = None,
) -> List[Path]:
    """Export slices of profile signals into individual text files.

    Parameters
    ----------
    research_id:
        Identifier of the research whose profiles should be exported.
    bottom_limit:
        The last signal sample (inclusive, zero based) that should be exported.
        The function will export samples from ``0`` up to ``bottom_limit``.
    output_dir:
        Directory where output files will be created. The directory is created
        if it doesn't exist.
    session:
        Optional existing SQLAlchemy session. When omitted a new session is
        created and closed automatically.

    Returns
    -------
    list[pathlib.Path]
        A list of paths to the files that were created.

    Raises
    ------
    ValueError
        If ``bottom_limit`` is negative or when no profiles/signal data are
        available for the provided ``research_id``.
    """

    if bottom_limit < 0:
        raise ValueError("bottom_limit must be non-negative")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    owns_session = session is None
    if owns_session:
        session = Session()

    try:
        assert session is not None
        profiles: Iterable[Profile] = session.query(Profile).filter(Profile.research_id == research_id).all()
        if not profiles:
            raise ValueError(f"No profiles found for research_id={research_id}")

        trace_records: List[Tuple[Any, Any, Any, Any, Sequence[Any]]] = []
        min_sample_length: Optional[int] = None

        for profile in profiles:
            signals = _load_json_sequence(profile.signal, "signal")
            if not signals:
                continue

            x_wgs = _load_json_sequence(profile.x_wgs, "x_wgs")
            y_wgs = _load_json_sequence(profile.y_wgs, "y_wgs")
            x_pulc = _load_json_sequence(profile.x_pulc, "x_pulc")
            y_pulc = _load_json_sequence(profile.y_pulc, "y_pulc")

            trace_count = min(len(signals), len(x_wgs), len(y_wgs), len(x_pulc), len(y_pulc))
            for trace_index in range(trace_count):
                trace = signals[trace_index]
                if not isinstance(trace, (list, tuple)):
                    continue
                if not trace:
                    continue

                sample_length = len(trace)
                if min_sample_length is None or sample_length < min_sample_length:
                    min_sample_length = sample_length

                trace_records.append(
                    (
                        x_wgs[trace_index],
                        y_wgs[trace_index],
                        x_pulc[trace_index],
                        y_pulc[trace_index],
                        trace,
                    )
                )

        if not trace_records or min_sample_length is None:
            raise ValueError(f"No signal data available for research_id={research_id}")

        last_sample_index = min(bottom_limit, min_sample_length - 1)
        created_files: List[Path] = []

        ui.progressBar.setMaximum(last_sample_index)
        for sample_index in range(last_sample_index + 1):
            file_path = output_path / f"{sample_index}.txt"
            with file_path.open("w", newline="", encoding="utf-8") as file_handle:
                writer = csv.writer(file_handle, delimiter="\t")
                writer.writerow(["x_wgs", "y_wgs", "x_pulc", "y_pulc", f"signal_{sample_index}"])
                for x_wgs_val, y_wgs_val, x_pulc_val, y_pulc_val, trace in trace_records:
                    writer.writerow(
                        [
                            x_wgs_val,
                            y_wgs_val,
                            x_pulc_val,
                            y_pulc_val,
                            trace[sample_index],
                        ]
                    )
            created_files.append(file_path)
            ui.progressBar.setValue(sample_index + 1)

        return created_files
    finally:
        if owns_session and session is not None:
            session.close()
