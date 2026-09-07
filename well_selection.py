"""Helpers for reliably identifying wells represented by UI list items."""


def well_id_from_text(text: str) -> int | None:
    """Return the numeric id from the final `` id<number>`` list-item suffix."""
    _prefix, separator, raw_id = str(text).rpartition(" id")
    if not separator:
        return None
    try:
        return int(raw_id.strip())
    except (TypeError, ValueError):
        return None


def find_well_row(items, well_id) -> int | None:
    """Find an exact well id match instead of matching id prefixes (1 vs 10)."""
    try:
        expected_id = int(well_id)
    except (TypeError, ValueError):
        return None

    for row, item in enumerate(items):
        if item is not None and well_id_from_text(item.text()) == expected_id:
            return row
    return None
