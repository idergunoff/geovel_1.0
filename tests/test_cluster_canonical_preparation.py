from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cluster_canonical_preparation import (
    normalize_curve_name,
    prepare_canonical_resolution_preview,
)


def test_normalize_curve_name():
    assert normalize_curve_name('  GR  ') == 'gr'
    assert normalize_curve_name(None) == ''


def test_prepare_preview_summary_and_statuses():
    mapping = {
        'gr': 'gamma',
        'nphi': 'neutron',
    }

    def resolver(raw_name):
        key = normalize_curve_name(raw_name)
        return mapping.get(key)

    rows, summary = prepare_canonical_resolution_preview(
        [' GR ', 'NPHI', 'UNKNOWN', None],
        resolver,
    )

    assert summary == {'total': 4, 'resolved': 2, 'unresolved': 2}
    assert rows[0]['status'] == 'resolved'
    assert rows[2]['status'] == 'unresolved'
    assert rows[3]['requested_name'] == ''
