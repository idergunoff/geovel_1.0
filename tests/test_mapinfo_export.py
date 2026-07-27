import json
from types import SimpleNamespace

import pytest

from mapinfo_export import (
    ProfileExportError,
    determine_export_zone,
    gauss_kruger_zone_from_easting,
    gauss_kruger_zone_from_longitude,
    prepare_export_profile,
    profile_length,
)


def make_profile(**overrides):
    values = dict(
        id=7,
        research_id=3,
        title="P-7",
        x_pulc=json.dumps([9_500_000.0, 9_500_003.0]),
        y_pulc=json.dumps([6_100_000.0, 6_100_004.0]),
        x_wgs=json.dumps([51.1, 51.1001]),
        y_wgs=json.dumps([55.0, 55.0001]),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_zone_is_derived_from_wgs_and_checked_against_projected_prefix():
    profile = prepare_export_profile(make_profile())
    assert determine_export_zone([profile]) == 9
    assert gauss_kruger_zone_from_longitude(51.1) == 9
    assert gauss_kruger_zone_from_easting(9_500_000) == 9


def test_zone_mismatch_is_rejected():
    profile = prepare_export_profile(
        make_profile(x_pulc=json.dumps([8_500_000.0, 8_500_003.0]))
    )
    with pytest.raises(ProfileExportError, match="не совпадает"):
        determine_export_zone([profile])


def test_profiles_crossing_a_zone_boundary_are_rejected():
    profile = prepare_export_profile(
        make_profile(x_wgs=json.dumps([51.1, 54.1]))
    )
    with pytest.raises(ProfileExportError, match="разных зонах"):
        determine_export_zone([profile])


def test_all_four_coordinate_arrays_must_have_equal_lengths():
    with pytest.raises(ProfileExportError, match="не совпадает количество"):
        prepare_export_profile(make_profile(y_wgs=json.dumps([55.0])))


def test_profile_length_uses_all_polyline_segments():
    assert profile_length([(0, 0), (3, 4), (6, 8)]) == 10
