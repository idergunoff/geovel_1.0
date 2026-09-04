import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models_db.model import (
    AliasBoundary,
    AliasWellOption,
    Base,
    Boundary,
    CanonicalBoundary,
    CanonicalWellOption,
    Well,
    WellLog,
    WellOptionally,
)
from models_db.model_cluster import AliasWellLog, CanonicalWellLog
from regression_target.service import TargetSettings, parse_numeric_value, resolve_target


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _well(db):
    row = Well(name="W-1", x_coord=0, y_coord=0)
    db.add(row); db.flush()
    return row


@pytest.mark.parametrize(("raw", "status", "value"), [
    ("125,4 м3/сут", "resolved", 125.4),
    ("1 250 м", "resolved", 1250.0),
    ("100 + 25 м", "resolved", 125.0),
    ("100/100", "resolved", 100.0),
    ("100/125", "ambiguous", None),
    ("100-125", "ambiguous", None),
    ("< 0.5", "ambiguous", None),
    ("нет данных", "missing", None),
])
def test_numeric_parser_is_conservative(raw, status, value):
    result = parse_numeric_value(raw)
    assert result.status == status
    assert result.value == value


def test_boundary_resolution_requires_manual_choice_for_alias_duplicates(db):
    well = _well(db)
    canonical = CanonicalBoundary(canonical_name="J1")
    db.add(canonical); db.flush()
    db.add_all([AliasBoundary(alias_name="J1", canonical_id=canonical.id),
                AliasBoundary(alias_name="Ю1", canonical_id=canonical.id)])
    db.add_all([Boundary(well_id=well.id, title="J1", depth=100.1),
                Boundary(well_id=well.id, title="ю1", depth=100.3)])
    db.commit()

    result = resolve_target(db, well.id, TargetSettings("boundary", canonical.id))

    assert result.status == "ambiguous"
    assert [candidate.value for candidate in result.candidates] == [100.1, 100.3]
    result.select(1)
    assert result.status == "resolved"
    assert result.value == 100.3
    assert result.details["manual_override"] is True


def test_boundary_resolution_matches_cyrillic_alias_case_insensitively(db):
    well = _well(db)
    canonical = CanonicalBoundary(canonical_name="Кровля пласта")
    db.add(canonical); db.flush()
    db.add(AliasBoundary(alias_name="кровля j1", canonical_id=canonical.id))
    db.add(Boundary(well_id=well.id, title="  КРОВЛЯ J1  ", depth=123.4))
    db.commit()

    result = resolve_target(db, well.id, TargetSettings("boundary", canonical.id))

    assert result.status == "resolved"
    assert result.value == 123.4


def test_equal_boundary_values_are_resolved_automatically(db):
    well = _well(db)
    canonical = CanonicalBoundary(canonical_name="J1")
    db.add(canonical); db.flush()
    db.add_all([AliasBoundary(alias_name="J1", canonical_id=canonical.id),
                AliasBoundary(alias_name="Ю1", canonical_id=canonical.id)])
    boundaries = [Boundary(well_id=well.id, title="J1", depth=100.0),
                  Boundary(well_id=well.id, title="Ю1", depth=100.0)]
    db.add_all(boundaries); db.commit()

    result = resolve_target(db, well.id, TargetSettings("boundary", canonical.id))

    assert result.status == "resolved"
    assert result.value == 100.0
    assert result.details["auto_selected_equal_values"] is True
    assert [row["source_id"] for row in result.details["equivalent_sources"]] == [
        row.id for row in boundaries
    ]


def test_well_data_resolution_sums_explicit_parts_and_preserves_source(db):
    well = _well(db)
    canonical = CanonicalWellOption(canonical_name="Дебит")
    db.add(canonical); db.flush()
    db.add(AliasWellOption(alias_name="Q", canonical_id=canonical.id)); db.flush()
    source = WellOptionally(well_id=well.id, option="q", value="100 + 25 м3/сут")
    db.add(source); db.commit()

    result = resolve_target(db, well.id, TargetSettings("well_data", canonical.id))

    assert result.status == "resolved"
    assert result.value == 125
    assert result.details["selected"]["details"]["well_option_id"] == source.id


def test_equal_well_data_values_are_resolved_automatically(db):
    well = _well(db)
    canonical = CanonicalWellOption(canonical_name="Дебит")
    db.add(canonical); db.flush()
    db.add(AliasWellOption(alias_name="Q", canonical_id=canonical.id)); db.flush()
    rows = [WellOptionally(well_id=well.id, option="Q", value="125"),
            WellOptionally(well_id=well.id, option="q", value="125,0 м3/сут")]
    db.add_all(rows); db.commit()

    result = resolve_target(db, well.id, TargetSettings("well_data", canonical.id))

    assert result.status == "resolved"
    assert result.value == 125.0
    assert result.details["auto_selected_equal_values"] is True
    assert len(result.details["equivalent_sources"]) == 2


def test_log_resolution_uses_boundary_interval_and_median(db):
    well = _well(db)
    boundary_name = CanonicalBoundary(canonical_name="Top")
    curve_name = CanonicalWellLog(canonical_name="GR")
    db.add_all([boundary_name, curve_name]); db.flush()
    db.add(AliasBoundary(alias_name="top", canonical_id=boundary_name.id))
    db.add(AliasWellLog(alias_name="gamma", canonical_id=curve_name.id))
    db.add(Boundary(well_id=well.id, title="TOP", depth=2.0))
    curve = WellLog(well_id=well.id, curve_name="GAMMA", begin=0, end=5, step=1,
                    curve_data=json.dumps([1, 2, 10, 20, 30, 40]))
    db.add(curve); db.commit()

    settings = TargetSettings("well_log", curve_name.id, boundary_canonical_id=boundary_name.id,
                              interval=2, interval_position="below", aggregation="median")
    result = resolve_target(db, well.id, settings)

    assert result.status == "resolved"
    assert result.value == 20
    details = result.details["selected"]["details"]
    assert details["well_log_id"] == curve.id
    assert details["interval"] == [2.0, 4.0]
    assert details["valid_point_count"] == 3


def test_equal_values_from_multiple_log_curves_are_resolved_automatically(db):
    well = _well(db)
    curve_name = CanonicalWellLog(canonical_name="GR")
    db.add(curve_name); db.flush()
    db.add_all([AliasWellLog(alias_name="gamma", canonical_id=curve_name.id),
                AliasWellLog(alias_name="gr", canonical_id=curve_name.id)])
    curves = [WellLog(well_id=well.id, curve_name="GAMMA", begin=0, end=2, step=1,
                      curve_data=json.dumps([10, 20, 30])),
              WellLog(well_id=well.id, curve_name="GR", begin=0, end=2, step=1,
                      curve_data=json.dumps([10, 20, 30]))]
    db.add_all(curves); db.commit()

    result = resolve_target(
        db, well.id,
        TargetSettings("well_log", curve_name.id, depth_mode="fixed", fixed_depth=0,
                       interval=2, aggregation="mean"),
    )

    assert result.status == "resolved"
    assert result.value == 20.0
    assert result.details["auto_selected_equal_values"] is True
    assert [row["source_id"] for row in result.details["equivalent_sources"]] == [
        row.id for row in curves
    ]


def test_log_boundary_choice_is_recalculated_into_curve_target(db):
    well = _well(db)
    boundary_name = CanonicalBoundary(canonical_name="Top")
    curve_name = CanonicalWellLog(canonical_name="GR")
    db.add_all([boundary_name, curve_name]); db.flush()
    db.add(AliasBoundary(alias_name="top", canonical_id=boundary_name.id))
    db.add(AliasWellLog(alias_name="gamma", canonical_id=curve_name.id))
    db.add_all([Boundary(well_id=well.id, title="TOP", depth=1.0),
                Boundary(well_id=well.id, title="top", depth=3.0)])
    db.add(WellLog(well_id=well.id, curve_name="GAMMA", begin=0, end=5, step=1,
                   curve_data=json.dumps([10, 20, 30, 40, 50, 60])))
    db.commit()

    settings = TargetSettings("well_log", curve_name.id, boundary_canonical_id=boundary_name.id,
                              interval=1, aggregation="mean")
    unresolved = resolve_target(db, well.id, settings)

    assert unresolved.status == "ambiguous"
    assert unresolved.details["pending_selection"] == "boundary_depth"

    result = resolve_target(db, well.id, settings, boundary_candidate=unresolved.candidates[1])

    assert result.status == "resolved"
    assert result.value == 45
    assert result.value != unresolved.candidates[1].value
    assert result.details["manual_override"] is True
    selected_details = result.details["selected"]["details"]
    assert selected_details["depth"] == 3.0
    assert selected_details["selected"]["source_id"] == unresolved.candidates[1].source_id


def test_log_boundary_choices_without_curve_samples_are_filtered_before_selection(db):
    well = _well(db)
    boundary_name = CanonicalBoundary(canonical_name="Top")
    curve_name = CanonicalWellLog(canonical_name="GR")
    db.add_all([boundary_name, curve_name]); db.flush()
    db.add(AliasBoundary(alias_name="top", canonical_id=boundary_name.id))
    db.add(AliasWellLog(alias_name="gamma", canonical_id=curve_name.id))
    unavailable = Boundary(well_id=well.id, title="TOP", depth=100.0)
    available = Boundary(well_id=well.id, title="top", depth=2.0)
    db.add_all([unavailable, available])
    db.add(WellLog(well_id=well.id, curve_name="GAMMA", begin=0, end=5, step=1,
                   curve_data=json.dumps([10, 20, 30, 40, 50, 60])))
    db.commit()

    result = resolve_target(
        db, well.id,
        TargetSettings("well_log", curve_name.id, boundary_canonical_id=boundary_name.id,
                       interval=1, aggregation="mean"),
    )

    assert result.status == "resolved"
    assert result.value == 35
    assert result.details["auto_selected_boundary_id"] == available.id
    assert "выбрана автоматически" in result.message


def test_log_boundary_check_reports_missing_samples_without_prompting_for_boundary(db):
    well = _well(db)
    boundary_name = CanonicalBoundary(canonical_name="Top")
    curve_name = CanonicalWellLog(canonical_name="GR")
    db.add_all([boundary_name, curve_name]); db.flush()
    db.add(AliasBoundary(alias_name="top", canonical_id=boundary_name.id))
    db.add(AliasWellLog(alias_name="gamma", canonical_id=curve_name.id))
    boundaries = [Boundary(well_id=well.id, title="TOP", depth=100.0),
                  Boundary(well_id=well.id, title="top", depth=200.0)]
    db.add_all(boundaries)
    db.add(WellLog(well_id=well.id, curve_name="GAMMA", begin=0, end=5, step=1,
                   curve_data=json.dumps([10, 20, 30, 40, 50, 60])))
    db.commit()

    result = resolve_target(
        db, well.id,
        TargetSettings("well_log", curve_name.id, boundary_canonical_id=boundary_name.id, interval=1),
    )

    assert result.status == "invalid"
    assert result.candidates == []
    assert result.details["checked_boundary_ids"] == [row.id for row in boundaries]
    assert "Ни для одной" in result.message


def test_log_boundary_prompt_contains_only_boundaries_with_curve_samples(db):
    well = _well(db)
    boundary_name = CanonicalBoundary(canonical_name="Top")
    curve_name = CanonicalWellLog(canonical_name="GR")
    db.add_all([boundary_name, curve_name]); db.flush()
    db.add(AliasBoundary(alias_name="top", canonical_id=boundary_name.id))
    db.add(AliasWellLog(alias_name="gamma", canonical_id=curve_name.id))
    available = [Boundary(well_id=well.id, title="TOP", depth=1.0),
                 Boundary(well_id=well.id, title="top", depth=3.0)]
    unavailable = Boundary(well_id=well.id, title="top", depth=100.0)
    db.add_all([*available, unavailable])
    db.add(WellLog(well_id=well.id, curve_name="GAMMA", begin=0, end=5, step=1,
                   curve_data=json.dumps([10, 20, 30, 40, 50, 60])))
    db.commit()

    result = resolve_target(
        db, well.id,
        TargetSettings("well_log", curve_name.id, boundary_canonical_id=boundary_name.id, interval=1),
    )

    assert result.status == "ambiguous"
    assert [row.source_id for row in result.candidates] == [row.id for row in available]
    assert result.details["unavailable_boundary_ids"] == [unavailable.id]
    assert result.details["pending_selection"] == "boundary_depth"


def test_log_ratio_reports_invalid_zero_denominator(db):
    well = _well(db)
    curve_name = CanonicalWellLog(canonical_name="GR")
    db.add(curve_name); db.flush()
    db.add(AliasWellLog(alias_name="GR", canonical_id=curve_name.id))
    db.add(WellLog(well_id=well.id, curve_name="gr", begin=0, end=4, step=1,
                   curve_data=json.dumps([4, 4, 0, 0, 0])))
    db.commit()
    settings = TargetSettings("well_log", curve_name.id, depth_mode="fixed", fixed_depth=2,
                              interval=2, operation="upper_lower_ratio")

    result = resolve_target(db, well.id, settings)

    assert result.status == "invalid"
