import json

import numpy as np
import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from models_db import model as m
from models_db import model_cluster, model_profile_features
from terrain_service import (TerrainError, TerrainSurface, bind_grid, delete_grid,
                             ensure_formation_terrain, ensure_profile_terrain,
                             read_grid_file, resolve_grid, display_heights, ensure_terrain_schema)


def nodes(z=100, size=5):
    return [[x, y, z] for x in range(size) for y in range(size)]


@pytest.fixture
def db():
    engine = create_engine('sqlite:///:memory:')
    m.Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as session:
        yield session
    engine.dispose()


def make_object(db):
    obj = m.GeoradarObject(title='Площадь')
    for i in range(2):
        research = m.Research(object=obj)
        profile = m.Profile(title=f'Профиль {i}', research=research,
                            x_pulc='[1, 2, 3]', y_pulc='[1, 2, 3]')
        upper = m.Layers(profile=profile, layer_line='[2, 2, 2]')
        lower = m.Layers(profile=profile, layer_line='[3, 3, 3]')
        m.Formation(profile=profile, layer_up=upper, layer_down=lower, top='[50, 50, 50]')
    db.add(obj)
    db.commit()
    return obj


def grid(db, z=100, local=False, data=None):
    grid = m.CommonGrid(title='Рельеф', type='r', is_local=local,
                        grid_table=json.dumps(nodes(z) if data is None else data))
    db.add(grid)
    db.flush()
    return grid


def test_exact_nodes_boundaries_and_missing_cell():
    data = nodes()
    terrain = TerrainSurface(data)
    assert terrain.covers([[0, 0], [4, 4]])
    assert terrain.sample([[2, 2], [1.5, 1.5]]) == [100, 100]
    assert not terrain.covers([[4.01, 4]])
    masked = TerrainSurface([row for row in data if row[:2] != [2, 2]])
    assert not masked.covers([[1.5, 1.5]])
    # A missing centre must be detected even when both endpoints are valid.
    assert not masked.covers([[0, 0], [2, 2], [4, 4]])
    assert not masked.covers([[0, 0], [4, 4]])
    with pytest.raises(TerrainError):
        masked.sample([[2, 2]])


@pytest.mark.parametrize('missing', [None, -9999, 1.70141e38, float('nan')])
def test_nodata_is_never_interpolated_across(missing):
    data = nodes()
    data[12][2] = missing
    terrain = TerrainSurface(data)
    assert not terrain.covers([[2, 2]])
    assert terrain.covers([[0, 0]])


def test_missing_whole_grid_row_is_not_bridged():
    terrain = TerrainSurface([row for row in nodes() if row[0] != 2])
    assert not terrain.covers([[2, 2]])


def test_binding_recalculates_all_researches_preserving_training_snapshot(db, tmp_path):
    obj = make_object(db)
    old = grid(db)
    local = grid(db, 200, True)
    model = m.TrainedModelReg(list_params='["land"]')
    independent = m.TrainedModelReg(list_params='["X"]')
    db.add_all([model, independent]); db.flush()
    profiles = [research.profiles[0] for research in obj.researches]
    for profile in profiles:
        ensure_formation_terrain(db, profile.formations[0])
        db.add_all([m.ProfileModelPrediction(profile=profile, model_id=model.id, type_model='reg', prediction='[1, 1, 1]'),
                    m.ProfileModelPrediction(profile=profile, model_id=independent.id, type_model='reg', prediction='[2, 2, 2]')])
    snapshot = tmp_path/'training.parquet'
    snapshot.write_bytes(b'existing immutable training snapshot')
    analysis = m.AnalysisReg(data=str(snapshot), up_data=True)
    legacy_analysis = m.AnalysisReg(data='{"land":{"0":100}}', up_data=True)
    db.add_all([analysis, legacy_analysis]); db.commit()
    bind_grid(db, obj.id, local.id)
    db.commit()
    for profile in profiles:
        assert json.loads(profile.abs_relief) == [200, 200, 200]
        assert json.loads(profile.depth_relief) == [0, 0, 0]
        assert json.loads(profile.formations[0].land) == [200, 200, 200]
        assert [p.is_stale for p in profile.predictions] == [False, False]
        assert [p.prediction for p in profile.predictions] == ['[1, 1, 1]', '[2, 2, 2]']
        assert json.loads(profile.abs_relief) == json.loads(profile.formations[0].land)
        assert json.loads(profile.formations[0].speed_cover) == [937.5]*3
    assert snapshot.read_bytes() == b'existing immutable training snapshot'
    assert analysis.up_data is True and analysis.data == str(snapshot)
    assert legacy_analysis.data == '{"land":{"0":100}}' and legacy_analysis.up_data is True


def test_failed_binding_keeps_old_source_and_values(db):
    obj = make_object(db)
    old = grid(db)
    bind_grid(db, obj.id, old.id)
    profile = obj.researches[0].profiles[0]
    before = ensure_profile_terrain(db, profile)
    invalid = grid(db, local=True, data=nodes(size=2))
    with pytest.raises(TerrainError, match='Профиль 1'):
        bind_grid(db, obj.id, invalid.id)
    assert obj.relief_grid_id == old.id
    assert json.loads(profile.abs_relief) == before


def test_local_grid_is_not_used_without_assignment_and_no_fallback(db):
    obj = make_object(db)
    local = grid(db, 200, True)
    profile = obj.researches[0].profiles[0]
    with pytest.raises(TerrainError):
        resolve_grid(db, profile)
    common = grid(db)
    assert resolve_grid(db, profile).id == common.id
    bind_grid(db, obj.id, local.id)
    profile.x_pulc = '[1, 2, 6]'
    with pytest.raises(TerrainError, match='Привязанная'):
        ensure_profile_terrain(db, profile)


def test_unbind_recomputes_from_common_grid_and_delete_used_grid_is_blocked(db):
    obj = make_object(db)
    grid(db)
    local = grid(db, 200, True)
    bind_grid(db, obj.id, local.id)
    with pytest.raises(TerrainError, match='Площадь'):
        delete_grid(db, local.id)
    bind_grid(db, obj.id, None)
    delete_grid(db, local.id)
    for research in obj.researches:
        profile = research.profiles[0]
        assert json.loads(profile.abs_relief) == [100]*3
        assert json.loads(profile.formations[0].land) == [100]*3
        assert json.loads(profile.formations[0].speed_cover) == [312.5]*3


def test_raw_relief_is_not_smoothed_and_display_does_not_modify_it(db):
    obj = make_object(db)
    data = nodes()
    data[12][2] = 500
    grid(db, data=data)
    profile = obj.researches[0].profiles[0]
    assert ensure_profile_terrain(db, profile) == [100, 500, 100]
    raw = profile.abs_relief
    assert display_heights(profile, 5)[1] < 500
    assert profile.abs_relief == raw
    assert display_heights(profile) == [100, 500, 100]
    assert json.loads(profile.depth_relief) == [400, 0, 400]


def test_xyz_import_preserves_first_row_and_nodata(tmp_path):
    path = tmp_path/'local.dat'
    path.write_text('0 0 1\n0 1 2\n1 0 3\n1 1 4\n2 0 -9999\n2 1 5\n')
    data = read_grid_file(path)
    assert data[0] == [0, 0, 1]
    assert len(data) == 6
    path.write_text('X Y Z\n0 0 1\n0 1 2\n1 0 3\n1 1 4\n')
    assert len(read_grid_file(path)) == 4
    path.write_text('X Y Z\n0 0 wrong\n')
    with pytest.raises(TerrainError, match='Строка 2'):
        read_grid_file(path)


def test_legacy_startup_schema_is_additive_and_idempotent():
    engine = create_engine('sqlite:///:memory:')
    with engine.begin() as c:
        for name in ('georadar_object', 'common_grid', 'profile', 'formation', 'profile_model_prediction'):
            c.exec_driver_sql(f'CREATE TABLE {name} (id INTEGER PRIMARY KEY)')
            c.exec_driver_sql(f'INSERT INTO {name} VALUES (1)')
    ensure_terrain_schema(engine)
    ensure_terrain_schema(engine)
    with engine.begin() as c:
        assert c.exec_driver_sql('SELECT id, is_local FROM common_grid').one() == (1, 0)
        assert c.exec_driver_sql('SELECT id, is_stale FROM profile_model_prediction').one() == (1, 0)
        assert inspect(c).get_foreign_keys('georadar_object')[0]['referred_table'] == 'common_grid'
    engine.dispose()


def test_moving_profile_validates_target_object_and_resets_values(db):
    from terrain_service import move_profile_to_research
    source = make_object(db)
    target = make_object(db)
    common = grid(db)
    local = grid(db, 200, True)
    bind_grid(db, target.id, local.id)
    profile = source.researches[0].profiles[0]
    ensure_formation_terrain(db, profile.formations[0])
    move_profile_to_research(db, profile, target.researches[0].id)
    assert profile.abs_relief is None and profile.formations[0].land is None
    assert ensure_profile_terrain(db, profile) == [200]*3
    profile.x_pulc = '[0, 1, 6]'
    move_profile_to_research(db, profile, source.researches[0].id)
    with pytest.raises(TerrainError):
        move_profile_to_research(db, profile, target.researches[0].id)
    assert profile.research.object.id == source.id


def test_alembic_upgrades_legacy_terrain_tables_without_resetting_data(tmp_path):
    from pathlib import Path
    from alembic import command
    from alembic.config import Config
    engine = create_engine(f'sqlite:///{tmp_path / "legacy.sqlite"}')
    with engine.begin() as c:
        for name in ('georadar_object', 'common_grid', 'profile', 'formation', 'profile_model_prediction'):
            c.exec_driver_sql(f'CREATE TABLE {name} (id INTEGER PRIMARY KEY, title VARCHAR)')
            c.exec_driver_sql(f"INSERT INTO {name} VALUES (1, 'unchanged')")
        c.exec_driver_sql('CREATE TABLE analysis_reg (id INTEGER PRIMARY KEY, data TEXT, up_data BOOLEAN)')
        c.exec_driver_sql("INSERT INTO analysis_reg VALUES (1, 'old.parquet', 1)")
    config = Config(str(Path(__file__).resolve().parents[1] / 'alembic.ini'))
    config.set_main_option('script_location', str(Path(__file__).resolve().parents[1] / 'alembic'))
    config.set_main_option('sqlalchemy.url', str(engine.url))
    command.stamp(config, '2f3a4b5c6d7e')
    command.upgrade(config, 'head')
    command.upgrade(config, 'head')
    with engine.begin() as c:
        assert c.exec_driver_sql('SELECT title, is_local FROM common_grid').one() == ('unchanged', 0)
        assert c.exec_driver_sql('SELECT title, relief_grid_id FROM georadar_object').one() == ('unchanged', None)
        assert c.exec_driver_sql('SELECT data, up_data FROM analysis_reg').one() == ('old.parquet', 1)
        assert inspect(c).get_foreign_keys('georadar_object')[0]['referred_table'] == 'common_grid'
    engine.dispose()


def test_object_point_queries_use_binding_and_never_fallback(db):
    from terrain_service import sample_object_terrain
    obj = make_object(db)
    common = grid(db, data=nodes(100, size=10))
    local = grid(db, 200, True)
    bind_grid(db, obj.id, local.id)
    assert sample_object_terrain(db, obj.id, [[1, 1]]) == [200]
    with pytest.raises(TerrainError, match='Привязанная'):
        sample_object_terrain(db, obj.id, [[6, 6]])


def test_monitoring_wellhead_import_uses_the_assigned_grid(db):
    import ast
    from pathlib import Path
    from terrain_service import sample_object_terrain
    obj = make_object(db)
    grid(db)
    local = grid(db, 200, True)
    bind_grid(db, obj.id, local.id)
    well = m.HorizontalWell(title='Well', object_id=obj.id)
    db.add(well); db.flush()
    path = Path(__file__).resolve().parents[1] / 'monitoring.py'
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'save_wellhead_to_db']
    namespace = {'session': db, 'HorizontalWell': m.HorizontalWell, 'sample_object_terrain': sample_object_terrain}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    namespace['save_wellhead_to_db'](1, 1, well.id)
    db.commit()
    assert (well.x_coord, well.y_coord, well.alt) == (1, 1, 200)


def test_deleting_common_source_clears_legacy_results_without_source_key(db):
    obj = make_object(db)
    common = grid(db)
    profile = obj.researches[0].profiles[0]
    profile.abs_relief = '[100,100,100]'
    profile.depth_relief = '[0,0,0]'
    profile.formations[0].land = '[100,100,100]'
    assert profile.relief_source_key is None
    delete_grid(db, common.id)
    assert profile.abs_relief is None and profile.formations[0].land is None


def test_binding_rebuilds_profiles_without_formations_and_reports_missing_cover_inputs(db):
    obj = make_object(db)
    profile = obj.researches[0].profiles[0]
    profile.formations[0].top = None
    empty = m.Profile(title='Без пластов', research=obj.researches[0], x_pulc='[1,2]', y_pulc='[1,2]')
    db.add(empty)
    local = grid(db, 200, True)
    progress = []
    missing = bind_grid(db, obj.id, local.id, progress=lambda done, total, profile: progress.append((done, total)))
    db.commit()
    assert len(missing) == 1 and 'Профиль 0' in missing[0]
    assert json.loads(profile.formations[0].land) == [200]*3
    assert profile.formations[0].speed_cover is None
    assert json.loads(empty.abs_relief) == [200]*2
    assert progress == [(0, 3), (1, 3), (2, 3), (3, 3)]


def test_terrain_changes_preserve_predictions_corrections_and_bindings(db):
    from terrain_service import invalidate_profile
    obj = make_object(db)
    common = grid(db)
    local = grid(db, 200, True)
    profile = obj.researches[0].profiles[0]
    formation = profile.formations[0]
    ensure_formation_terrain(db, formation)
    model = m.TrainedModelReg(list_params='["land", "speed_cover"]')
    db.add(model); db.flush()
    saved = m.ProfileModelPrediction(profile=profile, model_id=model.id, type_model='reg', prediction='[10, 20, 30]')
    correction = m.PredictionCorrect(prediction=saved, correct='[11, 21, 31]')
    binding = m.BindingLayerPrediction(prediction=saved, layer=formation.layer_up)
    index = m.IndexProductivity(prediction=saved, research=profile.research)
    db.add_all([saved, correction, binding, index]); db.commit()
    ids = (saved.id, correction.id, binding.id, index.id)
    bind_grid(db, obj.id, local.id)
    bind_grid(db, obj.id, None)
    invalidate_profile(db, profile)
    ensure_formation_terrain(db, formation)
    delete_grid(db, common.id)
    db.commit()
    assert (saved.id, correction.id, binding.id, index.id) == ids
    assert saved.prediction == '[10, 20, 30]' and saved.is_stale is False
    assert correction.correct == '[11, 21, 31]'
    assert binding.prediction_id == saved.id and index.prediction_id == saved.id


@pytest.mark.parametrize('repair', ['startup', 'migration'])
def test_legacy_prediction_flags_are_cleared_without_changing_results(db, tmp_path, repair):
    from alembic import command
    from alembic.config import Config
    from pathlib import Path
    engine = create_engine(f'sqlite:///{tmp_path / "predictions.sqlite"}')
    m.Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "INSERT INTO profile_model_prediction (id, profile_id, model_id, type_model, prediction, is_stale) "
            "VALUES (1, 7, 42, 'reg', '[10, 20, 30]', 1)")
        connection.exec_driver_sql("INSERT INTO prediction_correct (id, prediction_id, correct) VALUES (1, 1, '[11, 21, 31]')")
        connection.exec_driver_sql("INSERT INTO binding_layer_prediction (id, prediction_id, layer_id) VALUES (1, 1, 23)")
        connection.exec_driver_sql("INSERT INTO analysis_reg (id, data, up_data) VALUES (1, 'old.parquet', 1)")
    if repair == 'startup':
        ensure_terrain_schema(engine)
        ensure_terrain_schema(engine)
    else:
        root = Path(__file__).resolve().parents[1]
        config = Config(str(root / 'alembic.ini'))
        config.set_main_option('script_location', str(root / 'alembic'))
        config.set_main_option('sqlalchemy.url', str(engine.url))
        command.stamp(config, '3a4b5c6d7e8f')
        command.upgrade(config, 'head')
        command.upgrade(config, 'head')
    with engine.begin() as connection:
        assert connection.exec_driver_sql(
            'SELECT id, profile_id, model_id, prediction, is_stale FROM profile_model_prediction').one() == (
                1, 7, 42, '[10, 20, 30]', 0)
        assert connection.exec_driver_sql('SELECT id, prediction_id, correct FROM prediction_correct').one() == (1, 1, '[11, 21, 31]')
        assert connection.exec_driver_sql('SELECT prediction_id, layer_id FROM binding_layer_prediction').one() == (1, 23)
        assert connection.exec_driver_sql('SELECT data, up_data FROM analysis_reg').one() == ('old.parquet', 1)
    engine.dispose()
