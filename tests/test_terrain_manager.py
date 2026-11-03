import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import json
import pytest
from PyQt5 import QtWidgets
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models_db import model as m
from models_db import model_cluster, model_profile_features
from qt.terrain_manager import TerrainManager


def test_catalog_edit_and_assignment_controls(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    engine = create_engine('sqlite:///:memory:')
    m.Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    obj = m.GeoradarObject(title='Площадь')
    profile = m.Profile(title='Профиль', research=m.Research(object=obj), x_pulc='[0, 1]', y_pulc='[0, 1]')
    formation = m.Formation(profile=profile, top='[50, 50]',
        layer_up=m.Layers(profile=profile, layer_line='[2, 2]'),
        layer_down=m.Layers(profile=profile, layer_line='[3, 3]'))
    common = m.CommonGrid(title='Common', type='r', is_local=False,
                         grid_table=json.dumps([[0,0,50],[0,1,50],[1,0,50],[1,1,50]]))
    grid = m.CommonGrid(title='Local', type='r', is_local=True, comment='Original',
                        grid_table=json.dumps([[0,0,100],[0,1,101],[1,0,102],[1,1,103]]))
    db.add_all([obj, grid, common]); db.commit()
    monkeypatch.setattr(QtWidgets.QMessageBox, 'information', lambda *args: None)
    dialog = TerrainManager(db, object_id=obj.id)
    dialog.refresh(selected_id=grid.id)
    assert dialog.table.rowCount() == 2
    assert dialog.comment.toPlainText() == 'Original'
    assert len(dialog.objects.selectedItems()) == 1
    dialog.title.setText('Точная сетка')
    dialog.comment.setPlainText('Новый комментарий')
    dialog.save()
    assert grid.title == 'Точная сетка' and grid.comment == 'Новый комментарий'
    dialog.assign(False)
    assert obj.relief_grid_id == grid.id
    assert json.loads(formation.land) == [100, 103]
    assert json.loads(formation.speed_cover) == [312.5, 331.25]
    assert 'Точная сетка' in dialog.objects.item(0).text()
    dialog.assign(True)
    assert obj.relief_grid_id is None
    assert json.loads(formation.land) == [50, 50]
    assert json.loads(formation.speed_cover) == [0, 0]
    dialog.close()
    db.close(); engine.dispose()


@pytest.mark.parametrize('cancel', [False, True])
def test_batch_recalculation_failure_restores_all_objects(monkeypatch, cancel):
    from terrain_service import ensure_formation_terrain
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    engine = create_engine('sqlite:///:memory:')
    m.Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    common = m.CommonGrid(title='Common', type='r', is_local=False,
                         grid_table=json.dumps([[0,0,100],[0,1,100],[1,0,100],[1,1,100]]))
    local = m.CommonGrid(title='Local', type='r', is_local=True,
                        grid_table=json.dumps([[0,0,200],[0,1,200],[1,0,200],[1,1,200]]))
    db.add_all([common, local])
    formations = []
    for name in ['A', 'B']:
        profile = m.Profile(title=name, x_pulc='[0, 1]', y_pulc='[0, 1]',
                            research=m.Research(object=m.GeoradarObject(title=name)))
        formation = m.Formation(profile=profile, top='[50, 50]',
            layer_up=m.Layers(profile=profile, layer_line='[2, 2]'),
            layer_down=m.Layers(profile=profile, layer_line='[3, 3]'))
        db.add(formation); db.flush()
        ensure_formation_terrain(db, formation)
        formations.append(formation)
    if cancel:
        monkeypatch.setattr(QtWidgets.QProgressDialog, 'wasCanceled', lambda dialog: dialog.value() >= 1)
    else:
        formations[1].top = 'broken input'
    db.commit()
    messages = []
    monkeypatch.setattr(QtWidgets.QMessageBox, 'warning', lambda *args: messages.append(args[-1]))
    monkeypatch.setattr(QtWidgets.QMessageBox, 'information', lambda *args: messages.append('saved'))
    dialog = TerrainManager(db)
    dialog.refresh(selected_id=local.id)
    dialog.objects.selectAll()
    dialog.assign(False)
    assert messages and ('Пересчёт отменён' if cancel else 'Не удалось пересчитать') in messages[-1]
    assert 'saved' not in messages
    for formation in formations:
        assert formation.profile.research.object.relief_grid_id is None
        assert json.loads(formation.profile.abs_relief) == [100, 100]
        assert json.loads(formation.land) == [100, 100]
        assert json.loads(formation.speed_cover) == [312.5, 312.5]
    dialog.close()
    db.close(); engine.dispose()
