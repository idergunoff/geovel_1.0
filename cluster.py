import json

from func import *


def add_clust_analys_from_cls():
    cls_analys = session.query(AnalysisMLP).options(joinedload(AnalysisMLP.parameters)).filter_by(id=get_MLP_id()).first()
    list_param = [i.parameter for i in cls_analys.parameters]
    new_clust_analys = AnalysisCluster(title=f'CLS {cls_analys.title}', parameter=json.dumps(list_param))
    session.add(new_clust_analys)
    session.commit()
    update_list_clust_analys()

def add_clust_analys_from_reg():
    reg_analys = session.query(AnalysisReg).filter_by(id=get_regmod_id()).first()
    list_param = [i.parameter for i in reg_analys.parameters]
    new_clust_analys = AnalysisCluster(title=f'REG {reg_analys.title}', parameter=json.dumps(list_param))
    session.add(new_clust_analys)
    session.commit()
    update_list_clust_analys()


def remove_clust_analys():
    pass


def collect_clust_object():
    pass


def remove_clust_object():
    pass


def get_curr_clust_analys_id():
    return ui.comboBox_clust_set.currentText().split(' id')[-1]


def get_curr_clust_object_id():
    return ui.comboBox_clust_obj.currentText().split(' id')[-1]


def update_list_clust_analys():
    ui.comboBox_clust_set.clear()
    for i in session.query(AnalysisCluster).all():
        ui.comboBox_clust_set.addItem(f'{i.title} id{i.id}')

    update_list_clust_param()


def update_list_clust_param():
    ui.listWidget_clust_param.clear()
    try:
        list_param = json.loads(session.query(AnalysisCluster).filter_by(id=get_curr_clust_analys_id()).first().parameter)
        for i in list_param:
            ui.listWidget_clust_param.addItem(i)
    except AttributeError:
        pass


def update_list_clust_object():
    pass
