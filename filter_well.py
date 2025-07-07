from numpy.core.defchararray import center

from func import *
from mlp import update_list_mlp


def open_filter_well_cls():
    FilterWell = QtWidgets.QDialog()
    ui_fw = Ui_FilterWellForm()
    ui_fw.setupUi(FilterWell)
    FilterWell.show()
    FilterWell.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    m_width, m_height = get_width_height_monitor()
    FilterWell.resize(int(m_width/4), int(m_height/1.5))

    for b in get_names_boundary():
        check = QCheckBox(b)
        item = QListWidgetItem()
        ui_fw.listWidget_title_layer.addItem(item)
        ui_fw.listWidget_title_layer.setItemWidget(item, check)


    def set_alt_interval():
        if ui_fw.checkBox_alt.isChecked():
            try:
                list_alt = []
                for alt in session.query(Profile).filter(Profile.research_id == get_research_id()).all():
                    list_alt.extend(json.loads(alt.abs_relief))

                ui_fw.spinBox_alt_from.setValue(int(min(list_alt) - 5))
                ui_fw.spinBox_alt_to.setValue(int(max(list_alt) + 5))
            except TypeError:
                ui_fw.spinBox_alt_from.setValue(0)
                ui_fw.spinBox_alt_to.setValue(250)
                set_info('Рельеф рассчитан не для всех профилей', 'red')


    def push_filter():
        if ui.lineEdit_string.text() == '':
            set_info('Введите название для нового фильтра по анализу', 'red')
            return

        dict_title_depth = get_dict_check_checkbox(ui_fw.listWidget_title_layer)
        list_title_depth = [k for k, v in dict_title_depth.items() if v]

        center_object = get_center_object_coordinates()

        old_mlp = session.query(AnalysisMLP).filter_by(id=get_MLP_id()).first()
        new_mlp = AnalysisMLP(title=ui.lineEdit_string.text())
        session.add(new_mlp)
        session.commit()

        for old_marker in old_mlp.markers:
            new_marker = MarkerMLP(analysis_id=new_mlp.id, title=old_marker.title, color=old_marker.color)
            session.add(new_marker)

            for old_markup in session.query(MarkupMLP).filter_by(analysis_id=get_MLP_id(), marker_id=old_marker.id):
                well = session.query(Well).filter(Well.id == old_markup.well_id).first()
                if ui_fw.checkBox_distance.isChecked():
                    try:
                        distance = calc_distance(well.x_coord, well.y_coord, center_object[0], center_object[1])
                        if distance > ui_fw.spinBox_distance.value():
                            continue
                    except AttributeError:
                        continue

                if ui_fw.checkBox_alt.isChecked():
                    try:
                        if well.alt < ui_fw.spinBox_alt_from.value() or well.alt > ui_fw.spinBox_alt_to.value():
                            continue
                    except AttributeError:
                        continue

                if ui_fw.checkBox_depth.isChecked():
                    try:
                        bound = session.query(Boundary).filter(
                            Boundary.well_id == old_markup.well_id,
                            Boundary.title.in_(list_title_depth)
                        ).first()
                        if bound.depth < ui_fw.spinBox_depth_from.value() or bound.depth > ui_fw.spinBox_depth_to.value():
                            continue
                    except AttributeError:
                        continue


                new_markup = MarkupMLP(
                    analysis_id=new_mlp.id,
                    well_id=old_markup.well_id,
                    profile_id=old_markup.profile_id,
                    formation_id=old_markup.formation_id,
                    marker_id=new_marker.id,
                    list_measure=old_markup.list_measure,
                    type_markup=old_markup.type_markup
                )
                session.add(new_markup)
        session.commit()
        update_list_mlp()
        set_info(f'Скопирован анализ MLP - "{old_mlp.title}"', 'green')

    ui_fw.checkBox_alt.clicked.connect(set_alt_interval)
    ui_fw.pushButton_filter.clicked.connect(push_filter)

    FilterWell.exec_()

def get_names_boundary():
    buondaries = session.query(Boundary).all()
    bound_name = []
    for b in buondaries:
        if b.title not in bound_name:
            bound_name.append(b.title)
    return bound_name

#
# def get_center_object_coordinates():
#     # Получить координаты центра объекта
#
#     idx_research = get_research_id()
#     list_x, list_y = [], []
#     for p in session.query(Profile).filter_by(research_id=idx_research).all():
#         list_x.extend(json.loads(p.x_pulc))
#         list_y.extend(json.loads(p.y_pulc))
#     return [np.mean(list_x), np.mean(list_y)]


