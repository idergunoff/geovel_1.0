from func import *
from qt.add_well_dialog import *


def add_well():
    """Добавить новую скважину в БД"""
    Add_Well = QtWidgets.QDialog()
    ui_w = Ui_add_well()
    ui_w.setupUi(Add_Well)
    Add_Well.show()
    Add_Well.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    ui_w.lineEdit_well_alt.setText('0')
    def well_to_db():
        name_well = ui_w.lineEdit_well_name.text()
        x_well = ui_w.lineEdit_well_x.text()
        y_well = ui_w.lineEdit_well_y.text()
        alt_well = ui_w.lineEdit_well_alt.text()
        if name_well != '' and x_well != '' and y_well != '' and alt_well != '':
            new_well = Well(name=name_well, x_coord=float(x_well), y_coord=float(y_well), alt=float(alt_well))
            session.add(new_well)
            session.commit()
            update_list_well()
            Add_Well.close()
            set_info(f'Добавлена новая скважина - "{name_well}".', 'green')

    def cancel_add_well():
        Add_Well.close()

    ui_w.buttonBox.accepted.connect(well_to_db)
    ui_w.buttonBox.rejected.connect(cancel_add_well)
    Add_Well.exec_()


def remove_well():
    name_well = ui.listWidget_well.currentItem().text()
    session.query(Well).filter(Well.id == get_well_id()).delete()
    session.commit()
    update_list_well()
    set_info(f'Удалена скважина - "{name_well}".', 'rgb(188, 160, 3)')
