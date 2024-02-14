from func import *

def model_lineup():
    lineupModel = QtWidgets.QDialog()
    ui_l = Ui_Dialog_model_lineup()
    ui_l.setupUi(lineupModel)
    lineupModel.show()
    lineupModel.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    def update_lineup_list():
        ui_l.listWidget_lineup.clear()
        for i in session.query(LineupTrain).all():
            try:
                params = list(json.loads(i.list_param))
                item_text = f'{i.type_ml} {i.text_model.split(":")[0]} params: {len(params)}'
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, i.id)
                ui_l.listWidget_lineup.addItem(item)
            except AttributeError:
                pass

    def update_lineup_info():
        ui_l.plainTextEdit_info.clear()
        model = session.query(LineupTrain).filter_by(id=ui_l.listWidget_lineup.currentItem().data(Qt.UserRole)).first()
        ui_l.plainTextEdit_info.insertPlainText(model.text_model)

    def remove_lineup_model():
        model = session.query(LineupTrain).filter_by(id=ui_l.listWidget_lineup.currentItem().data(Qt.UserRole)).first()
        session.delete(model)
        session.commit()
        update_lineup_list()
        set_info(f'Модель {model.text_model.split(":")[0]} удалена из списка', 'green')


    def clear_lineup_model():
        for model in session.query(LineupTrain).all():
            session.delete(model)
            session.commit()
        update_lineup_list()
        ui_l.plainTextEdit_info.clear()
        set_info('Список моделей очищен', 'green')

    update_lineup_list()
    ui_l.listWidget_lineup.clicked.connect(update_lineup_info)
    ui_l.pushButton_remove.clicked.connect(remove_lineup_model)
    ui_l.pushButton_clear.clicked.connect(clear_lineup_model)
    lineupModel.exec_()

