from remote_db.model_remote_db import *
from models_db.model import *
from qt.rem_db_window import *
import hashlib

def open_rem_db_window():
    """ Открытие окна для работы с удаленной БД """
    RemoteDB = QtWidgets.QDialog()
    ui_rdb = Ui_rem_db()
    ui_rdb.setupUi(RemoteDB)
    RemoteDB.show()

    RemoteDB.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    # Функция получения id выбранного объекта
    def get_object_rem_id():
        try:
            return int(ui_rdb.comboBox_object_rem.currentText().split('id')[-1])
        except ValueError:
            pass

    # Функция получения id выбранного исследования
    def get_research_rem_id():
        return int(ui_rdb.comboBox_research_rem.currentText().split(' id')[-1])

    def update_object_rem_combobox():
        """ Функция для обновления списка объектов в выпадающем списке """
        # Очистка выпадающего списка объектов
        ui_rdb.comboBox_object_rem.clear()
        # Получение всех объектов из базы данных, отсортированных по дате исследования
        with get_session() as session_r:
            for i in session_r.query(GeoradarObject).order_by(GeoradarObject.title).all():
                # Добавление названия объекта, даты исследования и идентификатора объекта в выпадающий список
                ui_rdb.comboBox_object_rem.addItem(f'{i.title} id{i.id}')
        # Обновление выпадающего списка профилей
        update_research_rem_combobox()

    def update_research_rem_combobox():
        ui_rdb.comboBox_research_rem.clear()
        with get_session() as session_r:
            for i in session_r.query(Research).filter(Research.object_id == get_object_rem_id()).order_by(Research.date_research).all():
                ui_rdb.comboBox_research.addItem(f'{i.date_research.strftime("%m.%Y")} id{i.id}')
        update_profile_rem_combobox()

    def update_profile_rem_combobox():
        """ Обновление списка профилей в выпадающем списке """
        # Очистка выпадающего списка
        ui_rdb.comboBox_profile_rem.clear()
        try:
            with get_session() as session_r:
                profiles_rem = session_r.query(Profile).filter(Profile.research_id == get_research_rem_id()).all()
        except ValueError:
            return
        # Запрос на получение всех профилей, относящихся к объекту, и их добавление в выпадающий список
        for i in profiles_rem:
            count_measure = len(json.loads(i.signal))
            ui_rdb.comboBox_profile_rem.addItem(f'{i.title} ({count_measure} измерений) id{i.id}')

    # def load_object_rem():
    #     """ Загрузка объектов (с исследованиями и профилями) с удаленной базы данных на локальную """

    update_object_rem_combobox()
    RemoteDB.exec_()