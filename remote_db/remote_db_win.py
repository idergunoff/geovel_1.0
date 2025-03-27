from remote_db.model_remote_db import *
from models_db.model import *
from qt.rem_db_window import *
from func import *
import hashlib

def open_rem_db_window():
    """ Открытие окна для работы с удаленной БД """
    RemoteDB = QtWidgets.QDialog()
    ui_rdb = Ui_rem_db()
    ui_rdb.setupUi(RemoteDB)
    RemoteDB.show()

    RemoteDB.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    def get_object_rem_id():
        """ Получение id выбранного объекта """
        try:
            return int(ui_rdb.comboBox_object_rem.currentText().split('id')[-1])
        except ValueError:
            pass


    def get_research_rem_id():
        """ Получение id выбранного исследования """
        try:
            return int(ui_rdb.comboBox_research_rem.currentText().split(' id')[-1])
        except ValueError:
            pass


    def get_profile_rem_id():
        """ Получение id выбранного профиля """
        try:
            return int(ui_rdb.comboBox_profile_rem.currentText().split(' id')[-1])
        except ValueError:
            pass


    def update_profile_rem_combobox():
        """ Обновление списка профилей в выпадающем списке """
        # Очистка выпадающего списка
        ui_rdb.comboBox_profile_rem.clear()
        with get_session() as session_r:
            try:
                profiles_rem = session_r.query(ProfileRDB.id, ProfileRDB.title).filter(
                    ProfileRDB.research_id == get_research_rem_id()
                ).order_by(ProfileRDB.id).all()
            except ValueError:
                return
            # Запрос на получение всех профилей, относящихся к исследованию, и их добавление в выпадающий список
            for i in profiles_rem:
                # count_measure = len(json.loads(i.signal))
                ui_rdb.comboBox_profile_rem.addItem(f'{i[1]} id{i[0]}')


    def update_research_rem_combobox():
        """ Обновление списка исследований в выпадающем списке """
        ui_rdb.comboBox_research_rem.clear()
        with get_session() as session_r:
            for i in session_r.query(ResearchRDB).filter(ResearchRDB.object_id == get_object_rem_id()).order_by(
                    ResearchRDB.date_research).all():
                ui_rdb.comboBox_research_rem.addItem(f'{i.date_research.strftime("%m.%Y")} id{i.id}')
        update_profile_rem_combobox()

    def update_object_rem_combobox():
        """ Обновление списка объектов в выпадающем списке """

        # Очистка выпадающего списка объектов
        ui_rdb.comboBox_object_rem.clear()

        # Получение всех объектов из базы данных, отсортированных по дате исследования
        with get_session() as session_r:
            for i in session_r.query(GeoradarObjectRDB).order_by(GeoradarObjectRDB.title).all():

                # Добавление названия объекта, даты исследования и идентификатора объекта в выпадающий список
                ui_rdb.comboBox_object_rem.addItem(f'{i.title} id{i.id}')

        # Обновление выпадающих списков исследований
        update_research_rem_combobox()

    update_object_rem_combobox()
    # При изменении объекта -> обновить исследования
    ui_rdb.comboBox_object_rem.currentIndexChanged.connect(update_research_rem_combobox)
    # При изменении исследования -> обновить профили
    ui_rdb.comboBox_research_rem.currentIndexChanged.connect(update_profile_rem_combobox)

    def load_object_rem():
        """ Загрузка данных с удаленной БД на локальную """

        with get_session() as remote_session:

            # Обновляем хэш-суммы в обеих базах
            update_signal_hashes(session)
            update_signal_hashes_rdb(remote_session)

            # Получаем выбранный объект из удалённой базы
            remote_objects = remote_session.query(GeoradarObjectRDB).filter(GeoradarObjectRDB.id ==
                                                                            get_object_rem_id())

            for remote_object in remote_objects:
                # Проверяем существование объекта в локальной базе
                local_object = session.query(GeoradarObject).filter_by(title=remote_object.title).first()

                if not local_object:
                    # Добавляем отсутствующий объект
                    new_object = GeoradarObject(title=remote_object.title)
                    session.add(new_object)
                    session.commit()
                    local_object = new_object
                    set_info(f'Объект "{remote_object.title}" загружен с удаленной БД', 'green')
                else:
                    set_info(f'Объект "{remote_object.title}" есть в локальной БД', 'red')

                # Получаем все исследования для текущего объекта из удалённой базы
                remote_researches = remote_session.query(ResearchRDB).filter_by(object_id=remote_object.id).all()

                for remote_research in remote_researches:
                    # Проверяем существование исследования в локальной базе
                    local_research = session.query(Research).filter_by(
                        object_id=local_object.id,
                        date_research=remote_research.date_research
                    ).first()

                    if not local_research:
                        # Добавляем отсутствующее исследование
                        new_research = Research(
                            object_id=local_object.id,
                            date_research=remote_research.date_research
                        )
                        session.add(new_research)
                        session.commit()
                        local_research = new_research
                        set_info(f'Исследование от /{remote_research.date_research}/ загружено с удаленной БД', 'green')

                    else:
                        set_info(f'Исследование от /{remote_research.date_research}/ есть в локальной БД', 'red')

                    # Получаем все профили для текущего исследования из удалённой базы
                    remote_profiles = remote_session.query(ProfileRDB).filter_by(research_id=remote_research.id).all()

                    ui.progressBar.setMaximum(len(remote_profiles))

                    for n, remote_profile in enumerate(remote_profiles):
                        # Проверяем существование профиля в локальной базе по signal_hash
                        ui.progressBar.setValue(n+1)
                        local_profile = session.query(Profile).filter_by(
                            research_id=local_research.id,
                            signal_hash=remote_profile.signal_hash
                        ).first()

                        if not local_profile:
                            # Добавляем отсутствующий профиль
                            new_profile = Profile(
                                research_id=local_research.id,
                                title=remote_profile.title,
                                signal=remote_profile.signal,
                                signal_hash=remote_profile.signal_hash,
                                x_wgs=remote_profile.x_wgs,
                                y_wgs=remote_profile.y_wgs,
                                x_pulc=remote_profile.x_pulc,
                                y_pulc=remote_profile.y_pulc,
                                abs_relief=remote_profile.abs_relief,
                                depth_relief=remote_profile.depth_relief
                            )
                            session.add(new_profile)
                            set_info(f'Профиль "{remote_profile.title}" загружен с удаленной БД', 'green')

                        else:
                            set_info(f'Профиль "{remote_profile.title}" есть в локальной БД', 'red')

                    session.commit()

        update_object()


    def unload_object_rem():
        """ Выгрузка данных с локальной БД на удаленную """

        with get_session() as remote_session:

            # Обновляем хэш-суммы в обеих базах
            update_signal_hashes(session)
            update_signal_hashes_rdb(remote_session)

            # Получаем выбранный объект из локальной базы
            local_objects = session.query(GeoradarObject).filter(GeoradarObject.id == get_object_id())

            for local_object in local_objects:
                # Проверяем существование объекта в удаленной базе
                remote_object = remote_session.query(GeoradarObjectRDB).filter_by(title=local_object.title).first()

                if not remote_object:
                    # Добавляем отсутствующий объект
                    new_object = GeoradarObjectRDB(title=local_object.title)
                    remote_session.add(new_object)
                    remote_session.commit()
                    remote_object = new_object
                    set_info(f'Объект "{local_object.title}" выгружен на удаленную БД', 'green')

                else:
                    set_info(f'Объект "{local_object.title}" есть в удаленной БД', 'red')

                # Получаем все исследования для текущего объекта из локальной базы
                local_researches = session.query(Research).filter_by(object_id=local_object.id).all()

                for local_research in local_researches:
                    # Проверяем существование исследования в удаленной базе
                    remote_research = remote_session.query(ResearchRDB).filter_by(
                        object_id=remote_object.id,
                        date_research=local_research.date_research
                    ).first()

                    if not remote_research:
                        # Добавляем отсутствующее исследование
                        new_research = ResearchRDB(
                            object_id=remote_object.id,
                            date_research=local_research.date_research
                        )
                        remote_session.add(new_research)
                        remote_session.commit()
                        remote_research = new_research
                        set_info(f'Исследование от /{local_research.date_research}/ выгружено на удаленную БД',
                                 'green')

                    else:
                        set_info(f'Исследование от /{local_research.date_research}/ есть в удаленной БД', 'red')

                    # Получаем все профили для текущего исследования из локальной базы
                    local_profiles = session.query(Profile).filter_by(research_id=local_research.id).all()

                    ui.progressBar.setMaximum(len(local_profiles))

                    for n, local_profile in enumerate(local_profiles):
                        # Проверяем существование профиля в удаленной базе по signal_hash
                        ui.progressBar.setValue(n+1)
                        remote_profile = remote_session.query(ProfileRDB).filter_by(
                            research_id=remote_research.id,
                            signal_hash=local_profile.signal_hash
                        ).first()

                        if not remote_profile:
                            # Добавляем отсутствующий профиль
                            new_profile = ProfileRDB(
                                research_id=remote_research.id,
                                title=local_profile.title,
                                signal=local_profile.signal,
                                signal_hash=local_profile.signal_hash,
                                x_wgs=local_profile.x_wgs,
                                y_wgs=local_profile.y_wgs,
                                x_pulc=local_profile.x_pulc,
                                y_pulc=local_profile.y_pulc,
                                abs_relief=local_profile.abs_relief,
                                depth_relief=local_profile.depth_relief
                            )
                            remote_session.add(new_profile)
                            set_info(f'Профиль "{local_profile.title}" выгружен на удаленную БД', 'green')

                        else:
                            set_info(f'Профиль "{local_profile.title}" есть в удаленной БД', 'red')

                    remote_session.commit()

        update_object_rem_combobox()


    def delete_object_rem():
        title_object = ui_rdb.comboBox_object_rem.currentText().split(' id')[0]
        result = QtWidgets.QMessageBox.question(RemoteDB, 'Delete object in RemoteDB',
                                                f'Вы уверены, что хотите удалить объект "{title_object}" со всеми '
                                                f'исследованиями и профилями?',
                                                QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if result == QtWidgets.QMessageBox.Yes:
            with get_session() as remote_session:
                if remote_session.query(ProfileRDB.id).filter(ProfileRDB.id == get_profile_rem_id()).count() > 0:
                    remote_session.query(ProfileRDB).filter(ProfileRDB.research_id == get_research_rem_id()).delete()
                    remote_session.query(ResearchRDB).filter(ResearchRDB.object_id == get_object_rem_id()).delete()
                    remote_session.query(GeoradarObjectRDB).filter(GeoradarObjectRDB.id == get_object_rem_id()).delete()
                elif remote_session.query(ResearchRDB).filter(ResearchRDB.id == get_research_rem_id()).count() > 0:
                    remote_session.query(ResearchRDB).filter(ResearchRDB.object_id == get_object_rem_id()).delete()
                    remote_session.query(GeoradarObjectRDB).filter(GeoradarObjectRDB.id == get_object_rem_id()).delete()
                else:
                    remote_session.query(GeoradarObjectRDB).filter(GeoradarObjectRDB.id == get_object_rem_id()).delete()
                remote_session.commit()

            set_info(f'Объект "{title_object}" и все его исследования и профили удалены в удаленной БД', 'green')
            update_object_rem_combobox()
        else:
            pass

    ui_rdb.pushButton_load_obj_rem.clicked.connect(load_object_rem)
    ui_rdb.pushButton_unload_obj_rem.clicked.connect(unload_object_rem)
    ui_rdb.pushButton_delete_obj_rem.clicked.connect(delete_object_rem)


    # Функция вычисления хэш-суммы
    def calculate_hash(value):
        return hashlib.sha256(str(value).encode()).hexdigest()

    def update_signal_hashes(session):
        """ Обновление хэш-сумм в таблице Profile """
        profiles = session.query(Profile).all()
        for profile in profiles:
            if profile.signal and not profile.signal_hash:
                profile.signal_hash = calculate_hash(profile.signal)
        session.commit()

    def update_signal_hashes_rdb(session):
        """ Обновление хэш-сумм в таблице ProfileRDB """
        profiles = session.query(ProfileRDB).all()
        for profile in profiles:
            if profile.signal and not profile.signal_hash:
                profile.signal_hash = calculate_hash(profile.signal)
        session.commit()


    RemoteDB.exec_()