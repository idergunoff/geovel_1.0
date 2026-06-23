import pylab as p
from psycopg2 import OperationalError
from PyQt5.QtWidgets import QListWidget
import hashlib

from remote_db.sync_wells import *
from remote_db.sync_well_relations import load_well_relations, unload_well_relations
from remote_db.sync_formations import load_formations, unload_formations
from remote_db.sync_objects import sync_objects_direction
from remote_db.unload_mlp import unload_mlp_func
from remote_db.sync_genetic import *
from remote_db.unload_mlp_models import unload_cls_models_func
from remote_db.unload_regmod import unload_regmod_func
from regression import update_list_reg, update_list_trained_models_regmod
from remote_db.unload_reg_models import unload_reg_models_func
from remote_db.sync_features.sync_entropy_features import *
from remote_db.sync_features.sync_entropy_features_profile import *
from remote_db.deduplicate_wells import *
from mlp import update_list_mlp
from geochem import update_combobox_geochem
from remote_db.sync_geochem import *
from remote_db.unload_geochem import *

def open_rem_db_window():
    try:
        BaseRDB.metadata.create_all(engine_remote)
    except OperationalError:
        set_info(f'Ошибка подключения к удаленной БД', 'red')

    """ Открытие окна для работы с удаленной БД """
    RemoteDB = QtWidgets.QDialog()
    ui_rdb = Ui_rem_db()
    ui_rdb.setupUi(RemoteDB)
    RemoteDB.show()

    RemoteDB.setAttribute(Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

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

            # Отдельно находим последнее добавленное исследование (по ID или дате создания)
            last_added = session_r.query(ResearchRDB).order_by(ResearchRDB.id.desc()).first()

            # Если такое исследование есть - выбираем его в комбобоксе
            if last_added:
                last_item_text = f'{last_added.date_research.strftime("%m.%Y")} id{last_added.id}'
                index = ui_rdb.comboBox_research_rem.findText(last_item_text)
                if index >= 0:
                    ui_rdb.comboBox_research_rem.setCurrentIndex(index)

        update_profile_rem_combobox()

    def update_object_rem_combobox(from_local=False):
        """ Обновление списка объектов в выпадающем списке """

        # Очистка выпадающего списка объектов
        ui_rdb.comboBox_object_rem.clear()

        # Получение всех объектов из базы данных, отсортированных по дате исследования
        with get_session() as session_r:
            for i in session_r.query(GeoradarObjectRDB).order_by(GeoradarObjectRDB.title).all():

                # Добавление названия объекта, даты исследования и идентификатора объекта в выпадающий список
                ui_rdb.comboBox_object_rem.addItem(f'{i.title} id{i.id}')
            if from_local:
                # Отдельно находим последний добавленный объект (по ID или дате создания)
                last_added = session_r.query(GeoradarObjectRDB).order_by(GeoradarObjectRDB.id.desc()).first()

                # Если такой объект есть - выбираем его в комбобоксе
                if last_added:
                    last_item_text = f'{last_added.title} id{last_added.id}'
                    index = ui_rdb.comboBox_object_rem.findText(last_item_text)
                    if index >= 0:
                        ui_rdb.comboBox_object_rem.setCurrentIndex(index)

        # Обновление выпадающих списков исследований
        update_research_rem_combobox()


    update_object_rem_combobox()
    # При изменении объекта -> обновить исследования
    ui_rdb.comboBox_object_rem.currentIndexChanged.connect(update_research_rem_combobox)
    # При изменении исследования -> обновить профили
    ui_rdb.comboBox_research_rem.currentIndexChanged.connect(update_profile_rem_combobox)

    def load_object_rem():
        """ Загрузка объектов, исследований и профилей с удаленной БД на локальную """

        with get_session() as remote_session:

            # Обновляем хэш-суммы в обеих базах
            update_signal_hashes(session)
            update_signal_hashes_rdb(remote_session)

            # Получаем выбранный объект из удалённой базы
            remote_objects = remote_session.query(GeoradarObjectRDB).filter(GeoradarObjectRDB.id ==
                                                                            get_object_rem_id())

            loaded_obj_count = 0

            for remote_object in tqdm(remote_objects, desc='Загрузка объектов'):
                # Проверяем существование объекта в локальной базе
                local_object = session.query(GeoradarObject).filter_by(title=remote_object.title).first()

                if not local_object:
                    # Добавляем отсутствующий объект
                    new_object = GeoradarObject(title=remote_object.title)
                    session.add(new_object)
                    session.commit()
                    local_object = new_object
                    set_info(f'Объект "{remote_object.title}" загружен с удаленной БД', 'green')
                    loaded_obj_count += 1
                else:
                    set_info(f'Объект "{remote_object.title}" есть в локальной БД', 'red')

                # Получаем все исследования для текущего объекта из удалённой базы
                remote_researches = remote_session.query(ResearchRDB).filter_by(object_id=remote_object.id).all()

                for remote_research in tqdm(remote_researches, desc='Загрузка исследований'):
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

                    for n, remote_profile in tqdm(enumerate(remote_profiles), desc='Загрузка профилей'):
                        # Проверяем существование профиля в локальной базе по signal_hash
                        ui.progressBar.setValue(n+1)
                        local_profile = session.query(Profile).filter_by(
                            research_id=local_research.id,
                            signal_hash_md5=remote_profile.signal_hash_md5
                        ).first()

                        if not local_profile:
                            # Добавляем отсутствующий профиль
                            new_profile = Profile(
                                research_id=local_research.id,
                                title=remote_profile.title,
                                signal=remote_profile.signal,
                                signal_hash_md5=remote_profile.signal_hash_md5,
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

        if loaded_obj_count == 1:
            update_object(new_obj=True)
        else:
            update_object()

        set_info(f'Загрузка данных с удаленной БД на локальную завершена', 'blue')


    def unload_object_rem():
        """ Выгрузка объектов, исследований и профилей с локальной БД на удаленную """

        with get_session() as remote_session:

            # Обновляем хэш-суммы в обеих базах
            update_signal_hashes(session)
            update_signal_hashes_rdb(remote_session)

            # Получаем выбранный объект из локальной базы
            local_objects = session.query(GeoradarObject).filter(GeoradarObject.id == get_object_id())

            unloaded_obj_count = 0

            for local_object in tqdm(local_objects, desc='Выгрузка объекта'):
                # Проверяем существование объекта в удаленной базе
                remote_object = remote_session.query(GeoradarObjectRDB).filter_by(title=local_object.title).first()

                if not remote_object:
                    # Добавляем отсутствующий объект
                    new_object = GeoradarObjectRDB(title=local_object.title)
                    remote_session.add(new_object)
                    remote_session.commit()
                    remote_object = new_object
                    set_info(f'Объект "{local_object.title}" выгружен на удаленную БД', 'green')
                    unloaded_obj_count += 1

                else:
                    set_info(f'Объект "{local_object.title}" есть в удаленной БД', 'red')

                # Получаем все исследования для текущего объекта из локальной базы
                local_researches = session.query(Research).filter_by(object_id=local_object.id).all()

                for local_research in tqdm(local_researches, desc='Выгрузка исследований'):
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

                    for n, local_profile in tqdm(enumerate(local_profiles), desc='Выгрузка профилей'):
                        # Проверяем существование профиля в удаленной базе по signal_hash
                        ui.progressBar.setValue(n+1)
                        remote_profile = remote_session.query(ProfileRDB).filter_by(
                            research_id=remote_research.id,
                            signal_hash_md5=local_profile.signal_hash_md5
                        ).count()

                        if remote_profile == 0:
                            # Добавляем отсутствующий профиль
                            new_profile = ProfileRDB(
                                research_id=remote_research.id,
                                title=local_profile.title,
                                signal=local_profile.signal,
                                signal_hash_md5=local_profile.signal_hash_md5,
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

        if unloaded_obj_count == 1:
            update_object_rem_combobox(from_local=True)
        else:
            update_object_rem_combobox()

        set_info(f'Выгрузка данных с локальной БД на удаленную завершена', 'blue')

    def delete_object_rem():
        """ Удаление текущего объекта вместе со всеми связанными данными """
        title_object = ui_rdb.comboBox_object_rem.currentText().split(' id')[0]
        object_id = get_object_rem_id()

        result = QtWidgets.QMessageBox.question(
            RemoteDB,
            'Delete object in RemoteDB',
            f'Вы уверены, что хотите удалить объект "{title_object}" со всеми исследованиями, профилями, пластами и '
            f'обучающими скважинами?',
            QtWidgets.QMessageBox.Yes,
            QtWidgets.QMessageBox.No
        )

        if result == QtWidgets.QMessageBox.Yes:
            with get_session() as remote_session:
                try:
                    # Удаляем все markups mlp, связанные с профилями
                    remote_session.query(MarkupMLPRDB) \
                        .filter(MarkupMLPRDB.profile_id == ProfileRDB.id) \
                        .filter(ProfileRDB.research_id == ResearchRDB.id) \
                        .filter(ResearchRDB.object_id == object_id) \
                        .delete(synchronize_session=False)

                    # Удаляем все markups reg, связанные с профилями
                    remote_session.query(MarkupRegRDB) \
                        .filter(MarkupRegRDB.profile_id == ProfileRDB.id) \
                        .filter(ProfileRDB.research_id == ResearchRDB.id) \
                        .filter(ResearchRDB.object_id == object_id) \
                        .delete(synchronize_session=False)

                    # Удаляем все праметры пластов, связанные с объектом
                    remote_session.query(FormationFeatureRDB) \
                        .filter(FormationFeatureRDB.formation_id == FormationRDB.id) \
                        .filter(FormationRDB.profile_id == ProfileRDB.id) \
                        .filter(ProfileRDB.research_id == ResearchRDB.id) \
                        .filter(ResearchRDB.object_id == object_id) \
                        .delete(synchronize_session=False)

                    remote_session.query(WaveletFeatureRDB) \
                        .filter(WaveletFeatureRDB.formation_id == FormationRDB.id) \
                        .filter(FormationRDB.profile_id == ProfileRDB.id) \
                        .filter(ProfileRDB.research_id == ResearchRDB.id) \
                        .filter(ResearchRDB.object_id == object_id) \
                        .delete(synchronize_session=False)

                    remote_session.query(FractalFeatureRDB) \
                        .filter(FractalFeatureRDB.formation_id == FormationRDB.id) \
                        .filter(FormationRDB.profile_id == ProfileRDB.id) \
                        .filter(ProfileRDB.research_id == ResearchRDB.id) \
                        .filter(ResearchRDB.object_id == object_id) \
                        .delete(synchronize_session=False)

                    remote_session.query(EntropyFeatureRDB) \
                        .filter(EntropyFeatureRDB.formation_id == FormationRDB.id) \
                        .filter(FormationRDB.profile_id == ProfileRDB.id) \
                        .filter(ProfileRDB.research_id == ResearchRDB.id) \
                        .filter(ResearchRDB.object_id == object_id) \
                        .delete(synchronize_session=False)

                    remote_session.query(NonlinearFeatureRDB) \
                        .filter(NonlinearFeatureRDB.formation_id == FormationRDB.id) \
                        .filter(FormationRDB.profile_id == ProfileRDB.id) \
                        .filter(ProfileRDB.research_id == ResearchRDB.id) \
                        .filter(ResearchRDB.object_id == object_id) \
                        .delete(synchronize_session=False)

                    remote_session.query(MorphologyFeatureRDB) \
                        .filter(MorphologyFeatureRDB.formation_id == FormationRDB.id) \
                        .filter(FormationRDB.profile_id == ProfileRDB.id) \
                        .filter(ProfileRDB.research_id == ResearchRDB.id) \
                        .filter(ResearchRDB.object_id == object_id) \
                        .delete(synchronize_session=False)

                    remote_session.query(FrequencyFeatureRDB) \
                        .filter(FrequencyFeatureRDB.formation_id == FormationRDB.id) \
                        .filter(FormationRDB.profile_id == ProfileRDB.id) \
                        .filter(ProfileRDB.research_id == ResearchRDB.id) \
                        .filter(ResearchRDB.object_id == object_id) \
                        .delete(synchronize_session=False)

                    remote_session.query(EnvelopeFeatureRDB) \
                        .filter(EnvelopeFeatureRDB.formation_id == FormationRDB.id) \
                        .filter(FormationRDB.profile_id == ProfileRDB.id) \
                        .filter(ProfileRDB.research_id == ResearchRDB.id) \
                        .filter(ResearchRDB.object_id == object_id) \
                        .delete(synchronize_session=False)

                    remote_session.query(AutocorrFeatureRDB) \
                        .filter(AutocorrFeatureRDB.formation_id == FormationRDB.id) \
                        .filter(FormationRDB.profile_id == ProfileRDB.id) \
                        .filter(ProfileRDB.research_id == ResearchRDB.id) \
                        .filter(ResearchRDB.object_id == object_id) \
                        .delete(synchronize_session=False)

                    remote_session.query(EMDFeatureRDB) \
                        .filter(EMDFeatureRDB.formation_id == FormationRDB.id) \
                        .filter(FormationRDB.profile_id == ProfileRDB.id) \
                        .filter(ProfileRDB.research_id == ResearchRDB.id) \
                        .filter(ResearchRDB.object_id == object_id) \
                        .delete(synchronize_session=False)

                    remote_session.query(HHTFeatureRDB) \
                        .filter(HHTFeatureRDB.formation_id == FormationRDB.id) \
                        .filter(FormationRDB.profile_id == ProfileRDB.id) \
                        .filter(ProfileRDB.research_id == ResearchRDB.id) \
                        .filter(ResearchRDB.object_id == object_id) \
                        .delete(synchronize_session=False)

                    # Удаляем все энтропии профиля, связанные с объектом
                    remote_session.query(EntropyFeatureProfileRDB) \
                        .filter(EntropyFeatureProfileRDB.profile_id == ProfileRDB.id) \
                        .filter(ProfileRDB.research_id == ResearchRDB.id) \
                        .filter(ResearchRDB.object_id == object_id) \
                        .delete(synchronize_session=False)

                    # Удаляем все пласты, связанные с объектом
                    remote_session.query(FormationRDB) \
                        .filter(FormationRDB.profile_id == ProfileRDB.id) \
                        .filter(ProfileRDB.research_id == ResearchRDB.id) \
                        .filter(ResearchRDB.object_id == object_id) \
                        .delete(synchronize_session=False)

                    # Удаляем все профили, связанные с объектом
                    remote_session.query(ProfileRDB) \
                        .filter(ProfileRDB.research_id == ResearchRDB.id) \
                        .filter(ResearchRDB.object_id == object_id) \
                        .delete(synchronize_session=False)

                    # Удаляем все исследования объекта
                    remote_session.query(ResearchRDB) \
                        .filter(ResearchRDB.object_id == object_id) \
                        .delete(synchronize_session=False)

                    # Удаляем сам объект
                    remote_session.query(GeoradarObjectRDB) \
                        .filter(GeoradarObjectRDB.id == object_id) \
                        .delete()

                    remote_session.commit()
                    set_info(f'Объект "{title_object}" и все его исследования, профили, пласты и обучающие скважины '
                             f'удалены в удаленной БД',
                             'green')
                    update_object_rem_combobox()

                except Exception as e:
                    remote_session.rollback()
                    set_info(f'Ошибка при удалении: {str(e)}', 'red')

    def sync_all_objects():
        """Синхронизация всех объектов с исследованиями и профилями"""
        try:
            with get_session() as remote_session:

                update_signal_hashes(session)  # Обновляем хэши для локальной таблицы
                update_signal_hashes_rdb(remote_session)  # Обновляем хэши для удалённой таблицы

                set_info('Начало синхронизации...', 'blue')

                # Синхронизация объектов (удаленная -> локальная)
                set_info(f'Обновление объектов в локальной БД...', 'blue')
                sync_objects_direction(remote_session, session, GeoradarObjectRDB, ResearchRDB, ProfileRDB, GeoradarObject,
                               Research, Profile)
                update_object()
                set_info(f'Обновление объектов в локальной БД завершено', 'blue')

                # Синхронизация объектов (локальная -> удаленная)
                set_info(f'Обновление объектов в удаленной БД...', 'blue')
                sync_objects_direction(session, remote_session, GeoradarObject, Research, Profile, GeoradarObjectRDB,
                                       ResearchRDB, ProfileRDB)
                update_object_rem_combobox()
                set_info(f'Обновление объектов в удаленной БД завершено', 'blue')

                set_info('Синхронизация завершена', 'blue')


        except Exception as e:
            # Откат изменений в случае ошибки
            session.rollback()
            set_info(f'Синхронизация прервалась: {str(e)}', 'red')
            raise  # Проброс исключения для дальнейшей обработки
        finally:
            session.close()


    # Функция вычисления хэш-суммы
    def calculate_hash(value):
        return hashlib.md5(str(value).encode()).hexdigest()


    def update_signal_hashes(session):
        """ Обновление хэш-сумм в таблице Profile """
        profiles = session.query(Profile.id, Profile.signal_hash_md5).all()
        for p in profiles:
            if not p[1]:
                profile = session.query(Profile).filter_by(id=p[0]).first()
                if profile.signal:
                    profile.signal_hash_md5 = calculate_hash(profile.signal)
        session.commit()


    def update_signal_hashes_rdb(session):
        """ Обновление хэш-сумм в таблице ProfileRDB """
        profiles = session.query(ProfileRDB.id, ProfileRDB.signal_hash_md5).all()
        for p in profiles:
            if not p[1]:
                profile = session.query(ProfileRDB).filter_by(id=p[0]).first()
                if profile.signal:
                    profile.signal_hash_md5 = calculate_hash(profile.signal)
        session.commit()

    #####################################################
    ##################  Classification  #################
    #####################################################

    def update_mlp_rdb_combobox(from_local=False):
        """ Обновление списка анализов MLP """
        with get_session() as remote_session:
            ui_rdb.comboBox_mlp_rdb.clear()
            for i in remote_session.query(AnalysisMLPRDB.id, AnalysisMLPRDB.title).order_by(AnalysisMLPRDB.title).all():
                # count_markup = remote_session.query(MarkupMLPRDB).filter_by(analysis_id=i.id).count()
                # ui_rdb.comboBox_mlp_rdb.addItem(f'{i.title}({count_markup}) id{i.id}')
                ui_rdb.comboBox_mlp_rdb.addItem(f'{i.title} id{i.id}')
            if from_local:
                # Отдельно находим последний добавленный объект (по ID или дате создания)
                last_added = remote_session.query(AnalysisMLPRDB).order_by(AnalysisMLPRDB.id.desc()).first()

                # Если такой объект есть - выбираем его в комбобоксе
                if last_added:
                    last_item_text = f'{last_added.title} id{last_added.id}'
                    index = ui_rdb.comboBox_mlp_rdb.findText(last_item_text)
                    if index >= 0:
                        ui_rdb.comboBox_mlp_rdb.setCurrentIndex(index)

        update_trained_models_class_rdb(from_local=True)

    def unload_mlp():
        """ Запуск выгрузки анализов MLP """
        unload_mlp_func(ui_rdb, RemoteDB)
        update_mlp_rdb_combobox(from_local=True)

    def get_MLP_rdb_id():
        """Получение id текущего анализа MLP в удаленной БД"""
        try:
            return int(ui_rdb.comboBox_mlp_rdb.currentText().split('id')[-1])
        except ValueError:
            pass


    def check_dependencies():
        """Проверка наличия связанных данных MarkupMLP в локальной БД"""
        set_info('Проверка наличия всех связанных данных MarkupMLP в локальной БД', 'blue')
        errors = []
        with get_session() as remote_session:
            remote_analyzes = remote_session.query(AnalysisMLPRDB).filter(AnalysisMLPRDB.id == get_MLP_rdb_id())

        # Предзагрузка данных из локальной БД для проверки
            # Проверяем наличие всех связанных данных в локальной БД
            local_wells = {
                w.well_hash: w.id
                for w in session.query(Well.well_hash, Well.id).all()
            }

            local_profiles = {
                p.signal_hash_md5: p.id
                for p in session.query(Profile.signal_hash_md5, Profile.id).all()
            }

            local_formations = {}
            for f in session.query(Formation.up_hash, Formation.down_hash, Formation.id).all():
                local_formations[f.up_hash] = f.id
                local_formations[f.down_hash] = f.id

            for remote_analysis in remote_analyzes:

                remote_markers = remote_session.query(MarkerMLPRDB).filter_by(analysis_id=remote_analysis.id).all()

                for remote_marker in remote_markers:
                    remote_markups = remote_session.query(MarkupMLPRDB) \
                        .filter_by(
                        analysis_id=remote_analysis.id,
                        marker_id=remote_marker.id
                    ).all()

                    ui.progressBar.setMaximum(len(remote_markups))

                    for n, remote_markup in tqdm(enumerate(remote_markups), desc='Проверка зависимостей MLP'):
                        ui.progressBar.setValue(n + 1)
                        related_tables = []

                        # Проверяем скважину
                        try:
                            remote_well_hash = remote_markup.well.well_hash
                            if remote_well_hash not in local_wells:
                                related_tables.append('Well')
                        except AttributeError:
                            pass

                        # Проверяем профиль
                        try:
                            remote_profile_hash = remote_markup.profile.signal_hash_md5
                            if remote_profile_hash not in local_profiles:
                                related_tables.append('Profile')
                        except AttributeError:
                            pass

                        # Проверяем пласт
                        remote_formation_up_hash = remote_markup.formation.up_hash
                        remote_formation_down_hash = remote_markup.formation.down_hash
                        if (remote_formation_up_hash not in local_formations and
                                remote_formation_down_hash not in local_formations):
                            related_tables.append('Formation')

                        if related_tables:
                            try:
                                error_msg = (
                                    f'Для маркера "{remote_marker.title}" анализа "{remote_analysis.title}" '
                                    f'отсутствуют данные в таблицах: {", ".join(related_tables)}. '
                                )
                                errors.append(error_msg)
                            except AttributeError:
                                pass

        return errors

    def load_mlp():
        """Загрузка таблиц AnalysisMLP, MarkerMLP, MarkupMLP с удаленной БД на локальную"""

        set_info('Начало загрузки данных с удаленной БД на локальную', 'blue')

        if not ui_rdb.checkBox_dont_check_dependencies.isChecked():
            # Сначала выполняем проверку
            dependency_errors = check_dependencies()

            if dependency_errors:
                error_info = "Обнаружены следующие проблемы:\n\n" + "\n\n".join(dependency_errors)
                error_info += "\n\nНеобходимо сначала синхронизировать эти данные с удаленной БД."
                set_info('Обнаружены проблемы с зависимостями. Загрузка данных прекращена', 'red')
                QMessageBox.critical(RemoteDB, 'Ошибка зависимостей', error_info)
                return
            else:
                set_info('Проблем с зависимостями нет', 'green')

        # Если проверка пройдена, выполняем выгрузку
        with get_session() as remote_session:
            remote_analyzes = remote_session.query(AnalysisMLPRDB).filter(AnalysisMLPRDB.id == get_MLP_rdb_id())

            for remote_analysis in tqdm(remote_analyzes, desc='Загрузка анализа'):

                local_analysis = session.query(AnalysisMLP).filter_by(title=remote_analysis.title).first()

                if not local_analysis:
                    new_analysis = AnalysisMLP(title=remote_analysis.title)
                    session.add(new_analysis)
                    session.commit()
                    local_analysis = new_analysis
                    set_info(f'Анализ "{remote_analysis.title}" загружен с удаленной БД', 'green')
                else:
                    set_info(f'Анализ "{remote_analysis.title}" есть в локальной БД', 'blue')

                remote_markers = remote_session.query(MarkerMLPRDB).filter_by(analysis_id=remote_analysis.id).all()

                for remote_marker in tqdm(remote_markers, desc='Загрузка маркеров'):
                    local_marker = session.query(MarkerMLP).filter_by(
                        analysis_id=local_analysis.id,
                        title=remote_marker.title
                    ).first()

                    if not local_marker:
                        new_marker = MarkerMLP(
                            analysis_id=local_analysis.id,
                            title=remote_marker.title,
                            color=remote_marker.color
                        )
                        session.add(new_marker)
                        session.commit()
                        local_marker = new_marker
                        set_info(f'Маркер "{remote_marker.title}" загружен с удаленной БД', 'green')
                    else:
                        set_info(f'Маркер "{remote_marker.title}" есть в локальной БД', 'blue')

                    remote_markups = remote_session.query(MarkupMLPRDB) \
                        .filter_by(
                        analysis_id=remote_analysis.id,
                        marker_id=remote_marker.id
                    ).all()

                    # Повторно загружаем данные
                    local_wells = {
                        w.well_hash: w.id
                        for w in session.query(Well.well_hash, Well.id).all()
                    }

                    local_formations = {}
                    for f in session.query(Formation.up_hash, Formation.down_hash, Formation.id,
                                           Formation.profile_id).all():
                        local_formations[f.up_hash] = [f.id, f.profile_id]
                        local_formations[f.down_hash] = [f.id, f.profile_id]

                    added_markups_count = 0

                    ui.progressBar.setMaximum(len(remote_markups))

                    for n, remote_markup in tqdm(enumerate(remote_markups), desc='Загрузка обучающих скважин'):
                        ui.progressBar.setValue(n+1)

                        local_well_id = local_wells[remote_markup.well.well_hash] \
                            if remote_markup.well_id != None else 0

                        # Получаем ID пласта
                        local_formation_list = local_formations.get(remote_markup.formation.up_hash)
                        if not local_formation_list:
                            local_formation_list = local_formations.get(remote_markup.formation.down_hash)

                        local_markup = session.query(MarkupMLP).filter_by(
                            analysis_id=local_analysis.id,
                            well_id=local_well_id,
                            profile_id=local_formation_list[1],
                            formation_id=local_formation_list[0],
                            marker_id=local_marker.id
                        ).first()

                        if not local_markup:
                            new_markup = MarkupMLP(
                                analysis_id=local_analysis.id,
                                well_id=local_well_id,
                                profile_id=local_formation_list[1],
                                formation_id=local_formation_list[0],
                                marker_id=local_marker.id,
                                list_measure=remote_markup.list_measure,
                                type_markup=remote_markup.type_markup
                            )
                            session.add(new_markup)
                            added_markups_count += 1

                    session.commit()
                    set_info(
                        f'Загружено: '
                        f'{pluralize(added_markups_count, ["обучающая скважина", "обучающие скважины", "обучающих скважин"])}',
                        'green')

        update_list_mlp(True)

        set_info('Загрузка данных с удаленной БД на локальную завершена', 'blue')


    def delete_mlp_rdb():
        """Удаление текущего анализа MLP в удаленной БД"""
        title_analysis = ui_rdb.comboBox_mlp_rdb.currentText().split(' id')[0]
        analysis_id = get_MLP_rdb_id()

        result = QtWidgets.QMessageBox.question(
            RemoteDB,
            'Delete MLP in RemoteDB',
            f'Вы уверены, что хотите удалить анализ "{title_analysis}" со всеми маркерами и обучающими скважинами?',
            QtWidgets.QMessageBox.Yes,
            QtWidgets.QMessageBox.No
        )

        if result == QtWidgets.QMessageBox.Yes:
            with get_session() as remote_session:
                try:
                    # Удаляем все обучающие скважины, связанные с анализом
                    remote_session.query(MarkupMLPRDB) \
                        .filter(MarkupMLPRDB.analysis_id == analysis_id) \
                        .delete(synchronize_session=False)

                    # Удаляем все маркеры анализа
                    remote_session.query(MarkerMLPRDB) \
                        .filter(MarkerMLPRDB.analysis_id == analysis_id) \
                        .delete(synchronize_session=False)

                    # Удаляем все генетические анализы анализа
                    remote_session.query(GeneticAlgorithmCLSRDB) \
                        .filter(GeneticAlgorithmCLSRDB.analysis_id == analysis_id) \
                        .delete(synchronize_session=False)

                    # Удаляем все модели анализа
                    remote_session.query(TrainedModelClassRDB) \
                        .filter(TrainedModelClassRDB.analysis_id == analysis_id) \
                        .delete(synchronize_session=False)

                    # Удаляем сам анализ
                    remote_session.query(AnalysisMLPRDB) \
                        .filter(AnalysisMLPRDB.id == analysis_id) \
                        .delete()

                    remote_session.commit()
                    set_info(f'Анализ "{title_analysis}" и все его маркеры, обучающие скважины, генетические анализы и '
                             f'тренированные модели удалены в удаленной '
                             f'БД', 'green')
                    update_mlp_rdb_combobox()

                except Exception as e:
                    remote_session.rollback()
                    set_info(f'Ошибка при удалении: {str(e)}', 'red')

    ui_rdb.checkBox_check_ga_params.setChecked(False)

    def start_sync_ga_cls():
        """ Запуск синхронизации генетического анализа MLP """
        sync_genetic_cls_func(ui_rdb)

    def update_trained_models_class_rdb(from_local=False):
        """ Обновление списка моделей MLP в выпадающем списке """
        # Очистка выпадающего списка
        ui_rdb.comboBox_trained_model_rdb.clear()
        with get_session() as remote_session:
            try:
                models_rdb = remote_session.query(TrainedModelClassRDB.id, TrainedModelClassRDB.title).filter(
                    TrainedModelClassRDB.analysis_id == get_MLP_rdb_id()
                ).order_by(TrainedModelClassRDB.id).all()
            except ValueError:
                return
            # Запрос на получение всех моделей, относящихся к анализу, и их добавление в выпадающий список
            for i in models_rdb:
                ui_rdb.comboBox_trained_model_rdb.addItem(f'{i[1]} id{i[0]}')
            if from_local:
                # Отдельно находим последний добавленный объект (по ID или дате создания)
                last_added = remote_session.query(TrainedModelClassRDB).order_by(TrainedModelClassRDB.id.desc()).first()

                # Если такой объект есть - выбираем его в комбобоксе
                if last_added:
                    last_item_text = f'{last_added.title} id{last_added.id}'
                    index = ui_rdb.comboBox_trained_model_rdb.findText(last_item_text)
                    if index >= 0:
                        ui_rdb.comboBox_trained_model_rdb.setCurrentIndex(index)

        # При изменении анализа -> обновить модели
    ui_rdb.comboBox_mlp_rdb.currentIndexChanged.connect(update_trained_models_class_rdb)

    def start_unload_mlp_model():
        """ Запуск выгрузки моделей MLP """
        unload_cls_models_func(RemoteDB)
        update_trained_models_class_rdb()

    def get_trained_model_class_rdb_id():
        """ Получение id выбранной модели """
        try:
            return int(ui_rdb.comboBox_trained_model_rdb.currentText().split('id')[-1])
        except ValueError:
            pass

    def load_cls_models_func():
        """Загрузка модели MLP с удаленной БД на локальную"""
        set_info('Начало загрузки данных с удаленной БД на локальную', 'blue')

        model_id = get_trained_model_class_rdb_id()
        if model_id is None:
            set_info("Модель не выбрана", "red")
            return

        with get_session() as remote_session:

            remote_analysis_name = remote_session.query(AnalysisMLPRDB.title).filter_by(id=get_MLP_rdb_id()).first()[0]
            local_analysis = session.query(AnalysisMLP).filter_by(title=remote_analysis_name).first()
            if not local_analysis:
                set_info(f"Соответствующий анализ '{remote_analysis_name}' не найден в удаленной БД. Сначала загрузите "
                         f"анализ.", "red")
                return

            remote_model = remote_session.query(TrainedModelClassRDB).filter(
                TrainedModelClassRDB.analysis_id == get_MLP_rdb_id(),
                TrainedModelClassRDB.id == model_id
            ).first()

            local_model = session.query(TrainedModelClass).filter_by(
                analysis_id=local_analysis.id,
                title=remote_model.title,
                list_params=remote_model.list_params,
                except_signal=remote_model.except_signal,
                except_crl=remote_model.except_crl,
                comment=remote_model.comment,
            ).first()

            if local_model:
                set_info(f"Модель '{remote_model.title}' анализа '{remote_analysis_name}' есть в удаленной БД", 'red')
                QMessageBox.critical(RemoteDB, 'Error',
                                     f"Модель '{remote_model.title}' анализа '{remote_analysis_name}' есть в удаленной БД")
            else:
                loaded_model = pickle.loads(remote_model.file_model)
                path_model = f'models/classifier/{remote_model.title}.pkl'
                with open(path_model, 'wb') as f:
                    pickle.dump(loaded_model, f)

                new_local_model = TrainedModelClass(
                    analysis_id=local_analysis.id,
                    title=remote_model.title,
                    path_model=path_model,
                    list_params=remote_model.list_params,
                    except_signal=remote_model.except_signal,
                    except_crl=remote_model.except_crl,
                    comment=remote_model.comment,
                )
                session.add(new_local_model)
                session.commit()

                if remote_model.mask:
                    set_info(f'Загрузка модели с маской', 'blue')
                    param_mask = session.query(ParameterMask).filter_by(mask=remote_model.mask).first()
                    if param_mask:
                        new_trained_model_mask = TrainedModelClassMask(
                            mask_id=param_mask.id,
                            model_id=new_local_model.id
                        )

                    else:
                        info = (f'cls analysis: {remote_analysis_name}\n'
                                f'pareto analysis: REMOTE MODEL')
                        new_param_mask = ParameterMask(
                            count_param=len(json.loads(remote_model.mask)),
                            mask=remote_model.mask,
                            mask_info=info
                        )
                        session.add(new_param_mask)
                        session.commit()
                        new_trained_model_mask = TrainedModelClassMask(
                            mask_id=new_param_mask.id,
                            model_id=new_local_model.id
                        )

                    session.add(new_trained_model_mask)
                    session.commit()

                set_info(f"Модель '{remote_model.title}' анализа '{remote_analysis_name}' загружена на локальную БД", 'green')

        update_list_trained_models_class()
        set_info('Загрузка данных с удаленной БД на локальную завершена', 'blue')


    def delete_cls_model_rdb():
        """ Удаление текущей модели MLP """
        title_model = ui_rdb.comboBox_trained_model_rdb.currentText().split(' id')[0]
        title_analysis = ui_rdb.comboBox_mlp_rdb.currentText().split(' id')[0]
        analysis_id = get_MLP_rdb_id()

        result = QtWidgets.QMessageBox.question(
            RemoteDB,
            'Delete trained model in RemoteDB',
            f'Вы уверены, что хотите удалить модель "{title_model}" анализа "{title_analysis}"?',
            QtWidgets.QMessageBox.Yes,
            QtWidgets.QMessageBox.No
        )

        if result == QtWidgets.QMessageBox.Yes:
            with get_session() as remote_session:
                try:
                    remote_session.query(TrainedModelClassRDB) \
                        .filter(TrainedModelClassRDB.id == get_trained_model_class_rdb_id()) \
                        .delete(synchronize_session=False)

                    remote_session.commit()
                    set_info(f'Модель "{title_model}" анализа "{title_analysis}" удалена в удаленной '
                             f'БД', 'green')
                    update_trained_models_class_rdb()

                except Exception as e:
                    remote_session.rollback()
                    set_info(f'Ошибка при удалении: {str(e)}', 'red')


    #####################################################
    ###################  Regression  ####################
    #####################################################

    def update_regmod_rdb_combobox(from_local=False):
        """Обновить список анализов MLP"""
        with get_session() as remote_session:
            ui_rdb.comboBox_regmod_rdb.clear()
            for i in remote_session.query(AnalysisRegRDB.id, AnalysisRegRDB.title).order_by(AnalysisRegRDB.title).all():
                # count_markup = remote_session.query(MarkupRegRDB).filter_by(analysis_id=i.id).count()
                # ui_rdb.comboBox_regmod_rdb.addItem(f'{i.title}({count_markup}) id{i.id}')
                ui_rdb.comboBox_regmod_rdb.addItem(f'{i.title} id{i.id}')
            if from_local:
                # Отдельно находим последний добавленный объект (по ID или дате создания)
                last_added = remote_session.query(AnalysisRegRDB).order_by(AnalysisRegRDB.id.desc()).first()

                # Если такой объект есть - выбираем его в комбобоксе
                if last_added:
                    last_item_text = f'{last_added.title} id{last_added.id}'
                    index = ui_rdb.comboBox_regmod_rdb.findText(last_item_text)
                    if index >= 0:
                        ui_rdb.comboBox_regmod_rdb.setCurrentIndex(index)


    def unload_regmod():
        """ Запуск выгрузки регрессионного анализа """
        unload_regmod_func(ui_rdb, RemoteDB)
        update_regmod_rdb_combobox(from_local=True)

    def get_regmod_rdb_id():
        """ Получение id текущего регрессионного анализа """
        try:
            return int(ui_rdb.comboBox_regmod_rdb.currentText().split('id')[-1])
        except ValueError:
            pass

    def delete_regmod_rdb():
        """ Удаление текущего регрессионного анализа в удаленной БД (вместе с данными в связанных таблицах) """
        title_analysis = ui_rdb.comboBox_regmod_rdb.currentText().split(' id')[0]
        analysis_id = get_regmod_rdb_id()

        result = QtWidgets.QMessageBox.question(
            RemoteDB,
            'Delete RegMod in RemoteDB',
            f'Вы уверены, что хотите удалить анализ "{title_analysis}" со всеми обучающими скважинами?',
            QtWidgets.QMessageBox.Yes,
            QtWidgets.QMessageBox.No
        )

        if result == QtWidgets.QMessageBox.Yes:
            with get_session() as remote_session:
                try:
                    # Удаляем все обучающие скважины, связанные с анализом
                    remote_session.query(MarkupRegRDB) \
                        .filter(MarkupRegRDB.analysis_id == analysis_id) \
                        .delete(synchronize_session=False)

                    # Удаляем все генетические анализы анализа
                    remote_session.query(GeneticAlgorithmRegRDB) \
                        .filter(GeneticAlgorithmRegRDB.analysis_id == analysis_id) \
                        .delete(synchronize_session=False)

                    # Удаляем все модели анализа
                    remote_session.query(TrainedModelRegRDB) \
                        .filter(TrainedModelRegRDB.analysis_id == analysis_id) \
                        .delete(synchronize_session=False)

                    # Удаляем сам анализ
                    remote_session.query(AnalysisRegRDB) \
                        .filter(AnalysisRegRDB.id == analysis_id) \
                        .delete()

                    remote_session.commit()
                    set_info(f'Анализ "{title_analysis}" и все его обучающие скважины, генетические анализы и '
                             f'тренированные модели удалены в удаленной '
                             f'БД', 'green')
                    update_regmod_rdb_combobox()

                except Exception as e:
                    remote_session.rollback()
                    set_info(f'Ошибка при удалении: {str(e)}', 'red')

    def check_reg_dependencies():
        """ Проверка наличия связанных данных таблицы MarkupReg в локальной БД """
        set_info('Проверка наличия всех связанных данных в локальной БД', 'blue')
        errors = []
        with get_session() as remote_session:
            remote_analyzes = remote_session.query(AnalysisRegRDB).filter(AnalysisRegRDB.id == get_regmod_rdb_id())

            # Предзагрузка данных из локальной БД для проверки
            # Проверяем наличие всех связанных данных в локальной БД
            local_wells = {
                w.well_hash: w.id
                for w in session.query(Well.well_hash, Well.id).all()
            }

            local_profiles = {
                p.signal_hash_md5: p.id
                for p in session.query(Profile.signal_hash_md5, Profile.id).all()
            }

            local_formations = {}
            for f in session.query(Formation.up_hash, Formation.down_hash, Formation.id).all():
                if f.up_hash:
                    local_formations[f.up_hash] = f.id
                if f.down_hash:
                    local_formations[f.down_hash] = f.id

            for remote_analysis in remote_analyzes:

                remote_markups = remote_session.query(MarkupRegRDB) \
                    .filter_by(
                    analysis_id=remote_analysis.id
                ).all()

                ui.progressBar.setMaximum(len(remote_markups))

                for n, remote_markup in tqdm(enumerate(remote_markups), desc='Проверка зависимостей RegMod'):
                    ui.progressBar.setValue(n + 1)
                    related_tables = []

                    # Проверяем скважину
                    try:
                        remote_well_hash = remote_markup.well.well_hash
                        if remote_well_hash not in local_wells:
                            related_tables.append('Well')
                    except AttributeError:
                        pass

                    # Проверяем профиль
                    try:
                        remote_profile_hash = remote_markup.profile.signal_hash_md5
                        if remote_profile_hash not in local_profiles:
                            related_tables.append('Profile')
                    except AttributeError:
                        pass

                    # Проверяем пласт
                    try:
                        remote_formation_up_hash = remote_markup.formation.up_hash
                        remote_formation_down_hash = remote_markup.formation.down_hash
                        # Если в локальной БД вообще нет пластов
                        if not local_formations:
                            related_tables.append('Formation')
                        else:
                            # Проверяем наличие хотя бы одного хеша пласта
                            if (remote_formation_up_hash not in local_formations and
                                    remote_formation_down_hash not in local_formations):
                                related_tables.append('Formation')
                    except AttributeError:
                        related_tables.append('Formation')

                    if related_tables:
                        try:
                            error_msg = (
                                f'Для анализа "{remote_analysis.title}" '
                                f'отсутствуют данные в таблицах: {", ".join(related_tables)}. '
                            )
                            errors.append(error_msg)
                        except AttributeError:
                            pass

        return errors

    def load_regmod():
        """Загрузка данных таблиц AnalysisReg, MarkupReg с удаленной БД на локальную"""

        set_info('Начало загрузки данных с удаленной БД на локальную', 'blue')

        if not ui_rdb.checkBox_dont_check_reg_dependencies.isChecked():
            # Сначала выполняем проверку
            dependency_errors = check_reg_dependencies()

            if dependency_errors:
                error_info = "Обнаружены следующие проблемы:\n\n" + "\n\n".join(dependency_errors)
                error_info += "\n\nНеобходимо сначала синхронизировать эти данные с удаленной БД."
                set_info('Обнаружены проблемы с зависимостями. Загрузка данных прекращена', 'red')
                QMessageBox.critical(RemoteDB, 'Ошибка зависимостей', error_info)
                return
            else:
                set_info('Проблем с зависимостями нет', 'green')

        # Если проверка пройдена, выполняем выгрузку
        with get_session() as remote_session:
            remote_analyzes = remote_session.query(AnalysisRegRDB).filter(AnalysisRegRDB.id == get_regmod_rdb_id())

            for remote_analysis in tqdm(remote_analyzes, desc='Загрузка анализа'):

                local_analysis = session.query(AnalysisReg).filter_by(title=remote_analysis.title).first()

                if not local_analysis:
                    new_analysis = AnalysisReg(title=remote_analysis.title)
                    session.add(new_analysis)
                    session.commit()
                    local_analysis = new_analysis
                    set_info(f'Анализ "{remote_analysis.title}" загружен с удаленной БД', 'green')
                else:
                    set_info(f'Анализ "{remote_analysis.title}" есть в локальной БД', 'blue')

                remote_markups = remote_session.query(MarkupRegRDB) \
                    .filter_by(
                    analysis_id=remote_analysis.id
                ).all()

                # Повторно загружаем данные
                local_wells = {
                    w.well_hash: w.id
                    for w in session.query(Well.well_hash, Well.id).all()
                }

                local_formations = {}
                for f in session.query(Formation.up_hash, Formation.down_hash, Formation.id,
                                       Formation.profile_id).all():
                    local_formations[f.up_hash] = [f.id, f.profile_id]
                    local_formations[f.down_hash] = [f.id, f.profile_id]

                added_markups_count = 0

                ui.progressBar.setMaximum(len(remote_markups))

                for n, remote_markup in tqdm(enumerate(remote_markups), desc='Загрузка обучающих скважин'):
                    ui.progressBar.setValue(n + 1)

                    local_well_id = local_wells[remote_markup.well.well_hash] \
                        if remote_markup.well_id != None else 0

                    # Получаем ID пласта
                    local_formation_list = local_formations.get(remote_markup.formation.up_hash)
                    if not local_formation_list:
                        local_formation_list = local_formations.get(remote_markup.formation.down_hash)

                    local_markup = session.query(MarkupReg).filter_by(
                        analysis_id=local_analysis.id,
                        well_id=local_well_id,
                        profile_id=local_formation_list[1],
                        formation_id=local_formation_list[0]
                    ).first()

                    if not local_markup:
                        new_markup = MarkupReg(
                            analysis_id=local_analysis.id,
                            well_id=local_well_id,
                            profile_id=local_formation_list[1],
                            formation_id=local_formation_list[0],
                            target_value=remote_markup.target_value,
                            list_measure=remote_markup.list_measure,
                            type_markup=remote_markup.type_markup
                        )
                        session.add(new_markup)
                        added_markups_count += 1

                session.commit()
                set_info(
                    f'Загружено: '
                    f'{pluralize(added_markups_count, ["обучающая скважина", "обучающие скважины", "обучающих скважин"])}',
                    'green')

        update_list_reg()

        set_info('Загрузка данных с удаленной БД на локальную завершена', 'blue')

    ui_rdb.checkBox_check_ga_params_reg.setChecked(False)

    def start_sync_ga_reg():
        """ Запуск синхронизации регрессионного генетического анализа """
        sync_genetic_reg_func(ui_rdb, RemoteDB)

    def update_trained_models_reg_rdb(from_local=False):
        """ Обновление списка моделей в выпадающем списке """
        # Очистка выпадающего списка
        ui_rdb.comboBox_trained_model_reg_rdb.clear()
        with get_session() as remote_session:
            try:
                models_rdb = remote_session.query(TrainedModelRegRDB.id, TrainedModelRegRDB.title).filter(
                    TrainedModelRegRDB.analysis_id == get_regmod_rdb_id()
                ).order_by(TrainedModelRegRDB.id).all()
            except ValueError:
                return
            # Запрос на получение всех моделей, относящихся к анализу, и их добавление в выпадающий список
            for i in models_rdb:
                ui_rdb.comboBox_trained_model_reg_rdb.addItem(f'{i[1]} id{i[0]}')
            if from_local:
                # Отдельно находим последний добавленный объект (по ID или дате создания)
                last_added = remote_session.query(TrainedModelRegRDB).order_by(TrainedModelRegRDB.id.desc()).first()

                # Если такой объект есть - выбираем его в комбобоксе
                if last_added:
                    last_item_text = f'{last_added.title} id{last_added.id}'
                    index = ui_rdb.comboBox_trained_model_reg_rdb.findText(last_item_text)
                    if index >= 0:
                        ui_rdb.comboBox_trained_model_reg_rdb.setCurrentIndex(index)

        # При изменении анализа -> обновить модели

    ui_rdb.comboBox_regmod_rdb.currentIndexChanged.connect(update_trained_models_reg_rdb)

    def start_unload_reg_model():
        """ Запуск выгрузки регрессионной модели """
        unload_reg_models_func(RemoteDB)
        update_trained_models_reg_rdb()

    def get_trained_model_reg_rdb_id():
        """ Получение id текущей регрессионной модели в удаленной БД """
        try:
            return int(ui_rdb.comboBox_trained_model_reg_rdb.currentText().split('id')[-1])
        except ValueError:
            pass

    def load_reg_models_func():
        """ Загрузка регрессионной модели """
        set_info('Начало загрузки данных с удаленной БД на локальную', 'blue')

        model_id = get_trained_model_reg_rdb_id()
        if model_id is None:
            set_info("Модель не выбрана", "red")
            return

        with get_session() as remote_session:

            remote_analysis_name = remote_session.query(AnalysisRegRDB.title).filter_by(id=get_regmod_rdb_id()).first()[0]
            local_analysis = session.query(AnalysisReg).filter_by(title=remote_analysis_name).first()
            if not local_analysis:
                set_info(f"Соответствующий анализ '{remote_analysis_name}' не найден в удаленной БД. Сначала загрузите "
                         f"анализ.", "red")
                return

            remote_model = remote_session.query(TrainedModelRegRDB).filter(
                TrainedModelRegRDB.analysis_id == get_regmod_rdb_id(),
                TrainedModelRegRDB.id == model_id
            ).first()

            local_model = session.query(TrainedModelReg).filter_by(
                analysis_id=local_analysis.id,
                title=remote_model.title,
                list_params=remote_model.list_params,
                except_signal=remote_model.except_signal,
                except_crl=remote_model.except_crl,
                comment=remote_model.comment,
            ).first()

            if local_model:
                set_info(f"Модель '{remote_model.title}' анализа '{remote_analysis_name}' есть в удаленной БД", 'red')
                QMessageBox.critical(RemoteDB, 'Error',
                                     f"Модель '{remote_model.title}' анализа '{remote_analysis_name}' есть в удаленной БД")
            else:
                loaded_model = pickle.loads(remote_model.file_model)
                path_model = f'models/regression/{remote_model.title}.pkl'
                with open(path_model, 'wb') as f:
                    pickle.dump(loaded_model, f)

                new_local_model = TrainedModelReg(
                    analysis_id=local_analysis.id,
                    title=remote_model.title,
                    path_model=path_model,
                    list_params=remote_model.list_params,
                    except_signal=remote_model.except_signal,
                    except_crl=remote_model.except_crl,
                    comment=remote_model.comment,
                )
                session.add(new_local_model)
                session.commit()

                if remote_model.mask:
                    set_info(f'Загрузка модели с маской', 'blue')
                    param_mask = session.query(ParameterMask).filter_by(mask=remote_model.mask).first()
                    if param_mask:
                        new_trained_model_mask = TrainedModelRegMask(
                            mask_id=param_mask.id,
                            model_id=new_local_model.id
                        )

                    else:
                        info = (f'reg analysis: {remote_analysis_name}\n'
                                f'pareto analysis: REMOTE MODEL')
                        new_param_mask = ParameterMask(
                            count_param=len(json.loads(remote_model.mask)),
                            mask=remote_model.mask,
                            mask_info=info
                        )
                        session.add(new_param_mask)
                        session.commit()
                        new_trained_model_mask = TrainedModelRegMask(
                            mask_id=new_param_mask.id,
                            model_id=new_local_model.id
                        )

                    session.add(new_trained_model_mask)
                    session.commit()

                set_info(f"Модель '{remote_model.title}' анализа '{remote_analysis_name}' загружена на локальную БД", 'green')

        update_list_trained_models_regmod()
        set_info('Загрузка данных с удаленной БД на локальную завершена', 'blue')


    def delete_reg_model_rdb():
        """ Удаление регрессионной модели в удаленной БД """
        title_model = ui_rdb.comboBox_trained_model_reg_rdb.currentText().split(' id')[0]
        title_analysis = ui_rdb.comboBox_regmod_rdb.currentText().split(' id')[0]
        analysis_id = get_regmod_rdb_id()

        result = QtWidgets.QMessageBox.question(
            RemoteDB,
            'Delete trained model in RemoteDB',
            f'Вы уверены, что хотите удалить модель "{title_model}" анализа "{title_analysis}"?',
            QtWidgets.QMessageBox.Yes,
            QtWidgets.QMessageBox.No
        )

        if result == QtWidgets.QMessageBox.Yes:
            with get_session() as remote_session:
                try:
                    remote_session.query(TrainedModelRegRDB) \
                        .filter(TrainedModelRegRDB.id == get_trained_model_reg_rdb_id()) \
                        .delete(synchronize_session=False)

                    remote_session.commit()
                    set_info(f'Модель "{title_model}" анализа "{title_analysis}" удалена в удаленной '
                             f'БД', 'green')
                    update_trained_models_reg_rdb()

                except Exception as e:
                    remote_session.rollback()
                    set_info(f'Ошибка при удалении: {str(e)}', 'red')

    #####################################################
    ######################  Wells  ######################
    #####################################################

    def sync_wells():
        sync_wells_func()
        update_list_well_rdb()
        ui_rdb.label_wells.setText(f'Wells: {ui_rdb.listWidget_wells.count()}')

    def start_unload_well_rel():
        unload_well_relations()
        if get_well_id_rdb():
            show_data_well_rdb()
            update_boundaries_rdb()

    def update_list_well_rdb():
        """Обновить виджет списка скважин"""
        ui_rdb.listWidget_wells.clear()
        with get_session() as remote_session:
            wells = remote_session.query(WellRDB).order_by(WellRDB.name).all()
            inactive_brush = QBrush(QColor(255, 230, 230))  # Заранее создаем кисть

            for w in wells:
                item = QListWidgetItem(f'скв.№ {w.name} id{w.id}')

                if w.ignore:  # Проверяем флаг ignore
                    item.setBackground(inactive_brush)
                    item.setForeground(QColor(150, 150, 150))
                    item.setToolTip("Неактивная скважина")

                ui_rdb.listWidget_wells.addItem(item)

    def get_well_id_rdb():
        if ui_rdb.listWidget_wells.currentItem():
            return ui_rdb.listWidget_wells.currentItem().text().split(' id')[-1]

    def filter_wells():
        """Фильтрация с подсветкой и автопрокруткой к первому совпадению"""
        search_text = ui_rdb.lineEdit_well_search.text().lower().strip()
        first_match = None  # Для хранения первого совпадения

        ui_rdb.listWidget_wells.setUpdatesEnabled(False)
        try:
            highlight_brush = QBrush(QColor(230, 255, 230))  # Cветло-зеленый фон
            default_brush = QBrush(QColor(255, 255, 255))  # Белый фон

            for i in range(ui_rdb.listWidget_wells.count()):
                item = ui_rdb.listWidget_wells.item(i)
                item_text = item.text().lower()
                matches = search_text in item_text if search_text else False

                # Визуальное выделение
                item.setBackground(highlight_brush if matches else default_brush)

                # Запоминаем первое совпадение
                if matches and first_match is None:
                    first_match = item

            # Прокручиваем к первому совпадению
            if first_match:
                ui_rdb.listWidget_wells.scrollToItem(
                    first_match,
                    QListWidget.PositionAtTop  # Прокрутить чтобы элемент был сверху
                )
                ui_rdb.listWidget_wells.setCurrentItem(first_match)  # Опционально: выделить элемент
        finally:
            ui_rdb.listWidget_wells.setUpdatesEnabled(True)

    def ignore_activate_well():
        w_id = get_well_id_rdb()
        if not w_id:
            set_info('Скважина не выбрана', 'red')
            QMessageBox.critical(RemoteDB, 'Ошибка', 'Скважина не выбрана')
        else:
            with get_session() as remote_session:
                well = remote_session.query(WellRDB).filter_by(id=w_id).first()
                if well.ignore == True:
                   well.ignore = None
                   remote_session.commit()
                   set_info(f'Скважина id{w_id} помечена как активная в удаленной БД', 'blue')
                else:
                    well.ignore = True
                    remote_session.commit()
                    set_info(f'Скважина id{w_id} помечена как неативная в удаленной БД', 'blue')
            update_list_well_rdb()
            ui_rdb.textEdit_wells_data.clear()
            ui_rdb.listWidget_boundary.clear()

    def show_data_well_rdb():
        ui_rdb.textEdit_wells_data.clear()
        with get_session() as remote_session:
            well = remote_session.query(WellRDB).filter_by(id=get_well_id_rdb()).first()
            if not well:
                return

            count_well_log = remote_session.query(WellLogRDB).filter_by(well_id=well.id).count()

            text_content = []
            if count_well_log > 0:
                text_content.append(
                    f'<p style="background-color:#ADFCDF">'
                    f'<b>Количество каротажных кривых:</b> {count_well_log}</p>'
                )

            text_content.extend([
                f'<p><b>Скважина №</b> {well.name}</p>',
                f'<p><b>X:</b> {well.x_coord}</p>',
                f'<p><b>Y:</b> {well.y_coord}</p>',
                f'<p><b>Альтитуда:</b> {well.alt} м.</p>'
            ])

            for opt in remote_session.query(WellOptionallyRDB).filter_by(well_id=well.id):
                text_content.append(f'<p><b>{opt.option}:</b> {opt.value}</p>')

            ui_rdb.textEdit_wells_data.setHtml(''.join(text_content))

    def update_boundaries_rdb():
        ui_rdb.listWidget_boundary.clear()
        with get_session() as remote_session:
            boundaries = remote_session.query(BoundaryRDB).filter(BoundaryRDB.well_id == get_well_id_rdb()).order_by(
                BoundaryRDB.depth).all()
            for b in boundaries:
                ui_rdb.listWidget_boundary.addItem(f'{b.title} - {b.depth}m. id{b.id}')

    def remove_duplicate_wells():
        with get_session() as remote_session:
            deduplicate_wells(remote_session)
        ui_rdb.label_wells.setText(f'Wells: {ui_rdb.listWidget_wells.count()}')
        update_list_well_rdb()



    def sync_entropy_features():
        load_entropy_feature(RemoteDB)
        unload_entropy_feature(RemoteDB)

    def sync_entropy_features_profile():
        load_entropy_feature_profile(RemoteDB)
        unload_entropy_feature_profile(RemoteDB)



    update_mlp_rdb_combobox()
    update_regmod_rdb_combobox()
    update_list_well_rdb()
    ui_rdb.checkBox_dont_check_dependencies.setChecked(False)
    ui_rdb.checkBox_dont_check_reg_dependencies.setChecked(False)
    ui_rdb.pushButton_load_obj_rem.clicked.connect(load_object_rem)
    ui_rdb.pushButton_unload_obj_rem.clicked.connect(unload_object_rem)
    ui_rdb.pushButton_delete_obj_rem.clicked.connect(delete_object_rem)
    ui_rdb.pushButton_sync_objects.clicked.connect(sync_all_objects)
    ui_rdb.pushButton_sync_wells.clicked.connect(sync_wells)
    ui_rdb.pushButton_load_well_rel.clicked.connect(load_well_relations)
    ui_rdb.pushButton_unload_well_rel.clicked.connect(start_unload_well_rel)
    ui_rdb.pushButton_load_formations.clicked.connect(load_formations)
    ui_rdb.pushButton_unload_formations.clicked.connect(unload_formations)
    ui_rdb.pushButton_unload_mlp.clicked.connect(unload_mlp)
    ui_rdb.pushButton_load_mlp.clicked.connect(load_mlp)
    ui_rdb.pushButton_delete_mlp_rdb.clicked.connect(delete_mlp_rdb)
    ui_rdb.pushButton_sync_ga_cls.clicked.connect(start_sync_ga_cls)
    ui_rdb.pushButton_unload_model.clicked.connect(start_unload_mlp_model)
    ui_rdb.pushButton_load_model.clicked.connect(load_cls_models_func)
    ui_rdb.pushButton_delete_model_rdb.clicked.connect(delete_cls_model_rdb)
    ui_rdb.pushButton_unload_regmod.clicked.connect(unload_regmod)
    ui_rdb.pushButton_delete_regmod_rdb.clicked.connect(delete_regmod_rdb)
    ui_rdb.pushButton_load_regmod.clicked.connect(load_regmod)
    ui_rdb.pushButton_sync_ga_reg.clicked.connect(start_sync_ga_reg)
    ui_rdb.pushButton_unload_model_reg.clicked.connect(start_unload_reg_model)
    ui_rdb.pushButton_load_model_reg.clicked.connect(load_reg_models_func)
    ui_rdb.pushButton_delete_model_reg_rdb.clicked.connect(delete_reg_model_rdb)
    ui_rdb.label_wells.setText(f'Wells: {ui_rdb.listWidget_wells.count()}')
    ui_rdb.pushButton_ignore_well_rdb.clicked.connect(ignore_activate_well)
    ui_rdb.listWidget_wells.currentItemChanged.connect(show_data_well_rdb)
    ui_rdb.listWidget_wells.currentItemChanged.connect(update_boundaries_rdb)
    ui_rdb.lineEdit_well_search.setPlaceholderText("Поиск скважины...")
    ui_rdb.lineEdit_well_search.textChanged.connect(filter_wells)
    ui_rdb.pushButton_sync_entropy_profile.clicked.connect(sync_entropy_features_profile)
    ui_rdb.pushButton_remove_dupl_rdb.clicked.connect(remove_duplicate_wells)

    #####################################################
    #####################  Geochem  #####################
    #####################################################

    def get_geochem_rdb_id():
        """ Получение id выбранного объекта геохимических исследований """
        try:
            return int(ui_rdb.comboBox_geochem_rdb.currentText().split(' id')[-1])
        except ValueError:
            pass

    def get_g_well_rdb_id():
        """ Получение id выбранной скважины объекта геохимических исследований """
        try:
            return int(ui_rdb.comboBox_g_wells.currentText().split(' id')[-1])
        except ValueError:
            pass

    def update_geochem_rdb_combobox():
        """ Обновление списка объектов геохимических данных в выпадающем списке """

        # Очистка выпадающего списка объектов
        ui_rdb.comboBox_geochem_rdb.clear()

        # Получение всех объектов из базы данных, отсортированных по дате исследования
        with get_session() as remote_session:
            try:
                # Добавление названия объекта в выпадающий список
                geochems = remote_session.query(GeochemRDB).order_by(GeochemRDB.title).all()
            except ValueError:
                return
            for g in geochems:
                ui_rdb.comboBox_geochem_rdb.addItem(f'{g.title} id{g.id}')
        update_g_well_rdb_combobox()
        update_g_points_rdb()
        update_g_params_rdb()
        update_mask_maket_labels()

    def update_g_well_rdb_combobox():
        """ Обновление списка скважин объекта геохимических данных """
        # Очистка выпадающего списка скважин
        ui_rdb.comboBox_g_wells.clear()
        with get_session() as remote_session:
            try:
                g_wells = remote_session.query(GeochemWellRDB.id, GeochemWellRDB.title).filter(
                    GeochemWellRDB.geochem_id == get_geochem_rdb_id()
                ).order_by(GeochemWellRDB.title).all()
            except ValueError:
                return
            # Запрос на получение всех скважин, относящихся к объекту, и их добавление в выпадающий список
            for i in g_wells:
                ui_rdb.comboBox_g_wells.addItem(f'{i[1]} id{i[0]}')
        ui_rdb.label_g_wells.setText(f'Wells: {ui_rdb.comboBox_g_wells.count()}')
        update_g_well_points_rdb()

    def update_g_points_rdb():
        ui_rdb.listWidget_g_points.clear()
        with get_session() as remote_session:
            try:
                g_points = remote_session.query(GeochemPointRDB).filter(
                    GeochemPointRDB.geochem_id == get_geochem_rdb_id()
                ).order_by(GeochemPointRDB.title).all()
            except ValueError:
                return
            g_p_values_count = 0
            for p in g_points:
                ui_rdb.listWidget_g_points.addItem(f'{p.title} id{p.id}')
                g_p_values = remote_session.query(GeochemPointValueRDB).filter(
                    GeochemPointValueRDB.g_point_id == p.id
                ).count()
                g_p_values_count += g_p_values
            ui_rdb.label_g_p_values.setText(f'Points Values: {g_p_values_count}')
        ui_rdb.label_g_points.setText(f'Field Points: {ui_rdb.listWidget_g_points.count()}')

    def update_g_well_points_rdb():
        ui_rdb.listWidget_g_well_points.clear()
        with get_session() as remote_session:
            try:
                g_well_points = remote_session.query(GeochemWellPointRDB).filter(
                    GeochemWellPointRDB.g_well_id == get_g_well_rdb_id()
                ).order_by(GeochemWellPointRDB.title).all()
            except ValueError:
                return
            g_w_p_values_count = 0
            for w_p in g_well_points:
                ui_rdb.listWidget_g_well_points.addItem(f'{w_p.title} id{w_p.id}')
                g_w_p_values = remote_session.query(GeochemWellPointValueRDB).filter(
                    GeochemWellPointValueRDB.g_well_point_id == w_p.id
                ).count()
                g_w_p_values_count += g_w_p_values
            ui_rdb.label_g_w_p_values.setText(f'Well Points Values: {g_w_p_values_count}')

        ui_rdb.label_g_well_points.setText(f'Well Points: {ui_rdb.listWidget_g_well_points.count()}')

    def update_g_params_rdb():
        ui_rdb.listWidget_g_params.clear()
        with get_session() as remote_session:
            try:
                g_params = remote_session.query(GeochemParameterRDB).filter(
                    GeochemParameterRDB.geochem_id == get_geochem_rdb_id()
                ).order_by(GeochemParameterRDB.title).all()
            except ValueError:
                return
            for p in g_params:
                ui_rdb.listWidget_g_params.addItem(f'{p.title} id{p.id}')
        ui_rdb.label_g_params.setText(f'Parameters: {ui_rdb.listWidget_g_params.count()}')

    def update_mask_maket_labels():
        with get_session() as remote_session:
            # Обновление количества масок
            masks = remote_session.query(GeochemMaskRDB).filter(
                GeochemMaskRDB.geochem_id == get_geochem_rdb_id()
            ).count()
            ui_rdb.label_g_masks.setText(f'Masks: {masks}')

            # Обновление количества макетов
            makets = remote_session.query(GeochemMaketRDB).filter(
                GeochemMaketRDB.geochem_id == get_geochem_rdb_id()
            ).count()
            ui_rdb.label_g_makets.setText(f'Makets: {makets}')

    def load_geochem_rdb():
        """ Загрузка геохимических данных с удаленной БД на локальную """

        with get_session() as remote_session:
            # Получаем выбранный объект из удаленной базы
            remote_geochems = remote_session.query(GeochemRDB).filter(GeochemRDB.id == get_geochem_rdb_id())

            loaded_geochem_count = 0

            for remote_geochem in tqdm(remote_geochems, desc='Загрузка объектов геохимических данных'):
                set_info(f'Загрузка объекта геохимических данных "{remote_geochem.title}"...', 'blue')

                # Проверяем существование объекта в локальной базе
                local_geochem = session.query(Geochem).filter_by(title = remote_geochem.title).first()

                if not local_geochem:
                    # Добавляем отстутствующий объект
                    new_geochem = Geochem(title=remote_geochem.title)
                    session.add(new_geochem)
                    session.commit()
                    local_geochem = new_geochem
                    set_info(f'Объект загружен с удаленной БД', 'green')
                    loaded_geochem_count += 1
                else:
                    set_info(f'Объект уже есть в локальной БД', 'red')

                set_info("Загрузка связанных геохимических данных...", 'blue')

                # Получаем все параметры для текущего объекта из удаленной базы
                remote_parameters = remote_session.query(GeochemParameterRDB).filter_by(
                    geochem_id=remote_geochem.id).all()

                loaded_params_count = 0

                ui.progressBar.setMaximum(len(remote_parameters))
                for n, remote_parameter in enumerate(remote_parameters):
                    ui.progressBar.setValue(n+1)

                    # Проверяем существование параметра в локальной базе
                    local_parameter = session.query(GeochemParameter).filter_by(
                        geochem_id=local_geochem.id,
                        title=remote_parameter.title
                    ).first()

                    if not local_parameter:
                        # Добавляем отсутствующий параметр
                        new_parameter = GeochemParameter(
                            geochem_id=local_geochem.id,
                            title=remote_parameter.title
                        )
                        session.add(new_parameter)
                        loaded_params_count += 1

                session.commit()
                if loaded_params_count > 0:
                    added_word = "Загружена" if loaded_params_count == 1 else "Загружено"
                    set_info(f'{added_word} {pluralize((loaded_params_count), ["параметр", "параметра", "параметров"])}', 'green')
                else:
                    set_info('Все параметры объекта уже есть в локальной БД', 'red')

                # Словарь параметров локальной БД
                local_params_dict = {
                    p.title: p.id
                    for p in session.query(GeochemParameter.id, GeochemParameter.title)
                    .filter_by(geochem_id=local_geochem.id)
                    .all()
                }

                remote_params_dict = {p.id: p.title for p in remote_parameters}

                # Получаем все точки для текущего объекта из удаленной базы
                remote_points = remote_session.query(GeochemPointRDB).filter_by(geochem_id=remote_geochem.id).all()

                loaded_points_count = 0
                loaded_p_values_count = 0

                ui.progressBar.setMaximum(len(remote_points))
                for n, remote_point in enumerate(remote_points):
                    ui.progressBar.setValue(n+1)

                    # Проверяем существование точки в локальной базе
                    local_point = session.query(GeochemPoint).filter_by(
                        geochem_id=local_geochem.id,
                        title=remote_point.title
                    ).first()

                    if not local_point:
                        # Добавляем отсутствующую точку
                        new_point = GeochemPoint(
                            geochem_id=local_geochem.id,
                            title=remote_point.title,
                            x_coord=remote_point.x_coord,
                            y_coord=remote_point.y_coord,
                            fake=remote_point.fake,
                        )
                        session.add(new_point)
                        local_point = new_point
                        loaded_points_count += 1
                        session.flush()

                    # Получаем все значения геохимических параметров в точке
                    remote_p_values = remote_session.query(GeochemPointValueRDB).filter_by(g_point_id=remote_point.id).all()

                    existing_p_values_set = set()
                    if local_point:
                        existing_p_values = session.query(
                            GeochemPointValue.g_point_id,
                            GeochemPointValue.g_param_id
                        ).filter_by(g_point_id=local_point.id).all()
                        existing_p_values_set = {(p_v.g_point_id, p_v.g_param_id) for p_v in existing_p_values}

                    for remote_p_value in remote_p_values:
                        # Получаем заголовок параметра через словарь
                        param_title = remote_params_dict.get(remote_p_value.g_param_id)

                        if param_title:
                            local_param_id = local_params_dict.get(param_title)

                            if local_param_id:
                                key = (local_point.id, local_param_id)
                                if key not in existing_p_values_set:
                                    # Добавляем отсутствующее значение
                                    new_p_value = GeochemPointValue(
                                        g_point_id=local_point.id,
                                        g_param_id=local_param_id,
                                        value=remote_p_value.value
                                    )
                                    session.add(new_p_value)
                                    loaded_p_values_count += 1
                                    existing_p_values_set.add(key)

                session.commit()
                if loaded_points_count > 0:
                    added_word_point = "Загружена" if loaded_points_count == 1 else "Загружено"
                    set_info(f'{added_word_point} {pluralize((loaded_points_count), ["точка", "точки", "точек"])}', 'green')
                else:
                    set_info('Все точки объекта уже есть в удаленной БД', 'red')

                if loaded_p_values_count > 0:
                    added_word_p_value = "Загружена" if loaded_p_values_count == 1 else "Загружено"
                    set_info(f'{added_word_p_value} '
                             f'{pluralize((loaded_p_values_count), ["значение параметров в точках", "значения параметров в точках", "значений параметров в точках"])}',
                             'green')
                else:
                    set_info('Все значения параметров в точках объекта уже есть в локальной БД', 'red')

                # Получаем все маски для текущего объекта из удаленной базы
                remote_masks = remote_session.query(GeochemMaskRDB).filter_by(geochem_id=remote_geochem.id).all()

                loaded_masks_count = 0

                ui.progressBar.setMaximum(len(remote_masks))
                for n, remote_mask in enumerate(remote_masks):
                    ui.progressBar.setValue(n+1)

                    # Проверяем существование маски в локальной базе
                    local_mask = session.query(GeochemMask).filter_by(
                        geochem_id=local_geochem.id,
                        count_param=remote_mask.count_param,
                        count_points=remote_mask.count_points
                    ).first()

                    if not local_mask:
                        # Добавляем отсутсвующую маску
                        new_mask = GeochemMask(
                            geochem_id=local_geochem.id,
                            count_param=remote_mask.count_param,
                            count_points=remote_mask.count_points,
                            mask_param=remote_mask.mask_param,
                            mask_point=remote_mask.mask_point,
                            mask_info=remote_mask.mask_info
                        )
                        session.add(new_mask)
                        loaded_masks_count += 1

                session.commit()
                if loaded_masks_count > 0:
                    added_word = "Загружена" if loaded_masks_count == 1 else "Загружено"
                    set_info(f'{added_word} {pluralize((loaded_masks_count), ["маска", "маски", "масок"])}', 'green')
                else:
                    set_info('Все маски объекта уже есть в локальной БД', 'red')

                # Получаем все точки для текущего объекта из удаленной базы
                remote_wells = remote_session.query(GeochemWellRDB).filter_by(geochem_id=remote_geochem.id).all()

                loaded_wells_count = 0
                loaded_well_points_count = 0
                loaded_w_p_values_count = 0

                ui.progressBar.setMaximum(len(remote_wells))
                # Получаем все скважины для текущего объекта из удаленной базы
                for n, remote_well in enumerate(remote_wells):
                    ui.progressBar.setValue(n+1)

                    # Проверяем существование скважин в локальной базе
                    local_well = session.query(GeochemWell).filter_by(
                        geochem_id=local_geochem.id,
                        title=remote_well.title
                    ).first()

                    if not local_well:
                        # Добавляем отсутствующую скважину
                        new_well = GeochemWell(
                            geochem_id=local_geochem.id,
                            title=remote_well.title,
                            color=remote_well.color
                        )
                        session.add(new_well)
                        loaded_wells_count += 1
                        local_well = new_well
                        session.flush()

                    # Получаем все точки скважины из удаленной базы
                    remote_well_points = remote_session.query(GeochemWellPointRDB).filter_by(g_well_id=remote_well.id).all()

                    for remote_well_point in remote_well_points:

                        # Проверяем существование точек скважины в локальной базе
                        local_well_point = session.query(GeochemWellPoint).filter_by(
                            g_well_id=local_well.id,
                            title=remote_well_point.title
                        ).first()

                        if not local_well_point:
                            # Добавляем отсутствующий точки скважины
                            new_well_point = GeochemWellPoint(
                                g_well_id=local_well.id,
                                title=remote_well_point.title,
                                x_coord=remote_well_point.x_coord,
                                y_coord=remote_well_point.y_coord
                            )
                            session.add(new_well_point)
                            local_well_point = new_well_point
                            loaded_well_points_count += 1
                            session.flush()

                        # Получаем все значения геохимических параметров в точке скважины
                        remote_w_p_values = remote_session.query(GeochemWellPointValueRDB).filter_by(
                            g_well_point_id=remote_well_point.id).all()

                        # Создаем множество сущетсвующих значений для проверки дубликатов
                        existing_w_p_values_set = set()
                        if local_well_point:
                            existing_w_p_values = session.query(
                                GeochemWellPointValue.g_well_point_id,
                                GeochemWellPointValue.g_param_id
                            ).filter_by(g_well_point_id=local_well_point.id).all()
                            existing_w_p_values_set = {(w_p_v.g_well_point_id, w_p_v.g_param_id) for w_p_v in existing_w_p_values}

                        for remote_w_p_value in remote_w_p_values:
                            # Получаем заголовок парамера через словарь
                            param_title = remote_params_dict.get(remote_w_p_value.g_param_id)

                            if param_title:
                                local_param_id = local_params_dict.get(param_title)

                                if local_param_id:
                                    # Проверяем существование значения в локальной базе
                                    key = (local_well_point.id, local_param_id)
                                    if key not in existing_w_p_values_set:
                                        # Добавляем отсутствующее значение
                                        new_w_p_value = GeochemWellPointValue(
                                            g_well_point_id=local_well_point.id,
                                            g_param_id=local_param_id,
                                            value=remote_w_p_value.value
                                        )
                                        session.add(new_w_p_value)
                                        loaded_w_p_values_count += 1
                                        existing_w_p_values_set.add(key)

                session.commit()
                if loaded_wells_count > 0:
                    added_word_well = "Загружена" if loaded_wells_count == 1 else "Загружено"
                    set_info(
                        f'{added_word_well} '
                                f'{pluralize((loaded_wells_count), ["скважина", "скважины", "скважин"])}',
                            'green')
                else:
                    set_info('Все скважины объекта уже есть в локальной БД', 'red')
                if loaded_well_points_count > 0:
                    added_word = "Загружена" if loaded_well_points_count == 1 else "Загружено"
                    set_info(
                        f'{added_word} '
                        f'{pluralize((loaded_well_points_count), ["точка скважины", "точки скважины", "точек скважины"])} "{local_well.title}"',
                        'green')
                else:
                    set_info(f'Все точки скважин объекта уже есть в локальной БД', 'red')

                if loaded_w_p_values_count > 0:
                    added_word_p_value = "Загружена" if loaded_w_p_values_count == 1 else "Загружено"
                    set_info(
                        f'{added_word_p_value} '
                        f'{pluralize((loaded_w_p_values_count), ["значение параметров в точках скважины", "значения параметров в точках скважины", "значений параметров в точках скважины"])}'
                        f'"{local_well.title}"', 'green')
                else:
                    set_info(f'Все значения параметров в точках скважин объекта уже есть в локальной '
                             f'БД', 'red')


                # Получаем все макеты для текущего объекта из удаленной базы
                remote_makets = remote_session.query(GeochemMaketRDB).filter_by(
                    geochem_id=remote_geochem.id).all()

                loaded_makets_count = 0

                ui.progressBar.setMaximum(len(remote_makets))
                for n, remote_maket in enumerate(remote_makets):
                    ui.progressBar.setValue(n+1)

                    # Проверяем существование макета в локальной базе
                    local_maket = session.query(GeochemMaket).filter_by(
                        geochem_id=local_geochem.id,
                        title=remote_maket.title
                    ).first()

                    if not local_maket:
                        # Добавляем отсутствующий макет
                        new_maket = GeochemMaket(
                            geochem_id=local_geochem.id,
                            title=remote_maket.title
                        )
                        session.add(new_maket)
                        loaded_makets_count += 1
                session.commit()
                if loaded_makets_count > 0:
                    added_word = "Загружен" if loaded_makets_count == 1 else "Загружено"
                    set_info(
                        f'{added_word} '
                        f'{pluralize((loaded_makets_count), ["макет", "макета", "макетов"])}',
                        'green')
                else:
                    set_info('Все макеты объекта уже есть в локальной БД', 'red')

        update_combobox_geochem()
        set_info(f'Загрузка данных с удаленной БД на локальную завершена', 'blue')

    def unload_geochem_rdb():
        """ Выгрузка геохимических данных с локальной БД на удаленную """
        unload_geochem_func()
        update_geochem_rdb_combobox()


    def sync_all_geochem():
        """ Синхронизация всех объектов геохимии и связанных с ними данных """
        try:
            with get_session() as remote_session:
                set_info('Начало синхронизации...', 'blue')

                # Синхронизация объектов (удаленная -> локальная)
                set_info(f'Обновление геохимических данных в локальной БД...', 'blue')
                sync_geochem_direction(remote_session, session,
                                       GeochemRDB, GeochemParameterRDB, GeochemPointRDB, GeochemMaskRDB,
                                       GeochemWellRDB, GeochemWellPointRDB, GeochemPointValueRDB,
                                       GeochemWellPointValueRDB, GeochemMaketRDB,
                                       Geochem, GeochemParameter, GeochemPoint, GeochemMask,
                                       GeochemWell, GeochemWellPoint, GeochemPointValue,
                                       GeochemWellPointValue, GeochemMaket)

                update_combobox_geochem()
                set_info(f'Обновление геохимических в локальной БД завершено', 'blue')

                # Синхронизация объектов (локальная -> удаленная)
                set_info(f'Обновление геохимических данных в удаленной БД...', 'blue')
                sync_geochem_direction(session, remote_session,
                                       Geochem, GeochemParameter, GeochemPoint, GeochemMask,
                                       GeochemWell, GeochemWellPoint, GeochemPointValue,
                                       GeochemWellPointValue, GeochemMaket,
                                       GeochemRDB, GeochemParameterRDB, GeochemPointRDB, GeochemMaskRDB,
                                       GeochemWellRDB, GeochemWellPointRDB, GeochemPointValueRDB,
                                       GeochemWellPointValueRDB, GeochemMaketRDB)
                update_geochem_rdb_combobox()
                set_info(f'Обновление геохимических данных в удаленной БД завершено', 'blue')

                set_info('Синхронизация завершена', 'blue')

        except Exception as e:
            # Откат изменений в случае ошибки
            session.rollback()
            set_info(f'Синхронизация прервалась: {str(e)}', 'red')
            raise # Проброс исключения для дальнейшей обработки
        finally:
            session.close()


    def delete_geochem_rdb():
        """ Удаление текущего объекта вместе со всеми связанными данными """
        title_geochem = ui_rdb.comboBox_geochem_rdb.currentText().split('id')[0]
        geochem_id = get_geochem_rdb_id()

        result = QtWidgets.QMessageBox.question(
            RemoteDB,
            'Delete geochem in RemoteDB',
            f'Вы уверены, что хотите удалить объект "{title_geochem}" со всеми обучающими скважинами?',
            QtWidgets.QMessageBox.Yes,
            QtWidgets.QMessageBox.No
        )

        if result == QtWidgets.QMessageBox.Yes:
            with get_session() as remote_session:
                try:
                    # Удаляем все маски объекта
                    remote_session.query(GeochemMaskRDB) \
                        .filter(GeochemMaskRDB.geochem_id == geochem_id) \
                        .delete()

                    # Удаляем все макеты объекта
                    remote_session.query(GeochemMaketRDB) \
                        .filter(GeochemMaketRDB.geochem_id == geochem_id) \
                        .delete()

                    # Удаляем все значения параметров в точках
                    remote_session.query(GeochemPointValueRDB) \
                        .filter(GeochemPointValueRDB.g_point_id == GeochemPointRDB.id) \
                        .filter(GeochemPointRDB.geochem_id == geochem_id) \
                        .delete()

                    # Удаляем все значения параметров в точках скважин
                    remote_session.query(GeochemWellPointValueRDB) \
                        .filter(GeochemWellPointValueRDB.g_well_point_id == GeochemWellPointRDB.id) \
                        .filter(GeochemWellPointRDB.g_well_id == GeochemWellRDB.id) \
                        .filter(GeochemWellRDB.geochem_id == geochem_id) \
                        .delete()

                    # Удаляем все точки скважин объекта
                    remote_session.query(GeochemWellPointRDB) \
                        .filter(GeochemWellPointRDB.g_well_id == GeochemWellRDB.id) \
                        .filter(GeochemWellRDB.geochem_id == geochem_id) \
                        .delete()

                    # Удаляем все скважины объекта
                    remote_session.query(GeochemWellRDB) \
                        .filter(GeochemWellRDB.geochem_id == geochem_id) \
                        .delete()

                    # Удаляем все точки объекта
                    remote_session.query(GeochemPointRDB) \
                        .filter(GeochemPointRDB.geochem_id == geochem_id) \
                        .delete()

                    # Удаляем все параметры объекта
                    remote_session.query(GeochemParameterRDB) \
                        .filter(GeochemParameterRDB.geochem_id == geochem_id) \
                        .delete()

                    # Удаляем объект геохимических данных
                    remote_session.query(GeochemRDB) \
                        .filter(GeochemRDB.id == geochem_id) \
                        .delete()

                    remote_session.commit()
                    set_info(f'Объект геохимических данных "{title_geochem}" удален в удаленной БД', 'green')
                    update_geochem_rdb_combobox()

                except Exception as e:
                    remote_session.rollback()
                    set_info(f'Ошибка при удалении: {str(e)}', 'red')


    update_geochem_rdb_combobox()
    ui_rdb.pushButton_load_geochem.clicked.connect(load_geochem_rdb)
    ui_rdb.pushButton_unload_geochem.clicked.connect(unload_geochem_rdb)
    ui_rdb.pushButton_delete_geochem_rdb.clicked.connect(delete_geochem_rdb)
    ui_rdb.comboBox_geochem_rdb.currentIndexChanged.connect(update_g_well_rdb_combobox)
    ui_rdb.comboBox_geochem_rdb.currentIndexChanged.connect(update_g_params_rdb)
    ui_rdb.comboBox_geochem_rdb.currentIndexChanged.connect(update_g_points_rdb)
    ui_rdb.comboBox_geochem_rdb.currentIndexChanged.connect(update_mask_maket_labels)
    ui_rdb.comboBox_g_wells.currentIndexChanged.connect(update_g_well_points_rdb)
    ui_rdb.pushButton_sync_all_geochem.clicked.connect(sync_all_geochem)


    RemoteDB.exec_()