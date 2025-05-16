from remote_db.model_remote_db import *
from models_db.model import *
from func import *

def get_trained_model_class_id():
    """ Получение ID текущей выбранной модели """
    current_item = ui.listWidget_trained_model_class.currentItem()
    if current_item is not None:
        return current_item.data(Qt.UserRole)
    return None

def unload_cls_models_func(RemoteDB):
    set_info('Начало выгрузки данных с локальной БД на удаленную', 'blue')

    model_id = get_trained_model_class_id()
    if model_id is None:
        set_info("Модель не выбрана", "red")
        return

    with get_session() as remote_session:

        local_analysis_name = session.query(AnalysisMLP.title).filter_by(id=get_MLP_id()).first()[0]
        remote_analysis = remote_session.query(AnalysisMLPRDB).filter_by(title=local_analysis_name).first()
        if not remote_analysis:
            set_info(f"Соответствующий анализ '{local_analysis_name}' не найден в локальной БД. Сначала выгрузите или "
                     f"создайте анализ.", "red")
            return

        local_model = session.query(TrainedModelClass).filter(
            TrainedModelClass.analysis_id == get_MLP_id(),
            TrainedModelClass.id == model_id).first()

        local_mask = session.query(TrainedModelClassMask).filter_by(model_id=model_id).first()

        if local_mask:
            param_mask = session.query(ParameterMask).filter_by(id=local_mask.mask_id).first()

        remote_model = remote_session.query(TrainedModelClassRDB).filter_by(
            analysis_id=remote_analysis.id,
            title=local_model.title,
            list_params=local_model.list_params,
            except_signal=local_model.except_signal,
            except_crl=local_model.except_crl,
            comment=local_model.comment,
            mask=param_mask.mask if local_mask else ''
        ).first()

        if remote_model:
            set_info(f"Модель '{local_model.title}' анализа '{local_analysis_name}' есть в удаленной БД", 'red')
            QMessageBox.critical(RemoteDB, 'Error',
                                 f"Модель '{local_model.title}' анализа '{local_analysis_name}' есть в удаленной БД")
        else:
            with open(local_model.path_model, "rb") as f:
                data = pickle.load(f)
            data_model = pickle.dumps(data)
            new_remote_model = TrainedModelClassRDB(
                analysis_id=remote_analysis.id,
                title=local_model.title,
                file_model=data_model,
                list_params=local_model.list_params,
                except_signal=local_model.except_signal,
                except_crl=local_model.except_crl,
                comment=local_model.comment,
                mask=param_mask.mask if local_mask else ''
            )
            remote_session.add(new_remote_model)
            remote_session.commit()
            set_info(f"Модель '{local_model.title}' анализа '{local_analysis_name}' выгружена на удаленную БД", 'green')

    set_info('Выгрузка данных с локальной БД на удаленную завершена', 'blue')


