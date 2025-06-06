import pickle

from remote_db.model_remote_db import *
from models_db.model import *
from func import *

def update_checkfile(remote_ga, local_ga, GeneticAlgorithmRDB):
    """Обновление файла генетического анализа"""
    with open(local_ga.checkfile_path, "rb") as f:
        local_data = pickle.load(f)

    remote_data = pickle.loads(remote_ga.checkfile_path)

    if local_data == remote_data:
        set_info(f"Файлы для {local_ga.title} совпадают", 'blue')
        return

    set_info(f'Обновление файла {local_ga.title}...', 'blue')
    problem = Problem(len(json.loads(local_ga.list_params)), 1 if local_ga.type_problem == "no" else 2)

    # создаем отдельный объект Binary для каждой переменной
    for i in range(len(json.loads(local_ga.list_params))):
        problem.types[i] = Binary(1)  # Указываем размерность 1 для каждой переменной

    problem.directions[0] = Problem.MAXIMIZE  # Максимизация средней accuracy
    if local_ga.type_problem == "min":
        problem.directions[1] = Problem.MINIMIZE  # Минимизация числа признаков
    elif local_ga.type_problem == "max":
        problem.directions[1] = Problem.MAXIMIZE
    else:
        pass

    all_pop, nfe_list = [], []

    for check_data in [local_data, remote_data]:
        pop, nfe = _read_pop(check_data, problem)
        all_pop.extend(pop)
        nfe_list.append(nfe)

    front = select_best(all_pop, local_ga.population_size)
    new_nfe = max(nfe_list)

    _write_pop(front, new_nfe, local_ga, remote_ga, GeneticAlgorithmRDB)
    print(f"Записан объединённый чек‑пойнт "
          f"({len(front)} решений, nfe={new_nfe})")

    set_info(f'Обновление файла {local_ga.title} выполнено', 'green')

def _read_pop(check_data, problem):
    pop = []
    for x, fobj in zip(check_data["X"], check_data["F"]):
        s = Solution(problem)
        s.variables[:] = x
        if problem.nobjs == 1:
            # Одноцелевой: fobj — это скаляр
            s.objectives[0] = fobj
        else:
            # Многоцелевой: fobj — это список
            s.objectives[:] = fobj
        s.evaluated = True
        pop.append(s)
    return pop, check_data["nfe"]

def _write_pop(pop, nfe, local_ga, remote_ga, GeneticAlgorithmRDB):
    data = dict(
        X=[s.variables[:] for s in pop],

        F=[
            s.objectives[0] if len(s.objectives) == 1 else s.objectives[:]
            for s in pop
        ],
        nfe=nfe,
        rng=random.getstate(),  # актуальное состояние ГСЧ
        ngen=nfe // len(pop) if len(pop) else 0
    )
    with open(local_ga.checkfile_path, "wb") as f:
        pickle.dump(data, f)
    with get_session() as remote_session:
        remote_session.query(GeneticAlgorithmRDB).filter_by(id=remote_ga.id).update({'checkfile_path': pickle.dumps(data)})

def select_best(population, k):
    """Возвратить k решений по рангу+crowding (NSGA‑II style)."""
    # Применяем nondominated_sort к популяции (функция модифицирует объекты)
    nondominated_sort(population)

    # Группируем решения по рангам
    ranks = {}
    for solution in population:
        if not hasattr(solution, 'rank'):
            print("Warning: solution does not have 'rank' attribute after nondominated_sort")
            continue

        rank = solution.rank
        if rank not in ranks:
            ranks[rank] = []
        ranks[rank].append(solution)

    # Теперь мы имеем словарь, где ключи - ранги, значения - списки решений
    selected = []

    # Обрабатываем ранги в порядке возрастания (сначала лучшие)
    for rank in sorted(ranks.keys()):
        front = ranks[rank]
        crowding_distance(front)  # нужно для сортировки
        front.sort(key=lambda s: -s.crowding_distance)

        space_left = k - len(selected)
        selected.extend(front[:space_left])  # добираем столько, сколько нужно
        if len(selected) >= k:
            break  # набрали k, выходим

    return selected


def sync_genetic_cls_func(ui_rdb):
    """ Синхронизация GeneticAlgorithmCLS """
    set_info('Начало синхронизации данных', 'blue')

    with get_session() as remote_session:

        local_gen_analyzes = session.query(GeneticAlgorithmCLS).filter(
            GeneticAlgorithmCLS.analysis_id == get_MLP_id()).all()

        local_analysis_name = session.query(AnalysisMLP.title).filter_by(id=get_MLP_id()).first()[0]

        try:

            remote_analysis_id = remote_session.query(AnalysisMLPRDB).filter_by(title=local_analysis_name).first().id

            for local_ga in tqdm(local_gen_analyzes, desc='Синхронизация генетического анализа'):

                if ui_rdb.checkBox_check_ga_params.isChecked():
                    remote_ga = remote_session.query(GeneticAlgorithmCLSRDB).filter_by(
                        analysis_id=remote_analysis_id,
                        title=local_ga.title,
                        pipeline=local_ga.pipeline,
                        population_size=local_ga.population_size,
                        type_problem=local_ga.type_problem
                    ).first()

                else:
                    remote_ga = remote_session.query(GeneticAlgorithmCLSRDB).filter_by(
                        analysis_id=remote_analysis_id,
                        title=local_ga.title,
                        pipeline=local_ga.pipeline,
                        list_params=local_ga.list_params,
                        population_size=local_ga.population_size,
                        type_problem=local_ga.type_problem
                    ).first()

                if remote_ga:
                    update_checkfile(remote_ga, local_ga, GeneticAlgorithmCLSRDB)
                else:
                    with open(local_ga.checkfile_path, "rb") as f:
                        data = pickle.load(f)
                    data_checkfile = pickle.dumps(data)
                    new_remote_ga = GeneticAlgorithmCLSRDB(
                        analysis_id=remote_analysis_id,
                        title=local_ga.title,
                        pipeline=local_ga.pipeline,
                        checkfile_path=data_checkfile,
                        list_params=local_ga.list_params,
                        population_size=local_ga.population_size,
                        comment=local_ga.comment,
                        type_problem=local_ga.type_problem
                    )
                    remote_session.add(new_remote_ga)
                    remote_session.commit()
                    set_info(f"Генетический анализ {local_ga.title} синхронизирован с удаленной БД", 'green')

        except AttributeError:
            set_info(f'Анализ "{local_analysis_name}" отсутствует в удаленной БД', 'red')
            QMessageBox.critical(MainWindow, 'Ошибка', f'Анализ "{local_analysis_name}" отсутствует в удаленной БД. '
                                                       f'\nДля синхронизации генетического анализа необходимо '
                                                       f'выгрузить его с локальной БД.')
            return

    set_info('Синхронизация данных завершена', 'blue')

def sync_genetic_reg_func(ui_rdb, Window):
    """ Синхронизация GeneticAlgorithmReg """
    set_info('Начало синхронизации данных', 'blue')

    with get_session() as remote_session:

        local_gen_analyzes = session.query(GeneticAlgorithmReg).filter(
            GeneticAlgorithmReg.analysis_id == get_regmod_id()).all()

        local_analysis_name = session.query(AnalysisReg.title).filter_by(id=get_regmod_id()).first()[0]

        try:

            remote_analysis_id = remote_session.query(AnalysisRegRDB).filter_by(title=local_analysis_name).first().id

            for local_ga in tqdm(local_gen_analyzes, desc='Синхронизация генетического анализа'):

                if ui_rdb.checkBox_check_ga_params_reg.isChecked():
                    remote_ga = remote_session.query(GeneticAlgorithmRegRDB).filter_by(
                        analysis_id=remote_analysis_id,
                        title=local_ga.title,
                        pipeline=local_ga.pipeline,
                        population_size=local_ga.population_size,
                        type_problem=local_ga.type_problem
                    ).first()

                else:
                    remote_ga = remote_session.query(GeneticAlgorithmRegRDB).filter_by(
                        analysis_id=remote_analysis_id,
                        title=local_ga.title,
                        pipeline=local_ga.pipeline,
                        list_params=local_ga.list_params,
                        population_size=local_ga.population_size,
                        type_problem=local_ga.type_problem
                    ).first()

                if remote_ga:
                    update_checkfile(remote_ga, local_ga, GeneticAlgorithmRegRDB)
                else:
                    with open(local_ga.checkfile_path, "rb") as f:
                        data = pickle.load(f)
                    data_checkfile = pickle.dumps(data)
                    new_remote_ga = GeneticAlgorithmRegRDB(
                        analysis_id=remote_analysis_id,
                        title=local_ga.title,
                        pipeline=local_ga.pipeline,
                        checkfile_path=data_checkfile,
                        list_params=local_ga.list_params,
                        population_size=local_ga.population_size,
                        comment=local_ga.comment,
                        type_problem=local_ga.type_problem
                    )
                    remote_session.add(new_remote_ga)
                    remote_session.commit()
                    set_info(f"Генетический анализ {local_ga.title} синхронизирован с удаленной БД", 'green')

        except AttributeError:
            set_info(f'Анализ "{local_analysis_name}" отсутствует в удаленной БД', 'red')
            QMessageBox.critical(Window, 'Ошибка', f'Анализ "{local_analysis_name}" отсутствует в удаленной БД. '
                                                       f'\nДля синхронизации генетического анализа необходимо '
                                                       f'выгрузить его с локальной БД.')
            return


    set_info('Синхронизация данных завершена', 'blue')