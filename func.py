from object import *


def set_info(text, color):
    ui.info.append(f'<span style =\"color:{color};\" >{text}</span>')


def get_object_id():
    try:
        return int(ui.comboBox_object.currentText().split('id')[-1])
    except ValueError:
        pass


def get_object_name():
    return ui.comboBox_object.currentText().split(' id')[0]


def get_profile_id():
    return int(ui.comboBox_profile.currentText().split('id')[-1])


def query_to_list(query):
    """ результаты запроса в список """
    return sum(list(map(list, query)), [])

