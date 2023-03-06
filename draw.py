from func import *


def choose_minmax():
    if ui.checkBox_minmax.isChecked():
        radar = json.loads(session.query(CurrentProfileMinMax.signal).filter(CurrentProfileMinMax.id == 1).first()[0])
    else:
        radar = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])
    draw_image(radar)
    updatePlot()
