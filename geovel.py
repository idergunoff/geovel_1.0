import traceback

from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QColorDialog

from load import *
from filtering import *
from draw import *
from layer import *
from well import *
from lda import *
from mlp import *
from monitoring import *
from krige import *
from formation_ai import *
from regression import *

MainWindow.show()


# def show_globals():
#     print(globals().keys())



def mouse_moved_to_signal(evt):
    """ Отслеживаем координаты курсора и отображение на графике сигнала """
    global hor_line_sig, hor_line_rad, vert_line_rad, vert_line_graph


    # Удаление предыдущих линий при движении мыши
    if 'hor_line_sig' in globals():
        ui.signal.removeItem(hor_line_sig)
        radarogramma.removeItem(hor_line_rad)
        radarogramma.removeItem(vert_line_rad)
        ui.graph.removeItem(vert_line_graph)
    # Получение координат курсора
    pos = evt[0]
    vb = radarogramma.vb
    # Проверка, находится ли курсор в пределах области графика
    if radarogramma.sceneBoundingRect().contains(pos):
        mousePoint = vb.mapSceneToView(pos)
        # Создание бесконечных линий
        hor_line_sig = pg.InfiniteLine(pos=mousePoint.y(), angle=0, pen=pg.mkPen(color='w', width=0.5, dash=[4, 7]))
        hor_line_rad = pg.InfiniteLine(pos=mousePoint.y(), angle=0, pen=pg.mkPen(color='w', width=0.5, dash=[4, 7]))
        vert_line_rad = pg.InfiniteLine(pos=mousePoint.x(), angle=90, pen=pg.mkPen(color='w', width=0.5, dash=[4, 7]))
        vert_line_graph = pg.InfiniteLine(pos=mousePoint.x(), angle=90, pen=pg.mkPen(color='w', width=0.5, dash=[4, 7]))
        # Добавление линий на соответствующие графики
        ui.signal.addItem(hor_line_sig)
        radarogramma.addItem(hor_line_rad)
        radarogramma.addItem(vert_line_rad)
        ui.graph.addItem(vert_line_graph)


proxy = pg.SignalProxy(radarogramma.scene().sigMouseMoved, rateLimit=60, slot=mouse_moved_to_signal)


def log_uncaught_exceptions(ex_cls, ex, tb):
    """ Вывод ошибок программы """
    text = '{}: {}:\n'.format(ex_cls.__name__, ex)
    text += ''.join(traceback.format_tb(tb))
    print(text)
    QtWidgets.QMessageBox.critical(None, 'Error', text)
    sys.exit()


def change_color():
    button_color = ui.pushButton_color.palette().color(ui.pushButton_color.backgroundRole())
    color = QColorDialog.getColor(button_color)
    ui.pushButton_color.setStyleSheet(f"background-color: {color.name()};")
    ui.pushButton_color.setText(color.name())


    # result = QtWidgets.QMessageBox.question(None, 'Вопрос', 'Вы уверены, что хотите выйти?',
    #                                         QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.Cancel)
    #
    # if result == QtWidgets.QMessageBox.Yes:
    #     print('Да')
    #
    # elif result == QtWidgets.QMessageBox.Cancel:
    #     print('Отмена')


img.scene().sigMouseClicked.connect(mouseClicked)
img.scene().sigMouseMoved.connect(mouseLine)



ui.pushButton_save_signal.clicked.connect(save_signal)
ui.pushButton_draw_rad.clicked.connect(draw_radarogram)
ui.pushButton_draw_cur.clicked.connect(draw_current_radarogram)
ui.pushButton_vacuum.clicked.connect(vacuum)
ui.pushButton_uf.clicked.connect(load_uf_grid)
ui.pushButton_m.clicked.connect(load_m_grid)
ui.pushButton_r.clicked.connect(load_r_grid)
ui.pushButton_fft.clicked.connect(calc_fft)
ui.pushButton_add_fft.clicked.connect(calc_add_fft)
ui.pushButton_ifft.clicked.connect(calc_ifft)
ui.pushButton_medfilt.clicked.connect(calc_medfilt)
ui.pushButton_wiener.clicked.connect(calc_wiener)
ui.pushButton_savgol.clicked.connect(calc_savgol)
ui.pushButton_filtfilt.clicked.connect(calc_filtfilt)
ui.pushButton_reset.clicked.connect(reset_spinbox_fft)
# ui.pushButton_maxmin.clicked.connect(draw_max_min)
ui.pushButton_rfft2.clicked.connect(calc_rfft2)
ui.pushButton_irfft2.clicked.connect(calc_irfft2)
ui.pushButton_dct.clicked.connect(calc_dctn)
ui.pushButton_idct.clicked.connect(calc_idctn)
ui.pushButton_log.clicked.connect(calc_log)
ui.pushButton_rang.clicked.connect(calc_rang)
ui.pushButton_add_window.clicked.connect(add_window)
ui.pushButton_add_layer.clicked.connect(add_layer)
ui.pushButton_add_formation.clicked.connect(add_formation)
ui.pushButton_remove_layer.clicked.connect(remove_layer)
ui.pushButton_edges_layer.clicked.connect(save_layer)
ui.pushButton_layer_filt.clicked.connect(filter_layer)
# ui.pushButton_find_oil.clicked.connect(show_globals)
ui.pushButton_add_well.clicked.connect(add_well)
ui.pushButton_edit_well.clicked.connect(edit_well)
ui.pushButton_add_wells.clicked.connect(add_wells)
ui.pushButton_rem_well.clicked.connect(remove_well)
ui.pushButton_add_bound.clicked.connect(add_boundary)
ui.pushButton_rem_bound.clicked.connect(remove_boundary)
ui.pushButton_color.clicked.connect(change_color)
#   lda
ui.pushButton_add_lda.clicked.connect(add_lda)
ui.pushButton_rem_lda.clicked.connect(remove_lda)
ui.pushButton_copy_lda.clicked.connect(copy_lda)
ui.pushButton_copy_lda_to_mlp.clicked.connect(copy_lda_to_mlp)
ui.pushButton_add_mark_lda.clicked.connect(add_marker_lda)
ui.pushButton_rem_mark_lda.clicked.connect(remove_marker_lda)
ui.pushButton_add_well_lda.clicked.connect(add_well_markup_lda)
ui.pushButton_add_geovel_lda.clicked.connect(add_param_geovel_lda)
ui.pushButton_add_all_geovel_lda.clicked.connect(add_all_param_geovel_lda)
ui.pushButton_add_distr_lda.clicked.connect(add_param_distr_lda)
ui.pushButton_add_sep_lda.clicked.connect(add_param_sep_lda)
ui.pushButton_add_all_distr.clicked.connect(add_all_param_distr_lda)
ui.pushButton_update_well_lda.clicked.connect(update_well_markup_lda)
ui.pushButton_rem_well_lda.clicked.connect(remove_well_markup_lda)
ui.pushButton_rem_param_lda.clicked.connect(remove_param_geovel_lda)
ui.pushButton_clear_params_lda.clicked.connect(remove_all_param_geovel_lda)
ui.pushButton_draw_lda.clicked.connect(draw_LDA)
ui.pushButton_verify_lda.clicked.connect(calc_verify_lda)
ui.pushButton_reset_verify_lda.clicked.connect(reset_verify_lda)
ui.pushButton_calc_lda.clicked.connect(calc_LDA)
ui.pushButton_calc_obj_lda.clicked.connect(calc_obj_lda)
ui.pushButton_add_mfcc.clicked.connect(add_param_mfcc_lda)
ui.pushButton_add_all_mfcc.clicked.connect(add_all_param_mfcc_lda)
ui.listWidget_well_lda.currentItemChanged.connect(choose_marker_lda)
ui.comboBox_lda_analysis.activated.connect(update_list_marker_lda_db)
ui.pushButton_test.clicked.connect(test)

#   mlp
ui.pushButton_add_mlp.clicked.connect(add_mlp)
ui.pushButton_rem_mlp.clicked.connect(remove_mlp)
ui.pushButton_copy_mlp.clicked.connect(copy_mlp)
ui.pushButton_copy_to_lda.clicked.connect(copy_mlp_to_lda)
ui.pushButton_copy_regmod.clicked.connect(copy_mlp_to_regmod)
ui.pushButton_add_mark_mlp.clicked.connect(add_marker_mlp)
ui.pushButton_rem_mark_mlp.clicked.connect(remove_marker_mlp)
ui.pushButton_add_well_mlp.clicked.connect(add_well_markup_mlp)
ui.pushButton_add_geovel_mlp.clicked.connect(add_param_geovel_mlp)
ui.pushButton_add_all_geovel_mlp.clicked.connect(add_all_param_geovel_mlp)
ui.pushButton_add_distr_mlp.clicked.connect(add_param_distr_mlp)
ui.pushButton_add_sep_mlp.clicked.connect(add_param_sep_mlp)
ui.pushButton_add_all_distr_mlp.clicked.connect(add_all_param_distr_mlp)
ui.pushButton_update_well_mlp.clicked.connect(update_well_markup_mlp)
ui.pushButton_rem_well_mlp.clicked.connect(remove_well_markup_mlp)
ui.pushButton_rem_param_mlp.clicked.connect(remove_param_geovel_mlp)
ui.pushButton_clear_params_mlp.clicked.connect(remove_all_param_geovel_mlp)
ui.pushButton_draw_mlp.clicked.connect(draw_MLP)
ui.pushButton_calc_mlp.clicked.connect(calc_MLP)
ui.pushButton_calc_obj_mlp.clicked.connect(calc_obj_mlp)
ui.pushButton_add_mfcc_mlp.clicked.connect(add_param_mfcc_mlp)
ui.pushButton_add_all_mfcc_mlp.clicked.connect(add_all_param_mfcc_mlp)
ui.comboBox_mlp_analysis.activated.connect(update_list_marker_mlp_db)
ui.listWidget_well_mlp.currentItemChanged.connect(choose_marker_mlp)


#   regression
ui.pushButton_add_regmod.clicked.connect(add_regression_model)
ui.pushButton_rem_regmod.clicked.connect(remove_reg)
ui.pushButton_copy_reg.clicked.connect(copy_regmod)
# ui.pushButton_copy_to_lda.clicked.connect(copy_reg_to_lda)
ui.pushButton_add_well_reg.clicked.connect(add_well_markup_reg)
ui.pushButton_add_geovel_reg.clicked.connect(add_param_geovel_reg)
ui.pushButton_add_all_geovel_reg.clicked.connect(add_all_param_geovel_reg)
ui.pushButton_add_distr_reg.clicked.connect(add_param_distr_reg)
ui.pushButton_add_sep_reg.clicked.connect(add_param_sep_reg)
ui.pushButton_add_all_distr_reg.clicked.connect(add_all_param_distr_reg)
ui.pushButton_update_well_reg.clicked.connect(update_well_markup_reg)
ui.pushButton_rem_well_reg.clicked.connect(remove_well_markup_reg)
ui.pushButton_rem_param_reg.clicked.connect(remove_param_geovel_reg)
ui.pushButton_clear_params_reg.clicked.connect(remove_all_param_geovel_reg)
ui.pushButton_train_regmod.clicked.connect(train_regression_model)
ui.pushButton_calc_reg.clicked.connect(calc_profile_model_regmod)
ui.pushButton_calc_obj_reg.clicked.connect(calc_object_model_regmod)
ui.pushButton_add_mfcc_reg.clicked.connect(add_param_mfcc_reg)
ui.pushButton_add_all_mfcc_reg.clicked.connect(add_all_param_mfcc_reg)
ui.pushButton_rem_trained_model_reg.clicked.connect(remove_trained_model_regmod)
ui.comboBox_regmod.activated.connect(update_list_well_markup_reg)
ui.listWidget_well_regmod.currentItemChanged.connect(choose_markup_regmod)


ui.toolButton_add_obj.clicked.connect(add_object)
ui.toolButton_load_prof.clicked.connect(load_profile)
ui.toolButton_del_prof.clicked.connect(delete_profile)
ui.toolButton_load_plast.clicked.connect(load_param)
ui.toolButton_del_plast.clicked.connect(remove_formation)
ui.toolButton_crop_up.clicked.connect(crop_up)
ui.toolButton_crop_down.clicked.connect(crop_down)


ui.comboBox_object.activated.connect(update_research_combobox)
ui.comboBox_research.activated.connect(update_profile_combobox)
ui.comboBox_research.activated.connect(check_coordinates_research)
ui.comboBox_profile.activated.connect(update_formation_combobox)
ui.comboBox_profile.activated.connect(check_coordinates_profile)
ui.comboBox_plast.activated.connect(update_param_combobox)
ui.comboBox_plast.activated.connect(draw_formation)
ui.comboBox_param_plast.activated.connect(draw_param)

ui.checkBox_minmax.stateChanged.connect(choose_minmax)
ui.checkBox_draw_layer.stateChanged.connect(draw_layers)
ui.checkBox_all_formation.stateChanged.connect(draw_param)
ui.checkBox_profile_well.stateChanged.connect(update_list_well)
ui.checkBox_profile_intersec.stateChanged.connect(update_list_well)
ui.checkBox_profile_intersec.stateChanged.connect(set_title_list_widget_wells)

ui.spinBox_ftt_up.valueChanged.connect(draw_fft_spectr)
ui.spinBox_fft_down.valueChanged.connect(draw_fft_spectr)
ui.spinBox_roi.valueChanged.connect(changeSpinBox)
ui.spinBox_rad_up.valueChanged.connect(draw_rad_line)
ui.spinBox_rad_down.valueChanged.connect(draw_rad_line)
ui.spinBox_well_distance.valueChanged.connect(update_list_well)
ui.doubleSpinBox_vmin.valueChanged.connect(draw_bound_int)
ui.doubleSpinBox_vmax.valueChanged.connect(draw_bound_int)
ui.doubleSpinBox_vsr.valueChanged.connect(draw_wells)


ui.listWidget_well.currentItemChanged.connect(show_data_well)
ui.listWidget_well.currentItemChanged.connect(update_boundaries)
ui.listWidget_bound.currentItemChanged.connect(draw_bound_int)

###################################################################
#########################   Monitoring   ##########################
###################################################################

ui.comboBox_object_monitor.activated.connect(update_list_h_well)
ui.listWidget_h_well.currentItemChanged.connect(update_list_param_h_well)
ui.listWidget_param_h_well.currentItemChanged.connect(draw_param_h_well)
ui.listWidget_h_well.currentItemChanged.connect(update_list_thermogram)
ui.listWidget_thermogram.currentItemChanged.connect(show_thermogram)
ui.pushButton_add_h_well.clicked.connect(add_h_well)
ui.pushButton_rem_h_well.clicked.connect(remove_h_well)
ui.pushButton_edit_h_well.clicked.connect(edit_h_well)
ui.pushButton_param_h_well.clicked.connect(load_param_h_well)
ui.pushButton_rem_param.clicked.connect(remove_parameter)
ui.pushButton_inclin_h_well.clicked.connect(load_inclinometry_h_well)
ui.pushButton_therm_h_well.clicked.connect(load_thermogram_h_well)
ui.pushButton_wellhead.clicked.connect(load_wellhead)
ui.pushButton_wellhead_batch.clicked.connect(load_wellhead_batch)
ui.pushButton_show_incl.clicked.connect(show_inclinometry)
ui.pushButto_remove_therm.clicked.connect(remove_thermogram)
ui.pushButton_show_corr_therm.clicked.connect(show_corr_therm)
ui.doubleSpinBox_start_therm.valueChanged.connect(show_start_therm)
ui.doubleSpinBox_end_therm.valueChanged.connect(show_end_therm)
ui.pushButton_set_start_therm.clicked.connect(set_start_therm)
ui.pushButton_cut_end_therm.clicked.connect(cut_end_therm)
ui.pushButton_coord_therm.clicked.connect(coordinate_binding_thermogram)
ui.pushButton_show_therms.clicked.connect(show_therms_animation)
ui.pushButton_therm_mean_day.clicked.connect(mean_day_thermogram)
ui.pushButton_add_intersec.clicked.connect(add_intersection)
ui.pushButton_draw_map_therm.clicked.connect(draw_map_by_thermogram)


###################################################################
#######################   formation AI   ##########################
###################################################################

ui.pushButton_add_formation_ai.clicked.connect(add_formation_ai)
ui.pushButton_rem_formation_ai.clicked.connect(remove_formation_ai)
ui.pushButton_clear_formation_ai.clicked.connect(clear_formation_ai)
ui.pushButton_model_ai.clicked.connect(calc_model_ai)
ui.pushButton_add_model.clicked.connect(add_model_ai)
ui.pushButton_rem_model.clicked.connect(remove_model_ai)
ui.comboBox_model_ai.activated.connect(update_list_formation_ai)
ui.pushButton_model_calc_profile.clicked.connect(calc_model_profile)
ui.pushButton_rem_trained_model.clicked.connect(remove_trained_model)

ui.pushButton_map.clicked.connect(show_map)
ui.pushButton_profiles.clicked.connect(show_profiles)
ui.pushButton_get_attributes.clicked.connect(get_attributes)

roi.sigRegionChanged.connect(updatePlot)

ui.pushButton_find_oil.clicked.connect(filter19)
ui.pushButton_secret_filter.clicked.connect(secret_filter)

update_object()
update_list_object_monitor()
clear_current_profile()
clear_current_profile_min_max()
clear_spectr()
clear_window_profile()
# update_layers()
update_list_well()
set_info('Старт...', 'green')
set_random_color()
update_list_lda(True)
update_list_mlp(True)
update_list_reg()
set_param_lda_to_combobox()
set_param_mlp_to_combobox()
set_param_regmod_to_combobox()
update_combobox_model_ai()
update_list_trained_models()
update_list_trained_models_regmod()
check_coordinates_profile()
check_coordinates_research()


# update_list_param_lda()




sys.excepthook = log_uncaught_exceptions
# app.processEvents()
sys.exit(app.exec_())