import traceback

from load import *
from filtering import *
from draw import *
from layer import *
from calc_additional_features import *
from calc_profile_features import *
from well import *
from mlp import *
from monitoring import *
from krige import *
from formation_ai import *
from regression import *
from exploration import *
from geochem import *
from velocity_model import *
from velocity_prediction import *
from test_model import *
from lineup_train_models import *
from nn_torch_classifier import *
from nn_torch_regression import *
from feature_selection import *
from index_productivity import *
from pareto import *
from remote_db.remote_db_win import *
from well_log import *
from mask_params import *
from test_tsfresh import *

MainWindow.show()

m_width, m_height = get_width_height_monitor()
MainWindow.resize(m_width - 100, m_height - 200)


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
        hor_line_sig = pg.InfiniteLine(pos=mousePoint.y(), angle=0, pen=pg.mkPen(color='grey', width=0.5, dash=[4, 7]))
        hor_line_rad = pg.InfiniteLine(pos=mousePoint.y(), angle=0, pen=pg.mkPen(color='grey', width=0.5, dash=[4, 7]))
        vert_line_rad = pg.InfiniteLine(pos=mousePoint.x(), angle=90, pen=pg.mkPen(color='grey', width=0.5, dash=[4, 7]))
        vert_line_graph = pg.InfiniteLine(pos=mousePoint.x(), angle=90, pen=pg.mkPen(color='grey', width=0.5, dash=[4, 7]))

        # Добавление линий на соответствующие графики
        ui.signal.addItem(hor_line_sig)
        radarogramma.addItem(hor_line_rad)
        radarogramma.addItem(vert_line_rad)
        ui.graph.addItem(vert_line_graph)


def clear_info(evt):
    ui.info.clear()

# очистка информации
ui.info.mouseDoubleClickEvent = clear_info


proxy = pg.SignalProxy(radarogramma.scene().sigMouseMoved, rateLimit=60, slot=mouse_moved_to_signal)




def log_uncaught_exceptions(ex_cls, ex, tb):
    """ Вывод ошибок программы """
    text = '{}: {}:\n'.format(ex_cls.__name__, ex)
    text += ''.join(traceback.format_tb(tb))
    print(text)
    QtWidgets.QMessageBox.critical(None, 'Error', text)
    sys.exit()


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

ui.pushButton_test_tsfresh.clicked.connect(extract_signal_features)

ui.pushButton_rem_db.clicked.connect(open_rem_db_window)
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
ui.pushButton_wavelet_filter.clicked.connect(calc_wavelet_filter)
ui.pushButton_calc_all_param.clicked.connect(calc_all_params)
ui.toolButton_recalc_relief.clicked.connect(recalc_relief)

ui.pushButton_reset.clicked.connect(reset_spinbox_fft)
# ui.pushButton_maxmin.clicked.connect(draw_max_min)
ui.pushButton_rfft2.clicked.connect(calc_rfft2)
ui.pushButton_irfft2.clicked.connect(calc_irfft2)
ui.pushButton_dct.clicked.connect(calc_dctn)
ui.pushButton_idct.clicked.connect(calc_idctn)
ui.pushButton_log.clicked.connect(calc_log)
ui.pushButton_rang.clicked.connect(calc_rang)
ui.pushButton_butter.clicked.connect(butterworth_filter)
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
ui.pushButton_add_data_well.clicked.connect(add_data_well)
ui.pushButton_rem_well.clicked.connect(remove_well)
ui.pushButton_add_bound.clicked.connect(add_boundary)
ui.pushButton_rem_bound.clicked.connect(remove_boundary)
ui.pushButton_color.clicked.connect(change_color)
ui.pushButton_set_scale.clicked.connect(set_scale)
ui.checkBox_grid.clicked.connect(show_grid)
ui.pushButton_save_img.clicked.connect(save_image)
ui.pushButton_slide.clicked.connect(slide_average)
radarogramma.sigRangeChanged.connect(on_range_changed)
ui.checkBox_black_white.clicked.connect(change_background)

ui.pushButton_test.clicked.connect(draw_max_min)

#   mlp
ui.pushButton_add_mlp.clicked.connect(add_mlp)
ui.pushButton_rem_mlp.clicked.connect(remove_mlp)
ui.pushButton_copy_mlp.clicked.connect(copy_mlp)
ui.pushButton_copy_regmod.clicked.connect(copy_mlp_to_regmod)
ui.pushButton_add_mark_mlp.clicked.connect(add_marker_mlp)
ui.pushButton_rem_mark_mlp.clicked.connect(remove_marker_mlp)
ui.pushButton_add_well_mlp.clicked.connect(add_well_markup_mlp)
ui.pushButton_add_profile_mlp.clicked.connect(add_profile_mlp)
ui.pushButton_add_signal_mlp.clicked.connect(add_param_signal_mlp)
ui.pushButton_add_all_signal_mlp.clicked.connect(add_all_param_signal_mlp)
ui.pushButton_add_geovel_mlp.clicked.connect(add_param_geovel_mlp)
ui.pushButton_add_all_geovel_mlp.clicked.connect(add_all_param_geovel_mlp)
ui.pushButton_add_prof_ftr_mlp.clicked.connect(add_param_profile_mlp)
ui.pushButton_add_all_prof_ftr_mlp.clicked.connect(add_all_param_profile_mlp)
ui.pushButton_add_distr_mlp.clicked.connect(add_param_distr_mlp)
ui.pushButton_add_sep_mlp.clicked.connect(add_param_sep_mlp)
ui.pushButton_add_all_distr_mlp.clicked.connect(add_all_param_distr_mlp)
ui.pushButton_add_crl.clicked.connect(add_param_crl_mlp)
ui.pushButton_add_crl_nf.clicked.connect(add_param_crl_nf_mlp)
ui.pushButton_mlp_add_param_list.clicked.connect(add_param_list_mlp)
ui.pushButton_update_well_mlp.clicked.connect(update_well_markup_mlp)
ui.pushButton_rem_well_mlp.clicked.connect(remove_well_markup_mlp)
ui.pushButton_train_test_mlp.clicked.connect(split_well_train_test_mlp)
ui.pushButton_rem_param_mlp.clicked.connect(remove_param_geovel_mlp)
ui.pushButton_clear_params_mlp.clicked.connect(remove_all_param_geovel_mlp)
ui.pushButton_draw_mlp.clicked.connect(draw_MLP)
ui.pushButton_calc_mlp.clicked.connect(calc_class_profile)
ui.pushButton_calc_obj_mlp.clicked.connect(calc_object_class)
ui.pushButton_add_mfcc_mlp.clicked.connect(add_param_mfcc_mlp)
ui.pushButton_add_all_mfcc_mlp.clicked.connect(add_all_param_mfcc_mlp)
ui.pushButton_corr_mlp.clicked.connect(calc_corr_mlp)
ui.pushButton_updata_mlp.clicked.connect(update_list_param_mlp)
ui.pushButton_anova_mlp.clicked.connect(anova_mlp)
ui.comboBox_mlp_analysis.activated.connect(update_list_marker_mlp_db)
ui.listWidget_well_mlp.doubleClicked.connect(show_well_markup_mlp)
ui.pushButton_clear_fake.clicked.connect(clear_fake_mlp)
ui.pushButton_rem_trained_model_class.clicked.connect(remove_trained_model_class)
ui.pushButton_comment.clicked.connect(update_trained_model_comment)
ui.toolButton_get_param_list.clicked.connect(get_model_param_list)
ui.pushButton_export_model_class.clicked.connect(export_model_class)
ui.pushButton_import_model_class.clicked.connect(import_model_class)
ui.lineEdit_signal_except.returnPressed.connect(add_signal_except_mlp)
ui.lineEdit_crl_except.returnPressed.connect(add_crl_except_mlp)
ui.pushButton_test_model_class.clicked.connect(test_start)
ui.pushButton_edit_marker_mlp.clicked.connect(edit_marker_mlp)
ui.listWidget_trained_model_class.setContextMenuPolicy(Qt.CustomContextMenu)
ui.listWidget_trained_model_class.customContextMenuRequested.connect(rename_model_class)
ui.toolButton_list_param_to_line.clicked.connect(list_param_to_lineEdit)
ui.toolButton_feature_importance.clicked.connect(get_feature_importance_cls)
ui.toolButton_excel_markup_mlp.clicked.connect(markup_to_excel_mlp)
ui.pushButton_pareto.clicked.connect(pareto_start)
ui.pushButton_mask_param.clicked.connect(mask_param_form)
ui.toolButton_rm_well_fake.clicked.connect(remove_fake_current_markup)

# torch
ui.pushButton_add_torch.clicked.connect(torch_classifier_train)
ui.pushButton_torch.clicked.connect(torch_regressor_train)


#   regression
ui.pushButton_add_regmod.clicked.connect(add_regression_model)
ui.pushButton_rem_regmod.clicked.connect(remove_reg)
ui.pushButton_copy_reg.clicked.connect(copy_regmod)
ui.pushButton_add_signal_reg.clicked.connect(add_param_signal_reg)
ui.pushButton_add_all_signal_reg.clicked.connect(add_all_param_signal_reg)
ui.pushButton_add_crl_reg.clicked.connect(add_param_crl_reg)
ui.pushButton_add_crl_nf_reg.clicked.connect(add_param_crl_nf_reg)
ui.pushButton_add_well_reg.clicked.connect(add_well_markup_reg)
ui.pushButton_add_geovel_reg.clicked.connect(add_param_geovel_reg)
ui.pushButton_add_all_geovel_reg.clicked.connect(add_all_param_geovel_reg)
ui.pushButton_add_distr_reg.clicked.connect(add_param_distr_reg)
ui.pushButton_add_sep_reg.clicked.connect(add_param_sep_reg)
ui.pushButton_add_all_distr_reg.clicked.connect(add_all_param_distr_reg)
ui.pushButton_add_prof_ftr_reg.clicked.connect(add_param_profile_reg)
ui.pushButton_add_all_prof_ftr_reg.clicked.connect(add_all_param_profile_reg)
# ui.pushButton_model_param_reg.clicked.connect(add_model_param_reg)
ui.pushButton_update_well_reg.clicked.connect(update_well_markup_reg)
ui.pushButton_update_all_well_reg.clicked.connect(update_all_well_markup_reg)
ui.pushButton_train_test_reg.clicked.connect(split_well_train_test)
ui.pushButton_rem_well_reg.clicked.connect(remove_well_markup_reg)
ui.pushButton_exp_well_reg.clicked.connect(export_well_markup_reg)
ui.pushButton_rem_param_reg.clicked.connect(remove_param_geovel_reg)
ui.pushButton_clear_params_reg.clicked.connect(remove_all_param_geovel_reg)
ui.pushButton_train_regmod.clicked.connect(train_regression_model)
ui.pushButton_calc_reg.clicked.connect(calc_profile_model_regmod)
ui.pushButton_calc_obj_reg.clicked.connect(calc_object_model_regmod)
ui.pushButton_add_mfcc_reg.clicked.connect(add_param_mfcc_reg)
ui.pushButton_add_all_mfcc_reg.clicked.connect(add_all_param_mfcc_reg)
ui.pushButton_rem_trained_model_reg.clicked.connect(remove_trained_model_regmod)
ui.pushButton_corr_regmod.clicked.connect(calc_corr_regmod)
ui.pushButton_updata_regmod.clicked.connect(update_list_param_regmod)
ui.pushButton_anova_regmod.clicked.connect(anova_regmod)
ui.pushButton_reg_add_param_list.clicked.connect(add_param_list_reg)
ui.pushButton_add_all_well_regmod.clicked.connect(add_all_well_markup_reg)
ui.toolButton_excel_markup_reg.clicked.connect(markup_to_excel_reg)


ui.comboBox_regmod.activated.connect(update_list_well_markup_reg)
ui.listWidget_well_regmod.doubleClicked.connect(choose_markup_regmod)
ui.pushButton_reg_clear_fake.clicked.connect(clear_fake_reg)
ui.pushButton_reg_comment.clicked.connect(update_trained_model_reg_comment)
ui.toolButton_get_param_list_reg.clicked.connect(get_model_param_list_reg)
ui.pushButton_export_model_reg.clicked.connect(export_model_reg)
ui.pushButton_import_model_reg.clicked.connect(import_model_reg)
ui.pushButton_test_model_reg.clicked.connect(regression_test)
ui.lineEdit_signal_except_reg.returnPressed.connect(add_signal_except_reg)
ui.lineEdit_crl_except_reg.returnPressed.connect(add_crl_except_reg)
ui.listWidget_trained_model_reg.itemDoubleClicked.connect(rename_model_reg)
ui.toolButton_list_param_to_line_reg.clicked.connect(list_param_reg_to_lineEdit)

ui.toolButton_add_obj.clicked.connect(add_object)
ui.toolButton_del_obj.clicked.connect(remove_object)
ui.toolButton_load_prof.clicked.connect(load_profile)
ui.toolButton_del_prof.clicked.connect(delete_profile)
ui.toolButton_load_plast.clicked.connect(load_param)
ui.toolButton_del_plast.clicked.connect(remove_formation)
ui.toolButton_crop_up.clicked.connect(crop_up)
ui.toolButton_crop_down.clicked.connect(crop_down)
ui.toolButton_feature_importance_reg.clicked.connect(get_feature_importance_reg)



ui.comboBox_object.activated.connect(update_research_combobox)
ui.comboBox_research.activated.connect(update_profile_combobox)
ui.comboBox_research.activated.connect(check_coordinates_research)
ui.comboBox_profile.activated.connect(update_formation_combobox)
ui.comboBox_profile.activated.connect(check_coordinates_profile)
ui.comboBox_profile.currentTextChanged.connect(update_list_binding)
ui.comboBox_profile.currentTextChanged.connect(update_list_velocity_model)
ui.comboBox_profile.activated.connect(calc_add_features_profile)
ui.comboBox_profile.activated.connect(calc_profile_features)
ui.comboBox_profile.currentTextChanged.connect(update_list_bind_vel_prediction)
ui.comboBox_profile.currentTextChanged.connect(index_prod_list_update)

ui.comboBox_plast.activated.connect(update_param_combobox)
ui.comboBox_plast.activated.connect(draw_formation)
ui.comboBox_param_plast.activated.connect(draw_param)

ui.checkBox_minmax.stateChanged.connect(choose_minmax)
ui.checkBox_draw_layer.stateChanged.connect(draw_layers)
ui.checkBox_all_formation.stateChanged.connect(draw_param)
ui.checkBox_profile_well.stateChanged.connect(update_list_well)
ui.checkBox_prof_intersect.stateChanged.connect(draw_profile_intersection)
ui.checkBox_show_bound.stateChanged.connect(update_list_well)
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
ui.doubleSpinBox_vsr.valueChanged.connect(draw_bound_int)


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
ui.pushButton_remove_therm_date.clicked.connect(remove_therm_by_date)
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
ui.pushButton_model_calc_object.clicked.connect(calc_model_object)
ui.pushButton_rem_trained_model.clicked.connect(remove_trained_model)
ui.pushButton_export_model_f_ai.clicked.connect(export_model_formation_ai)
ui.pushButton_import_model_f_ai.clicked.connect(import_model_formation_ai)


ui.pushButton_map.clicked.connect(show_map)
ui.pushButton_profiles.clicked.connect(show_profiles)
ui.pushButton_get_attributes.clicked.connect(get_attributes)

roi.sigRegionChanged.connect(updatePlot)

ui.pushButton_find_oil.clicked.connect(filter19)
ui.pushButton_secret_filter.clicked.connect(secret_filter)
ui.listWidget_formation_ai.currentItemChanged.connect(choose_formation_ai)


#################################################################
######################### WELL LOG ##############################
#################################################################

ui.pushButton_well_log.clicked.connect(show_well_log)

#################################################################
###################### Exploration ##############################
#################################################################

ui.pushButton_add_expl.clicked.connect(add_exploration)
ui.pushButton_del_expl.clicked.connect(remove_exploration)
ui.pushButton_add_set_point.clicked.connect(add_set_point)
ui.pushButton_del_set_point.clicked.connect(remove_set_point)
ui.pushButton_add_train_point.clicked.connect(add_train_set_point)
ui.pushButton_del_train_point.clicked.connect(remove_train_set_point)
ui.pushButton_add_analysis_expl.clicked.connect(add_analysis)
ui.pushButton_del_analysis_expl.clicked.connect(remove_analysis)
ui.pushButton_add_train_param.clicked.connect(add_analysis_parameter_tolist)
ui.pushButton_del_param_expl.clicked.connect(del_analysis_parameter)
ui.pushButton_clear_param_expl.clicked.connect(clear_all_analysis_parameters)
ui.pushButton_add_all_train_param.clicked.connect(add_all_analysis_parameter_tolist)
ui.pushButton_load_point.clicked.connect(load_point_exploration)
ui.pushButton_interp.clicked.connect(draw_interpolation)
ui.pushButton_train_interp.clicked.connect(train_interpolation)
ui.pushButton_field.clicked.connect(exploration_MLP)
ui.pushButton_calc_expl_class.clicked.connect(calc_exploration_class)
ui.pushButton_load_train_point.clicked.connect(load_train_data)
ui.pushButton_add_geovel_expl.clicked.connect(add_geo_analysis_param)
ui.pushButton_add_all_geovel_expl.clicked.connect(add_all_geo_analysis_param)
ui.pushButton_calc_expl.clicked.connect(show_interp_map)
ui.pushButton_rem_trained_model_expl.clicked.connect(remove_model_exploration)
ui.comboBox_expl.activated.connect(update_list_param_exploration)
ui.comboBox_expl.activated.connect(update_list_set_point)
ui.comboBox_set_point.activated.connect(update_list_point_exploration)
ui.comboBox_set_point.activated.connect(update_analysis_combobox)
ui.comboBox_analysis_expl.activated.connect(update_analysis_list)
ui.comboBox_analysis_expl.activated.connect(update_models_expl_list)
ui.comboBox_train_point.activated.connect(update_train_list)
ui.comboBox_train_point.activated.connect(update_analysis_combobox)
ui.comboBox_object.activated.connect(update_train_combobox)

######################################################
###################### GEOCHEM #######################
######################################################

ui.pushButton_del_geochem.clicked.connect(remove_geochem)
ui.pushButton_load_geochem.clicked.connect(load_geochem)
ui.comboBox_geochem.activated.connect(update_listwidget_geochem_point)
ui.comboBox_geochem.activated.connect(update_combobox_geochem_well)
ui.comboBox_geochem.activated.connect(update_listwidget_param_geochem)
ui.comboBox_geochem_well.activated.connect(update_listwidget_geochem_well_point)
ui.pushButton_geochem_anova.clicked.connect(anova_geochem)
ui.pushButton_geochem_tsne.clicked.connect(tsne_geochem)
ui.pushButton_add_g_maket.clicked.connect(add_maket)
ui.pushButton_rem_g_maket.clicked.connect(remove_maket)
ui.pushButton_add_g_param_train.clicked.connect(add_geochem_param_train)
ui.pushButton_add_all_g_param.clicked.connect(add_all_geochem_param_train)
ui.pushButton_rem_g_param_train.clicked.connect(remove_geochem_param_train)
ui.pushButton_add_g_cat.clicked.connect(add_category)
ui.pushButton_rem_g_cat.clicked.connect(remove_category)
ui.comboBox_geochem_maket.activated.connect(update_geochem_param_train_list)
ui.comboBox_geochem_maket.activated.connect(update_category_combobox)
ui.comboBox_geochem_maket.activated.connect(update_g_model_list)
ui.comboBox_geochem_cat.activated.connect(update_g_train_point_list)
ui.pushButton_add_g_train_well.clicked.connect(add_whole_well_to_list)
ui.pushButton_add_train_g_point.clicked.connect(add_field_point_to_list)
ui.listWidget_g_train_point.doubleClicked.connect(del_g_train_point)
ui.pushButton_geochem_train_model.clicked.connect(train_model_geochem)
ui.pushButton_calc_g_class.clicked.connect(calc_geochem_classification)
ui.pushButton_g_drop_fake.clicked.connect(drop_fake_geochem)
ui.pushButton_rem_trained_g_model.clicked.connect(remove_g_model)
ui.pushButton_g_graph.clicked.connect(draw_point_graph)
ui.pushButton_g_comment.clicked.connect(update_trained_model_geochem_comment)
ui.pushButton_vector.clicked.connect(calc_vector)
ui.pushButton_isoforest.clicked.connect(calc_isolation_forest)
ui.listWidget_g_point.doubleClicked.connect(add_geochem_point_fake)
ui.pushButton_clear_fake_point.clicked.connect(clear_fake_point)

# radarogramma.getViewBox().sigRangeChanged.connect(draw_axis)

############### VELOCITY MODEL ################
ui.pushButton_add_bind.clicked.connect(add_binding)
ui.pushButton_del_bind.clicked.connect(remove_binding)
ui.pushButton_calc_vel_form.clicked.connect(calc_velocity_model)
ui.listWidget_bind.currentItemChanged.connect(draw_bind_point)
ui.listWidget_bind.itemDoubleClicked.connect(remove_bind_point)
ui.pushButton_del_vel_model.clicked.connect(remove_velocity_model)
ui.listWidget_vel_model.itemClicked.connect(draw_vel_model_point)
ui.checkBox_vel_color.clicked.connect(draw_vel_model_point)
ui.pushButton_draw_deep_prof.clicked.connect(draw_deep_profile)
ui.listWidget_vel_model.itemDoubleClicked.connect(remove_fill_form)
ui.pushButton_lineup.clicked.connect(model_lineup)
ui.checkBox_relief.clicked.connect(draw_relief)
ui.checkBox_vel.clicked.connect(draw_relief)
ui.checkBox_velmod.clicked.connect(draw_velocity_model_color)

########## Profile Model Prediction ############
ui.comboBox_profile.activated.connect(update_list_model_prediction)
ui.pushButton_model_pred_rmv.clicked.connect(remove_model_prediction)
ui.listWidget_model_pred.currentItemChanged.connect(draw_profile_model_prediction)
ui.listWidget_model_pred.activated.connect(draw_profile_model_prediction)
ui.listWidget_model_pred.clicked.connect(draw_profile_model_prediction)
ui.pushButton_add_predict_mlp.clicked.connect(add_predict_mlp)
ui.pushButton_add_predict_reg.clicked.connect(add_predict_reg)
ui.pushButton_draw_predict.clicked.connect(draw_profile_model_predict)
ui.pushButton_save_excel_predict.clicked.connect(save_excel_profile_model_predict)
ui.pushButton_correct_predict.clicked.connect(correct_profile_model_predict)
ui.checkBox_corr_pred.clicked.connect(draw_profile_model_prediction)
ui.toolButton_check_uf.clicked.connect(check_uf)

ui.toolButton_ix_prod_add.clicked.connect(index_prod_add)
ui.toolButton_ix_prod_rmv.clicked.connect(index_prod_remove)
ui.toolButton_ix_prod_draw.clicked.connect(index_prod_draw)
ui.toolButton_ix_prod_excel.clicked.connect(index_prod_save)

########## Velocity Prediction ############
ui.pushButton_add_bind_vel_pred.clicked.connect(add_bind_vel_prediction)
ui.pushButton_rmv_bind_vel_pred.clicked.connect(rmv_bind_vel_prediction)
ui.pushButton_draw_vel_pred.clicked.connect(draw_radar_vel_pred)
ui.checkBox_model_nn.clicked.connect(update_list_model_nn)
ui.listWidget_model_nn.currentItemChanged.connect(draw_relief)

time = datetime.datetime.now()

check_trained_model()
update_object()
update_list_object_monitor()
clear_current_profile()
clear_current_profile_min_max()
clear_current_velocity_model()
clear_spectr()
clear_window_profile()
# update_layers()
update_list_well()
set_info('Старт...', 'green')
set_random_color(ui.pushButton_color)
update_list_mlp(True)
update_list_reg()
set_param_mlp_to_combobox()
set_param_regmod_to_combobox()
set_param_expl_to_combobox()
update_combobox_model_ai()
update_list_trained_models()
update_list_trained_models_regmod()
update_list_trained_models_class()
check_coordinates_profile()
check_coordinates_research()
update_list_exploration()
update_models_expl_list()
update_combobox_geochem()
update_maket_combobox()
update_category_combobox()
update_geochem_param_train_list()
update_g_train_point_list()
check_and_create_folders()
update_list_binding()
update_list_velocity_model()
update_list_model_prediction()
update_list_bind_vel_prediction()
index_prod_list_update()



time2 = datetime.datetime.now() - time
print(time2)



sys.excepthook = log_uncaught_exceptions
# app.processEvents()
sys.exit(app.exec_())