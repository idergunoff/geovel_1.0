import csv

from func import *
from qt.slice_form import Ui_SliceForm


def push_slice():
    SliceForm = QtWidgets.QDialog()
    ui_slc = Ui_SliceForm()
    ui_slc.setupUi(SliceForm)
    SliceForm.show()
    m_width, m_height = get_width_height_monitor()
    SliceForm.resize(int(m_width/1.5), m_height - 200)
    SliceForm.setAttribute(QtCore.Qt.WA_DeleteOnClose) # атрибут удаления виджета после закрытия

    figure = plt.figure()
    canvas = FigureCanvas(figure)
    ui_slc.verticalLayout_slice.addWidget(canvas)

    def get_slice(research_id, idx):
        profiles = session.query(Profile).filter(Profile.research_id == research_id).all()
        if not profiles:
            raise ValueError(f"No profiles found for research_id={research_id}")

        list_x, list_y, list_z = [], [], []

        for profile in profiles:
            x = json.loads(profile.x_pulc)
            y = json.loads(profile.y_pulc)
            z = []
            for sig in json.loads(profile.signal):
                z.append(sig[idx-1])

            list_x.extend(x)
            list_y.extend(y)
            list_z.extend(z)

        return list_x, list_y, list_z


    def draw_slice(research_id, idx):
        figure.clear()
        list_x, list_y, list_z = get_slice(research_id, idx)

        plt.tripcolor(list_x, list_y, list_z, cmap=ui_slc.comboBox_cmap.currentText())
        plt.grid()
        figure.suptitle(f'Slice {idx}')
        figure.tight_layout()
        canvas.draw()

    def show_slice():
        draw_slice(get_research_id(), ui_slc.spinBox_slice.value())

    def export_slice():
        name = get_object_name()
        year = get_research_name().split('.')[-1]
        research_id = get_research_id()
        idx = ui_slc.spinBox_slice.value()
        list_x, list_y, list_z = get_slice(research_id, idx)

        # Получаем путь к файлу
        file_path, _ = QFileDialog.getSaveFileName(
            caption=f"Export slice {idx}",
            directory=f'{name}_{year}_{idx - 1}.txt',
            filter="Text files (*.txt)"
        )
        if not file_path:
            return
        with open(file_path, "w", newline="", encoding="utf-8") as file_handle:
            writer = csv.writer(file_handle, delimiter="\t")
            writer.writerow(["X", "Y", f"signal_{idx-1}"])
            for i in range(len(list_x)):
                writer.writerow([list_x[i], list_y[i], list_z[i]])

    show_slice()

    ui_slc.spinBox_slice.valueChanged.connect(show_slice)
    ui_slc.comboBox_cmap.activated.connect(show_slice)
    ui_slc.pushButton_export.clicked.connect(export_slice)
    SliceForm.exec_()

