import optuna
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import torch
import torch.nn as nn
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from torch import optim, randperm
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, RandomSampler, SubsetRandomSampler
from geochem import *
from func import *
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from regression import update_list_trained_models_regmod


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'data': torch.tensor(self.data[idx], dtype=torch.float),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return sample


class HiddenBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate, activation_func):
        super(HiddenBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation_func # ReLU is commonly used for regression tasks
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        out = self.batch_norm(out)
        out = self.dropout(out)
        return out

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units, dropout_rate, activation_func):
        super(Model, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_units[0])
        self.batch_norm1 = nn.BatchNorm1d(hidden_units[0])
        self.hidden_blocks = nn.ModuleList(
            [HiddenBlock(hidden_units[i], hidden_units[i + 1], dropout_rate, activation_func) for i in range(len(hidden_units) - 1)]
        )
        self.output_layer = nn.Linear(hidden_units[-1], output_dim)

    def forward(self, x):
        out = torch.relu(self.input_layer(x))
        out = self.batch_norm1(out)
        for block in self.hidden_blocks:
            out = block(out)
        out = self.output_layer(out)
        return out

def draw_results_graphs(loss, epochs):
    fig, axs = plt.subplots(1, 1, figsize=(16, 8))
    epoch = list(range(1, epochs + 1))
    axs.plot(epoch, loss, marker='o', linestyle='-', label='Val Loss')
    axs.set_xlabel('Epochs')
    axs.set_ylabel('Loss')
    axs.set_title('Loss vs Epochs')
    axs.legend()

    fig.suptitle(f'\nTrain Loss Plot: ')
    plt.subplots_adjust(top=0.8)
    plt.show()


class PyTorchRegressor:
    def __init__(self, model, input_dim, output_dim, hidden_units,
                            dropout_rate, activation_function,
                            loss_function, optimizer, learning_rate, weight_decay,
                            epochs, regular, early_stopping, patience, batch_size=20):
        self.model = model(input_dim, output_dim, hidden_units,
                           dropout_rate, activation_function)
        self.criterion = loss_function
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.epochs = epochs
        self.regular = regular
        self.early_stopping = early_stopping
        self.patience = patience
        self.batch_size = batch_size

    def fit(self, X_train, y_train):
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        losses = []
        best_loss = float('inf')
        patience = self.patience

        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0
            indices = torch.arange(X_train.shape[0])
            indices = torch.randperm(len(indices))
            batch_size = 32
            for i in range(0, len(X_train), batch_size):
                start_idx = i
                end_idx = min(start_idx + batch_size, len(X_train))
                inputs = X_train[start_idx:end_idx]
                labels = y_train[start_idx:end_idx].unsqueeze(1)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                l2_lambda = self.regular
                l2_reg = torch.tensor(0.)
                for param in self.model.parameters():
                    l2_reg += torch.norm(param)
                loss += l2_lambda * l2_reg
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {running_loss / (X_train.shape[0] / self.batch_size)}')
            if self.early_stopping:
                if loss < best_loss:
                    best_loss = loss
                    patience = self.patience
                else:
                    patience -= 1
                    if patience == 0:
                        print(f"     Epoch [{epoch}/{self.epochs}] Early stopping")
                        self.epochs = epoch
                        break
            losses.append(running_loss / (X_train.shape[0] / self.batch_size))
        draw_results_graphs(losses, self.epochs)

    def predict(self, X):
        mark_pred = []
        X = torch.from_numpy(X).float()

        self.model.eval()
        with torch.no_grad():
            pred_batch = self.model(X)
            mark_pred.extend([pred.numpy() for pred in pred_batch])
        mark = [item for m in mark_pred for item in m]
        return mark

    def score(self, X, y):
        y = np.array(y)
        y = torch.from_numpy(y).float()
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

def torch_save_regressor(pipeline, r2, list_params, text_model):
    model_name = 'torch_NN'
    result = QtWidgets.QMessageBox.question(
        MainWindow,
        'Сохранение модели',
        f'Сохранить модель {model_name}?',
        QtWidgets.QMessageBox.Yes,
        QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        path_model = f'models/regression/{model_name}_{round(r2, 3)}_{datetime.datetime.now().strftime("%d%m%y")}.pkl'
        if os.path.exists(path_model):
            path_model = f'models/regression/{model_name}_{round(r2, 3)}_{datetime.datetime.now().strftime("%d%m%y_%H%M%S")}.pkl'
        with open(path_model, 'wb') as f:
            pickle.dump(pipeline, f)

        new_trained_model = TrainedModelReg(
            analysis_id=get_regmod_id(),
            title=f'{model_name}_{round(r2, 3)}_{datetime.datetime.now().strftime("%d%m%y")}',
            path_model=path_model,
            list_params=json.dumps(list_params),
            except_signal=ui.lineEdit_signal_except_reg.text(),
            except_crl=ui.lineEdit_crl_except_reg.text(),
            comment=text_model
        )
        session.add(new_trained_model)
        session.commit()
        update_list_trained_models_regmod()
    else:
        pass

#
# def nn_torch_reg(ui_tch, training_sample, target, list_param_name):
#     x_train, x_test, y_train, y_test = train_test_split(
#         training_sample, target, test_size=0.2, random_state=42)
#     x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train),\
#         np.array(x_test), np.array(y_test)
#
#     input_dim = x_train.shape[1]
#     output_dim = 1
#
#     epochs = ui_tch.spinBox_epochs.value()
#     learning_rate = ui_tch.doubleSpinBox_choose_lr.value()
#     hidden_units = list(map(int, ui_tch.lineEdit_choose_layers.text().split()))
#     dropout_rate = ui_tch.doubleSpinBox_choose_dropout.value()
#     weight_decay = ui_tch.doubleSpinBox_choose_decay.value()
#     regular = ui_tch.doubleSpinBox_choose_reagular.value()
#
#     if ui_tch.comboBox_activation_func.currentText() == 'ReLU':
#         activation_function = nn.ReLU()
#
#     if ui_tch.comboBox_optimizer.currentText() == 'Adam':
#         optimizer = torch.optim.Adam
#     elif ui_tch.comboBox_optimizer.currentText() == 'SGD':
#         optimizer = torch.optim.SGD
#
#     if ui_tch.comboBox_loss.currentText() == 'MSE':
#         loss_function = nn.MSELoss()
#     elif ui_tch.comboBox_loss.currentText() == 'MAE':
#         loss_function = nn.L1Loss()
#     elif ui_tch.comboBox_loss.currentText() == 'HuberLoss':
#         loss_function = nn.HuberLoss()
#     elif ui_tch.comboBox_loss.currentText() == 'SmoothL1Loss':
#         loss_function = nn.SmoothL1Loss()
#
#     early_stopping = False
#     patience = 0
#     if ui_tch.checkBox_early_stop.isChecked():
#         early_stopping = True
#         patience = ui_tch.spinBox_stop_patience.value()
#
#     pipeline = Pipeline(
#         [('features', FeatureUnion([
#             ('scaler', StandardScaler())
#         ])),
#         ('regressor', PyTorchRegressor(Model, input_dim, output_dim, hidden_units,
#                             dropout_rate, activation_function,
#                             loss_function, optimizer, learning_rate, weight_decay,
#                             epochs, regular, early_stopping, patience, batch_size=20))
#     ])
#
#     start_time = datetime.datetime.now()
#
#     pipeline.fit(x_train, y_train)
#     y_pred = pipeline.predict(x_test)
#
#     r_squared = r2_score(y_test, y_pred)
#     print('Коэффициент детерминации (R²):', r_squared)
#     mse = round(mean_squared_error(y_test, y_pred), 5)
#     print('MSE: ', mse)
#
#     train_time = datetime.datetime.now() - start_time
#     print(train_time)
#
#     y_remain = [round(y_test[i] - y_pred[i], 5) for i in range(len(y_pred))]
#     data_graph = pd.DataFrame({
#         'y_test': y_test,
#         'y_pred': y_pred,
#         'y_remain': y_remain
#     })
#     fig, axes = plt.subplots(nrows=2, ncols=2)
#     fig.set_size_inches(15, 10)
#     fig.suptitle(f'Модель TorchNNRegression:\n'
#                  f' Mean Squared Error:\n {mse}, R2: {round(r_squared, 2)} \n время обучения: {train_time}')
#     sns.scatterplot(data=data_graph, x='y_test', y='y_pred', ax=axes[0, 0])
#     sns.regplot(data=data_graph, x='y_test', y='y_pred', ax=axes[0, 0])
#     sns.scatterplot(data=data_graph, x='y_test', y='y_remain', ax=axes[1, 0])
#     sns.regplot(data=data_graph, x='y_test', y='y_remain', ax=axes[1, 0])
#     try:
#         sns.histplot(data=data_graph, x='y_remain', kde=True, ax=axes[1, 1])
#     except MemoryError:
#         pass
#     fig.tight_layout()
#     fig.show()
#
#     if ui_tch.checkBox_save_model.isChecked():
#         text_model = '*** TORCH NN *** \n' + 'MSE: ' + str(round(mse, 3)) + '\nвремя обучения: ' \
#                      + str(train_time) + '\nlearning_rate: ' + str(learning_rate) + '\nhidden units: ' + str(hidden_units) \
#                      + '\nweight decay: ' + str(weight_decay) + '\ndropout rate: ' + str(dropout_rate) + \
#                      '\nregularization: ' + str(regular) + '\n'
#         torch_save_regressor(pipeline, r_squared, list_param_name, text_model)
#         print('Model saved')

def torch_regressor_train():
    TorchRegressor = QtWidgets.QDialog()
    ui_tch = Ui_TorchNNRegressor()
    ui_tch.setupUi(TorchRegressor)
    TorchRegressor.show()
    TorchRegressor.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    data_train, list_param_name = build_table_train(True, 'regmod')
    list_param_reg = get_list_param_numerical_for_train(list_param_name)

    list_nan_param, count_nan = [], 0
    for i in data_train.index:
        for param in list_param_reg:
            if pd.isna(data_train[param][i]):
                count_nan += 1
                list_nan_param.append(param)
    if count_nan > 0:
        list_col = data_train.columns.tolist()
        data_train = pd.DataFrame(imputer.fit_transform(data_train), columns=list_col)
        set_info(f'Заполнены пропуски в {count_nan} параметрах {", ".join(list_nan_param)}', 'red')

    training_sample = data_train[list_param_reg].values.tolist()
    target = sum(data_train[['target_value']].values.tolist(), [])

    x_train, x_test, y_train, y_test = train_test_split(
        training_sample, target, test_size=0.2, random_state=42)
    x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), \
        np.array(x_test), np.array(y_test)

    input_dim = x_train.shape[1]
    output_dim = 1

    def train():
        epochs = ui_tch.spinBox_epochs.value()
        learning_rate = ui_tch.doubleSpinBox_choose_lr.value()
        hidden_units = list(map(int, ui_tch.lineEdit_choose_layers.text().split()))
        dropout_rate = ui_tch.doubleSpinBox_choose_dropout.value()
        weight_decay = ui_tch.doubleSpinBox_choose_decay.value()
        regular = ui_tch.doubleSpinBox_choose_reagular.value()

        if ui_tch.comboBox_activation_func.currentText() == 'ReLU':
            activation_function = nn.ReLU()

        if ui_tch.comboBox_optimizer.currentText() == 'Adam':
            optimizer = torch.optim.Adam
        elif ui_tch.comboBox_optimizer.currentText() == 'SGD':
            optimizer = torch.optim.SGD

        if ui_tch.comboBox_loss.currentText() == 'MSE':
            loss_function = nn.MSELoss()
        elif ui_tch.comboBox_loss.currentText() == 'MAE':
            loss_function = nn.L1Loss()
        elif ui_tch.comboBox_loss.currentText() == 'HuberLoss':
            loss_function = nn.HuberLoss()
        elif ui_tch.comboBox_loss.currentText() == 'SmoothL1Loss':
            loss_function = nn.SmoothL1Loss()

        early_stopping = False
        patience = 0
        if ui_tch.checkBox_early_stop.isChecked():
            early_stopping = True
            patience = ui_tch.spinBox_stop_patience.value()

        pipeline = Pipeline(
            [('features', FeatureUnion([
                ('scaler', StandardScaler())
            ])),
             ('regressor', PyTorchRegressor(Model, input_dim, output_dim, hidden_units,
                                            dropout_rate, activation_function,
                                            loss_function, optimizer, learning_rate, weight_decay,
                                            epochs, regular, early_stopping, patience, batch_size=20))
             ])

        start_time = datetime.datetime.now()

        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        r_squared = r2_score(y_test, y_pred)
        print('Коэффициент детерминации (R²):', r_squared)
        mse = round(mean_squared_error(y_test, y_pred), 5)
        print('MSE: ', mse)

        train_time = datetime.datetime.now() - start_time
        print(train_time)

        y_remain = [round(y_test[i] - y_pred[i], 5) for i in range(len(y_pred))]
        data_graph = pd.DataFrame({
            'y_test': y_test,
            'y_pred': y_pred,
            'y_remain': y_remain
        })
        fig, axes = plt.subplots(nrows=2, ncols=2)
        fig.set_size_inches(15, 10)
        fig.suptitle(f'Модель TorchNNRegression:\n'
                     f' Mean Squared Error:\n {mse}, R2: {round(r_squared, 2)} \n время обучения: {train_time}')
        sns.scatterplot(data=data_graph, x='y_test', y='y_pred', ax=axes[0, 0])
        sns.regplot(data=data_graph, x='y_test', y='y_pred', ax=axes[0, 0])
        sns.scatterplot(data=data_graph, x='y_test', y='y_remain', ax=axes[1, 0])
        sns.regplot(data=data_graph, x='y_test', y='y_remain', ax=axes[1, 0])
        try:
            sns.histplot(data=data_graph, x='y_remain', kde=True, ax=axes[1, 1])
        except MemoryError:
            pass
        fig.tight_layout()
        fig.show()

        if ui_tch.checkBox_save_model.isChecked():
            text_model = '*** TORCH NN *** \n' + 'MSE: ' + str(round(mse, 3)) + '\nвремя обучения: ' \
                         + str(train_time) + '\nlearning_rate: ' + str(learning_rate) + '\nhidden units: ' + str(
                hidden_units) \
                         + '\nweight decay: ' + str(weight_decay) + '\ndropout rate: ' + str(dropout_rate) + \
                         '\nregularization: ' + str(regular) + '\n'
            torch_save_regressor(pipeline, r_squared, list_param_name, text_model)
            print('Model saved')

    def torch_regressor_lineup():
        epochs = ui_tch.spinBox_epochs.value()
        learning_rate = ui_tch.doubleSpinBox_choose_lr.value()
        hidden_units = list(map(int, ui_tch.lineEdit_choose_layers.text().split()))
        dropout_rate = ui_tch.doubleSpinBox_choose_dropout.value()
        weight_decay = ui_tch.doubleSpinBox_choose_decay.value()
        regular = ui_tch.doubleSpinBox_choose_reagular.value()

        if ui_tch.comboBox_activation_func.currentText() == 'ReLU':
            activation_function = nn.ReLU()

        if ui_tch.comboBox_optimizer.currentText() == 'Adam':
            optimizer = torch.optim.Adam
        elif ui_tch.comboBox_optimizer.currentText() == 'SGD':
            optimizer = torch.optim.SGD

        if ui_tch.comboBox_loss.currentText() == 'MSE':
            loss_function = nn.MSELoss()
        elif ui_tch.comboBox_loss.currentText() == 'MAE':
            loss_function = nn.L1Loss()
        elif ui_tch.comboBox_loss.currentText() == 'HuberLoss':
            loss_function = nn.HuberLoss()
        elif ui_tch.comboBox_loss.currentText() == 'SmoothL1Loss':
            loss_function = nn.SmoothL1Loss()

        early_stopping = False
        patience = 0
        if ui_tch.checkBox_early_stop.isChecked():
            early_stopping = True
            patience = ui_tch.spinBox_stop_patience.value()

        pipeline = Pipeline(
            [('features', FeatureUnion([
                ('scaler', StandardScaler())
            ])),
             ('regressor', PyTorchRegressor(Model, input_dim, output_dim, hidden_units,
                                            dropout_rate, activation_function,
                                            loss_function, optimizer, learning_rate, weight_decay,
                                            epochs, regular, early_stopping, patience, batch_size=20))
             ])

        model_name = 'torch_NN_reg'
        text_model = model_name + ' StandardScaler'

        except_reg = session.query(ExceptionReg).filter_by(analysis_id=get_regmod_id()).first()

        new_lineup = LineupTrain(
            type_ml='reg',
            analysis_id=get_regmod_id(),
            list_param=json.dumps(list_param_reg),
            list_param_short=json.dumps(list_param_name),
            except_signal=except_reg.except_signal,
            except_crl=except_reg.except_crl,
            text_model=text_model,
            model_name=model_name,
            over_sampling='none',
            pipe=pickle.dumps(pipeline)
        )
        session.add(new_lineup)
        session.commit()

        set_info(f'Модель {model_name} добавлена в очередь\n{text_model}', 'green')
        pass

    ui_tch.pushButton_train.clicked.connect(train)
    ui_tch.pushButton_lineup.clicked.connect(torch_regressor_lineup)
    TorchRegressor.exec()