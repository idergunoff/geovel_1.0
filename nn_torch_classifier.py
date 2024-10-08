import optuna
from sklearn.base import BaseEstimator
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
from build_table import *
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'data': torch.tensor(self.data[idx], dtype=torch.float),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)  # Предположим, что метки - целочисленные значения
        }
        return sample


class HiddenBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate, activation_func):
        super(HiddenBlock, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.activation = activation_func
        self.batch_norm = torch.nn.BatchNorm1d(output_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        out = self.batch_norm(out)
        out = self.dropout(out)
        return out


class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units, dropout_rate, activation_function):
        super(Model, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, hidden_units[0])
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_units[0])
        self.hidden_blocks = torch.nn.ModuleList(
            [HiddenBlock(hidden_units[i], hidden_units[i + 1], dropout_rate, activation_function) for i in range(len(hidden_units) - 1)]
        )
        self.output_layer = torch.nn.Linear(hidden_units[-1], output_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out = torch.relu(self.input_layer(x))
        out = self.batch_norm1(out)
        for block in self.hidden_blocks:
            out = block(out)
        out = self.output_layer(out)
        out = self.sigmoid(out)
        return out


def calculate_metrics(ui_tch, model, val_dl, threshold=False):
    model.eval()
    all_targets = []
    all_predictions = []
    predictions = []
    if ui_tch.comboBox_dataset.currentText() == 'shuffle':
        with torch.no_grad():
            for xb, yb in val_dl:
                pred_batch = model(xb)
                print('XB: ', xb)
                all_targets.extend([y.numpy() for y in yb])
                all_predictions.extend([pred.numpy() for pred in pred_batch])
                predictions.extend([np.hstack((pred.numpy(), 1 - pred.numpy())) for pred in pred_batch])

    elif ui_tch.comboBox_dataset.currentText() == 'well split':
        with torch.no_grad():
            for batch in val_dl:
                xb, yb = batch['data'], batch['label'].unsqueeze(1)
                pred = model(xb)
                comp = 1 - pred
                pred_array = np.array([comp.numpy(), pred.numpy()])
                all_targets.extend(yb.numpy())
                all_predictions.append(pred_array)
    print('predictions ', predictions)
    # Применяем порог к предсказаниям, если это необходимо
    if threshold is True:
        fpr, tpr, thresholds = roc_curve(all_targets, all_predictions)
        opt_threshold = thresholds[np.argmax(tpr - fpr)]
    else:
        opt_threshold = 0.5
    all_predictions = [1 if p >= opt_threshold else 0 for p in all_predictions]

    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)

    return accuracy, precision, recall, f1


def plot_roc_curve(ui_tch, model, val_dl):
    model.eval()
    all_targets = []
    all_predictions = []
    if ui_tch.comboBox_dataset.currentText() == 'shuffle':
        with torch.no_grad():
            for xb, yb in val_dl:
                pred = model(xb)
                all_targets.extend(yb.numpy())
                all_predictions.extend(pred.numpy())
    elif ui_tch.comboBox_dataset.currentText() == 'well split':
        with torch.no_grad():
            for batch in val_dl:
                xb, yb = batch['data'], batch['label'].unsqueeze(1)
                pred = model(xb)
                all_targets.extend(yb.numpy())
                all_predictions.extend(pred.numpy())

    fpr, tpr, _ = roc_curve(all_targets, all_predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()

def fit_split(epochs, model, optimizer, loss_function, regular, train_dl, val_dl, ui_tch, early_stopping=False):
    best_loss = float('inf')
    patience = 20
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        epoch_train_loss = 0.0
        model.train()
        for batch in train_dl:
            xb, yb = batch['data'], batch['label'].unsqueeze(1)
            pred = model(xb)
            loss = loss_function(pred, yb.float())
            l2_lambda = regular
            l2_reg = torch.tensor(0.)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_train_loss += loss.item()
        train_losses.append(epoch_train_loss / len(train_dl))

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                xb, yb = batch['data'], batch['label'].unsqueeze(1)
                predictions = model(xb)
                val_loss = loss_function(predictions, yb.float())
                l2_lambda = regular
                l2_reg = torch.tensor(0.)
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                val_loss += l2_lambda * l2_reg
                epoch_val_loss += val_loss.item()
            val_losses.append(epoch_val_loss / len(val_dl))
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

        # early stopping
        if early_stopping:
            if val_loss < best_loss:
                best_loss = val_loss
                patience = 20
            else:
                patience -= 1
                if patience == 0:
                    print(f"     Epoch [{epoch+1}/{epochs}] Early stopping")
                    break
    return train_losses, val_losses


def fit_shuffle(epochs, model, optimizer, loss_function, regular, train_dl, val_dl, ui_tch, early_stopping=False):
    best_loss = float('inf')
    patience = 20
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        epoch_train_loss = 0.0
        model.train()
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_function(pred, yb)
            l2_lambda = regular
            l2_reg = torch.tensor(0.)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_train_loss += loss.item()
        train_losses.append(epoch_train_loss / len(train_dl))

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                predictions = model(xb)
                val_loss = loss_function(predictions, yb)
                l2_lambda = regular
                l2_reg = torch.tensor(0.)
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                val_loss += l2_lambda * l2_reg
                epoch_val_loss += val_loss.item()
            val_losses.append(epoch_val_loss / len(val_dl))

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

        # early stopping
        if early_stopping:
            if val_loss < best_loss:
                best_loss = val_loss
                patience = 20
            else:
                patience -= 1
                if patience == 0:
                    print(f"     Epoch [{epoch + 1}/{epochs}] Early stopping")
                    break
    return train_losses, val_losses


def fit_optuna_shuffle(epochs, learning_rate, num_hidden_units, dropout_rate, weight_decay,
               regularization, trial, input_dim, output_dim, ui_tch, train_dl, val_dl):
    print("Trial number:", trial.number)
    if ui_tch.comboBox_activation_func.currentText() == 'ReLU':
        activation_function = nn.ReLU()
    elif ui_tch.comboBox_activation_func.currentText() == 'Sigmoid':
        activation_function = nn.Sigmoid()
    elif ui_tch.comboBox_activation_func.currentText() == 'Tanh':
        activation_function = nn.Tanh()
    model = Model(input_dim, output_dim, num_hidden_units, dropout_rate, activation_function)
    if ui_tch.comboBox_optimizer.currentText() == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif ui_tch.comboBox_optimizer.currentText() == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if ui_tch.comboBox_loss.currentText() == 'CrossEntropy':
        loss_function = nn.CrossEntropyLoss()
    elif ui_tch.comboBox_loss.currentText() == 'BCEWithLogitsLoss':
        loss_function = nn.BCEWithLogitsLoss()
    elif ui_tch.comboBox_loss.currentText() == 'BCELoss':
        loss_function = nn.BCELoss()

    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for xb, yb in train_dl:
            pred = model(xb)
            if ui_tch.comboBox_loss.currentText() == 'CrossEntropy' or ui_tch.comboBox_loss.currentText() == 'BCELoss':
                pred = torch.sigmoid(pred)
            loss = loss_function(pred, yb)

            l2_lambda = regularization
            l2_reg = torch.tensor(0.)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_losses.append(loss.item())
        epoch_loss = sum(epoch_losses) / len(epoch_losses)  # Вычисляем среднюю потерю для текущей эпохи
        losses.append(epoch_loss)
        if epoch % 10 == 0:
            print(f'    Epoch [{epoch}/{epochs}], Loss: {loss.item():.7f}')
        trial.report(epoch_loss, epoch)
        if trial.should_prune():
            print(f'        Epoch [{epoch}/{epochs}], Trial pruned by optuna')
            raise optuna.TrialPruned()

    model.eval()
    # Список для хранения всех прогнозов
    all_predictions = []
    all_targets = []
    # Проходим по DataLoader и делаем прогнозы
    with torch.no_grad():
        for xb, yb in val_dl:
            # Получаем прогнозы от модели
            predictions = model(xb)
            # Добавляем прогнозы в общий список
            all_predictions.append(predictions)
            all_targets.append(yb)
    # Объединяем все прогнозы в один тензор
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    # Преобразуем тензор прогнозов в массив NumPy, если нужно
    predictions_numpy = all_predictions.numpy()
    targets_numpy = all_targets.numpy()
    binary_predictions = (predictions_numpy >= 0.5).astype(int)
    accuracy = accuracy_score(targets_numpy, binary_predictions)
    return accuracy


def fit_optuna_split(epochs, learning_rate, num_hidden_units, dropout_rate, weight_decay,
               regularization, trial, input_dim, output_dim, ui_tch, train_dl, val_dl):
    print("Trial number:", trial.number)
    if ui_tch.comboBox_activation_func.currentText() == 'ReLU':
        activation_function = nn.ReLU()
    elif ui_tch.comboBox_activation_func.currentText() == 'Sigmoid':
        activation_function = nn.Sigmoid()
    elif ui_tch.comboBox_activation_func.currentText() == 'Tanh':
        activation_function = nn.Tanh()
    model = Model(input_dim, output_dim, num_hidden_units, dropout_rate, activation_function)
    if ui_tch.comboBox_optimizer.currentText() == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif ui_tch.comboBox_optimizer.currentText() == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if ui_tch.comboBox_loss.currentText() == 'CrossEntropy':
        loss_function = nn.CrossEntropyLoss()
    elif ui_tch.comboBox_loss.currentText() == 'BCEWithLogitsLoss':
        loss_function = nn.BCEWithLogitsLoss()
    elif ui_tch.comboBox_loss.currentText() == 'BCELoss':
        loss_function = nn.BCELoss()

    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for batch in train_dl:
            xb, yb = batch['data'], batch['label'].unsqueeze(1)
            pred = model(xb)
            if ui_tch.comboBox_loss.currentText() == 'CrossEntropy' or ui_tch.comboBox_loss.currentText() == 'BCELoss':
                pred = torch.sigmoid(pred)
            loss = loss_function(pred, yb.float())

            l2_lambda = regularization
            l2_reg = torch.tensor(0.)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_losses.append(loss.item())
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(epoch_loss)
        if epoch % 10 == 0:
            print(f'    Epoch [{epoch}/{epochs}], Loss: {loss.item():.7f}')
        trial.report(epoch_loss, epoch)
        if trial.should_prune():
            print(f'        Epoch [{epoch}/{epochs}], Trial pruned by optuna')
            raise optuna.TrialPruned()

    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch in val_dl:
            xb, yb = batch['data'], batch['label'].unsqueeze(1)
            predictions = model(xb)
            all_predictions.append(predictions)
            all_targets.append(yb)
    # Объединяем все прогнозы в один тензор
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    # Преобразуем тензор прогнозов в массив NumPy, если нужно
    predictions_numpy = all_predictions.numpy()
    targets_numpy = all_targets.numpy()
    binary_predictions = (predictions_numpy >= 0.5).astype(int)
    accuracy = accuracy_score(targets_numpy, binary_predictions)
    return accuracy

def fit_best_model_shuffle(epochs, final_model, loss_function, optimizer,
                           regularization, train_dl, val_dl, trial, ui_tch,
                           early_stopping=False):
    best_loss = float('inf')
    patience = 20
    losses = []
    val_losses = []
    epoch_val = []

    for epoch in range(epochs):
        epoch_losses = []
        final_model.train()
        for xb, yb in train_dl:
            pred = final_model(xb)
            if ui_tch.comboBox_loss.currentText() == 'CrossEntropy' or ui_tch.comboBox_loss.currentText() == 'BCELoss':
                pred = torch.sigmoid(pred)
            loss = loss_function(pred, yb)

            l2_lambda = regularization
            l2_reg = torch.tensor(0.)
            for param in final_model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_losses.append(loss.item())
        losses.append(np.mean(epoch_losses))

        final_model.eval()
        with torch.no_grad():
            for xb, yb in val_dl:
                predictions = final_model(xb)
                if ui_tch.comboBox_loss.currentText() == 'CrossEntropy' or ui_tch.comboBox_loss.currentText() == 'BCELoss':
                    predictions = torch.sigmoid(predictions)
                val_loss = loss_function(predictions, yb)
                epoch_val.append(val_loss)
            val_losses.append(np.mean(epoch_val))

        print(f'-- TEST -- Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.7f}, Val Loss: {val_losses[-1]:.7f}')

        # pruner
        trial.report(val_loss, epoch)
        if trial.should_prune():
            print(f'    VAL Epoch [{epoch + 1}/{epochs}], Trial pruned by optuna')
            raise optuna.TrialPruned()
        if early_stopping:
            if val_loss < best_loss:
                best_loss = val_loss
                patience = 20
            else:
                patience -= 1
                if patience == 0:
                    print(f"     Epoch [{epoch + 1}/{epochs}] Early stopping")
                    break

    return losses, val_losses

def fit_best_model_split(epochs, final_model, loss_function, optimizer,
                           regularization, train_dl, val_dl, trial,ui_tch, early_stopping=False):
    best_loss = float('inf')
    patience = 20
    losses = []
    val_losses = []
    epoch_val = []

    for epoch in range(epochs):
        epoch_losses = []
        final_model.train()
        for batch in train_dl:
            xb, yb = batch['data'], batch['label'].unsqueeze(1)
            pred = final_model(xb)
            if ui_tch.comboBox_loss.currentText() == 'CrossEntropy' or ui_tch.comboBox_loss.currentText() == 'BCELoss':
                pred = torch.sigmoid(pred)
            loss = loss_function(pred, yb.float())

            l2_lambda = regularization
            l2_reg = torch.tensor(0.)
            for param in final_model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_losses.append(loss.item())
        losses.append(np.mean(epoch_losses))

        final_model.eval()
        with torch.no_grad():
            for batch in val_dl:
                xb, yb = batch['data'], batch['label'].unsqueeze(1)
                predictions = final_model(xb)
                if ui_tch.comboBox_loss.currentText() == 'CrossEntropy' or ui_tch.comboBox_loss.currentText() == 'BCELoss':
                    predictions = torch.sigmoid(predictions)
                val_loss = loss_function(predictions, yb.float())
                epoch_val.append(val_loss)
            val_losses.append(np.mean(epoch_val))

        print(f'-- TEST -- Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.7f}, Val Loss: {val_losses[-1]:.7f}')

        # pruner
        trial.report(val_loss, epoch)
        if trial.should_prune():
            print(f'    VAL Epoch [{epoch + 1}/{epochs}], Trial pruned by optuna')
            raise optuna.TrialPruned()
        if early_stopping:
            if val_loss < best_loss:
                best_loss = val_loss
                patience = 20
            else:
                patience -= 1
                if patience == 0:
                    print(f"     Epoch [{epoch + 1}/{epochs}] Early stopping")
                    break
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 20
        else:
            patience -= 1
            if patience == 0:
                print(f"     Epoch [{epoch + 1}/{epochs}] Early stopping")
                break

    return losses, val_losses


def predictions_to_binary(all_predictions, all_targets, threshold=False):
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    predictions_numpy = all_predictions.numpy()
    targets_numpy = all_targets.numpy()
    fpr, tpr, thresholds = roc_curve(targets_numpy, predictions_numpy)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    if threshold:
        binary_predictions = (predictions_numpy >= optimal_threshold).astype(int)
    else:
        binary_predictions = (predictions_numpy >= 0.5).astype(int)
    return binary_predictions, targets_numpy

def torch_save_model(model, accuracy, list_params, text_model):
    model_name = 'torch_NN'
    result = QtWidgets.QMessageBox.question(
        MainWindow,
        'Сохранение модели',
        f'Сохранить модель {model_name}?',
        QtWidgets.QMessageBox.Yes,
        QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        # Сохранение модели в файл с помощью pickle
        path_model = f'models/classifier/{model_name}_{round(accuracy, 3)}_{datetime.datetime.now().strftime("%d%m%y")}.pth'
        if os.path.exists(path_model):
            path_model = f'models/classifier/{model_name}_{round(accuracy, 3)}_{datetime.datetime.now().strftime("%d%m%y_%H%M%S")}.pth'
        torch.save(model, path_model)

        new_trained_model = TrainedModelClass(
            analysis_id=get_MLP_id(),
            title=f'{model_name}_{round(accuracy, 3)}_{datetime.datetime.now().strftime("%d%m%y")}',
            path_model=path_model,
            list_params=json.dumps(list_params),
            except_signal=ui.lineEdit_signal_except.text(),
            except_crl=ui.lineEdit_crl_except.text(),
            comment=text_model
        )
        session.add(new_trained_model)
        session.commit()
        update_list_trained_models_class()
    else:
        pass


def nn_choose_params(ui_tch, data, list_param):
    if ui_tch.comboBox_dataset.currentText() == 'well split':
        batch_size = 50
        list_param = data.columns[2:].to_list()
        training_sample_train, training_sample_test,\
            markup_train, markup_test = train_test_split_cvw(data,
                    [0, 1], 'mark', list_param, random_seed=ui.spinBox_seed.value(), test_size=0.2)
        train_dataset = CustomDataset(training_sample_train, markup_train)
        val_dataset = CustomDataset(training_sample_test, markup_test)
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for batch in train_dl:
            xb, yb = batch['data'], batch['label']
            break
        input_dim = xb.size(1)
        output_dim = 1
    elif ui_tch.comboBox_dataset.currentText() == 'shuffle':
        input_cols = list(data.columns[2:])
        output_cols = ['mark']

        def dataframe_to_arrays(data):
            df = data.copy(deep=True)
            input_array = df[input_cols].to_numpy()
            input_scaler = StandardScaler()
            input_array = input_scaler.fit_transform(input_array)
            target_array = df[output_cols].to_numpy()
            return input_array, target_array

        inputs_array, targets_array = dataframe_to_arrays(data)
        inputs = torch.from_numpy(inputs_array).type(torch.float)
        targets = torch.from_numpy(targets_array).type(torch.float)

        dataset = TensorDataset(inputs, targets)
        train_ds, val_ds = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(10))
        batch_size = 50
        train_dl = DataLoader(train_ds, batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size)
        input_dim = len(train_ds[0][0])
        output_dim = 1

    epochs = ui_tch.spinBox_epochs.value()
    learning_rate = ui_tch.doubleSpinBox_choose_lr.value()
    hidden_units = list(map(int, ui_tch.lineEdit_choose_layers.text().split()))
    dropout_rate = ui_tch.doubleSpinBox_choose_dropout.value()
    weight_decay = ui_tch.doubleSpinBox_choose_decay.value()
    regularization = ui_tch.doubleSpinBox_choose_reagular.value()

    if ui_tch.comboBox_activation_func.currentText() == 'ReLU':
        activation_function = nn.ReLU()
    elif ui_tch.comboBox_activation_func.currentText() == 'Sigmoid':
        activation_function = nn.Sigmoid()
    elif ui_tch.comboBox_activation_func.currentText() == 'Tanh':
        activation_function = nn.Tanh()

    model = Model(input_dim, output_dim, hidden_units, dropout_rate, activation_function)
    if ui_tch.comboBox_optimizer.currentText() == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif ui_tch.comboBox_optimizer.currentText() == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif ui_tch.comboBox_optimizer.currentText() == 'LBFGS':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate)

    if ui_tch.comboBox_loss.currentText() == 'CrossEntropy':
        loss_function = nn.CrossEntropyLoss()
    elif ui_tch.comboBox_loss.currentText() == 'BCEWithLogitsLoss':
        loss_function = nn.BCEWithLogitsLoss()
    elif ui_tch.comboBox_loss.currentText() == 'BCELoss':
        loss_function = nn.BCELoss()

    early_stopping = False
    threshold = False
    if ui_tch.checkBox_early_stop.isChecked():
        early_stopping = True
    if ui_tch.checkBox_threshold.isChecked():
        threshold = True
    start_time = datetime.datetime.now()
    if ui_tch.comboBox_dataset.currentText() == 'well split':
        losses, val_losses = fit_split(epochs, model, optimizer, loss_function,
                                                                    regularization, train_dl, val_dl,
                                                                    ui_tch, early_stopping=early_stopping)
    elif ui_tch.comboBox_dataset.currentText() == 'shuffle':
        losses, val_losses = fit_shuffle(epochs, model, optimizer, loss_function,
                                                                     regularization, train_dl, val_dl,
                                                                     ui_tch, early_stopping=early_stopping)


    ###  Результаты обучения ###
    accuracy, precision, recall, f1 = calculate_metrics(ui_tch, model, val_dl, threshold=threshold)
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
    end_time = datetime.datetime.now() - start_time

    # Построение графиков
    plot_roc_curve(ui_tch, model, val_dl)
    epochs = range(1, len(losses) + 1)
    val_epoch = range(1, len(val_losses) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    axs[0].plot(epochs, losses, marker='o', linestyle='-', label='Train Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Loss vs Epochs')
    axs[0].legend()

    axs[1].plot(val_epoch, val_losses, marker='o', linestyle='-', label='Val Loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Val Loss')
    axs[1].set_title('Val Loss vs Epochs')
    axs[1].legend()

    fig.suptitle(f'Accuracy: {accuracy:.4f}\n Precision: {precision:.4f}\n Recall: {recall:.4f}\n F1-Score: {f1:.4f}')
    plt.subplots_adjust(top=0.8)
    plt.show()

    # Сохранение модели
    if ui_tch.checkBox_save_model.isChecked():
        text_model = '*** TORCH NN *** \n' + 'test_accuray: ' + str(round(accuracy, 3)) + '\nвремя обучения: ' \
                     + str(end_time) + '\nlearning_rate: ' + str(learning_rate) + '\nhidden units: ' + str(hidden_units) \
                    + '\nweight decay: ' + str(weight_decay) + '\ndropout rate: ' + str(dropout_rate) + \
                     '\nregularization: ' + str(regularization) + '\n'
        torch_save_model(model, accuracy, list_param, text_model)
        print("model saved")


def nn_tune_params(ui_tch, data, list_param):
    if ui_tch.comboBox_dataset.currentText() == 'well split':
        batch_size = 50
        # list_param = data.columns[2:].to_list()
        training_sample_train, training_sample_test, \
            markup_train, markup_test = train_test_split_cvw(data,
                                                             [0, 1], 'mark', list_param,
                                                             random_seed=ui.spinBox_seed.value(), test_size=0.2)
        train_dataset = CustomDataset(training_sample_train, markup_train)
        val_dataset = CustomDataset(training_sample_test, markup_test)
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        for batch in train_dl:
            xb, yb = batch['data'], batch['label']
            break
        input_dim = xb.size(1)
        output_dim = 1
    elif ui_tch.comboBox_dataset.currentText() == 'shuffle':
        input_cols = list(data.columns[2:])
        output_cols = ['mark']

        def dataframe_to_arrays(data):
            df = data.copy(deep=True)
            input_array = df[input_cols].to_numpy()
            input_scaler = StandardScaler()
            input_array = input_scaler.fit_transform(input_array)
            target_array = df[output_cols].to_numpy()
            return input_array, target_array

        inputs_array, targets_array = dataframe_to_arrays(data)
        inputs = torch.from_numpy(inputs_array).type(torch.float)
        targets = torch.from_numpy(targets_array).type(torch.float)

        dataset = TensorDataset(inputs, targets)
        train_ds, val_ds = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(10))
        batch_size = 50
        train_dl = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)
        val_dl = DataLoader(val_ds, batch_size, drop_last=True)
        input_dim = len(train_ds[0][0])
        output_dim = 1

    epochs = ui_tch.spinBox_epochs.value()
    learning_rate = str_to_interval(ui_tch.lineEdit_tune_lr.text())
    dropout_rate = str_to_interval(ui_tch.lineEdit_tune_dropout.text())
    weight_decay = str_to_interval(ui_tch.lineEdit_tune_decay.text())
    regularization = ui_tch.doubleSpinBox_choose_reagular.value()
    hidden_units = str_to_interval(ui_tch.lineEdit_tune_layers.text())
    layers_num = ui_tch.spinBox_layers_num.value()

    def objective(trial):
        op_learning_rate = trial.suggest_float('learning_rate', learning_rate[0], learning_rate[1], log=True)
        op_num_hidden_units = [trial.suggest_int(f'num_hidden_units_layer{i}',
                                                 hidden_units[0], hidden_units[1]) for i in range(1, layers_num)]
        op_dropout_rate = trial.suggest_float('dropout_rate', dropout_rate[0], dropout_rate[1], log=True)
        op_weight_decay = trial.suggest_float('weight_decay', weight_decay[0], weight_decay[1], log=True)
        op_regularization = trial.suggest_float('regularization', regularization, regularization + 0.1, log=True)

        if ui_tch.comboBox_dataset.currentText() == 'well split':
            accuracy = fit_optuna_split(epochs, op_learning_rate, op_num_hidden_units,
                                  op_dropout_rate, op_weight_decay, op_regularization, trial,
                                  input_dim, output_dim, ui_tch, train_dl, val_dl)
        elif ui_tch.comboBox_dataset.currentText() == 'shuffle':
            accuracy = fit_optuna_shuffle(epochs, op_learning_rate, op_num_hidden_units,
                                  op_dropout_rate, op_weight_decay, op_regularization, trial,
                                  input_dim, output_dim, ui_tch, train_dl, val_dl)
        return accuracy

    num_trials = ui_tch.spinBox_trials.value()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize',
                                pruner=optuna.pruners.MedianPruner(
                                    n_startup_trials=2, n_warmup_steps=20, interval_steps=1
                                ), sampler=optuna.samplers.RandomSampler(seed=10))
    study.optimize(objective, n_trials=num_trials)

    print("Number of finished trials:", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("Value: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    best_learning_rate = trial.params['learning_rate']
    best_num_hidden_units = [trial.params[f'num_hidden_units_layer{i}'] for i in range(1, layers_num)]
    best_dropout_rate = trial.params['dropout_rate']
    best_weight_decay = trial.params['weight_decay']
    best_regularization = trial.params['regularization']

    epochs = ui_tch.spinBox_epochs.value()
    if ui_tch.comboBox_activation_func.currentText() == 'ReLU':
        activation_function = nn.ReLU()
    elif ui_tch.comboBox_activation_func.currentText() == 'Sigmoid':
        activation_function = nn.Sigmoid()
    elif ui_tch.comboBox_activation_func.currentText() == 'Tanh':
        activation_function = nn.Tanh()

    final_model = Model(input_dim, output_dim, best_num_hidden_units, best_dropout_rate, activation_function)
    if ui_tch.comboBox_optimizer.currentText() == 'Adam':
        optimizer = torch.optim.Adam(final_model.parameters(), lr=best_learning_rate, weight_decay=best_weight_decay)
    elif ui_tch.comboBox_optimizer.currentText() == 'SGD':
        optimizer = torch.optim.SGD(final_model.parameters(), lr=best_learning_rate, weight_decay=best_weight_decay)

    if ui_tch.comboBox_loss.currentText() == 'BCEWithLogitsLoss':
        loss_function = torch.nn.BCEWithLogitsLoss()
    elif ui_tch.comboBox_loss.currentText() == 'CrossEntropy':
        loss_function = torch.nn.CrossEntropyLoss()
    elif ui_tch.comboBox_loss.currentText() == 'BCELoss':
        loss_function = torch.nn.BCELoss()

    early_stopping = False
    threshold = False
    if ui_tch.checkBox_early_stop.isChecked():
        early_stopping = True
    if ui_tch.checkBox_threshold.isChecked():
        threshold = True

    start_time = datetime.datetime.now()
    if ui_tch.comboBox_dataset.currentText() == 'well split':
        losses, val_losses= fit_best_model_split(epochs, final_model, loss_function,
                                                            optimizer, best_regularization, train_dl, val_dl, trial,
                                                            ui_tch, early_stopping=early_stopping)
    elif ui_tch.comboBox_dataset.currentText() == 'shuffle':
        losses, val_losses= fit_best_model_shuffle(epochs, final_model, loss_function,
                                                            optimizer, best_regularization, train_dl, val_dl, trial,
                                                            ui_tch, early_stopping=early_stopping)
    accuracy, precision, recall, f1 = calculate_metrics(ui_tch, final_model, val_dl, threshold)
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
    end_time = datetime.datetime.now() - start_time

    # Построение графиков
    plot_roc_curve(ui_tch, final_model, val_dl)
    epochs = range(1, len(losses) + 1)
    val_epoch = range(1, len(val_losses) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    axs[0].plot(epochs, losses, marker='o', linestyle='-', label='Train Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Loss vs Epochs')
    axs[0].legend()

    axs[1].plot(val_epoch, val_losses, marker='o', linestyle='-', label='Val Loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Val Loss')
    axs[1].set_title('Val Loss vs Epochs')
    axs[1].legend()

    fig.suptitle(f'Accuracy: {accuracy:.4f}\n Precision: {precision:.4f}\n Recall: {recall:.4f}\n F1-Score: {f1:.4f}')
    plt.subplots_adjust(top=0.8)
    plt.show()

    if ui_tch.checkBox_save_model.isChecked():
        text_model = '*** TORCH NN *** \n' + 'test_accuray: ' + str(round(accuracy, 3)) + '\nвремя обучения: ' \
                     + str(end_time) + '\nParams: \n'
        for key, value in trial.params.items():
            text_model += str(f"    {key}: {value}\n")
        torch_save_model(final_model, accuracy, list_param, text_model)
        print("model saved")


def cross_validation_shuffle(dataset, epochs, model, optimizer, loss_function,
                             regularization, ui_tch, threshold=False):
    print(f'----------------- CROSS VALIVATION -----------------\n')
    k_folds = ui_tch.spinBox_cross_val.value()
    kfold = KFold(n_splits=k_folds, shuffle=True)
    accuracy_list = []

    loss_list = []
    for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        train_subsampler = SubsetRandomSampler(train_ids)
        valid_subsampler = SubsetRandomSampler(valid_ids)

        train_loader = DataLoader(dataset, batch_size=99, sampler=train_subsampler)
        valid_loader = DataLoader(dataset, batch_size=99, sampler=valid_subsampler)

        losses, val_losses = fit_shuffle(epochs, model, optimizer, loss_function, regularization,
                                         train_loader, valid_loader, ui_tch)
        loss_list.append(val_losses)

        accuracy, precision, recall, f1 = calculate_metrics(ui_tch, model, valid_loader, threshold)
        print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
        accuracy_list.append(accuracy)

    flattened_list = [item for sublist in loss_list for item in sublist]
    print('Training Ended')
    print('Average Loss: {}\n'.format(np.mean(flattened_list)))

    plt.figure(figsize=(8, 6))
    bars = plt.bar(range(1, len(accuracy_list) + 1), accuracy_list, color='skyblue')
    for bar, value in zip(bars, accuracy_list):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=10)
    plt.title('Cross Validation')
    plt.xlabel('Folds')
    plt.ylabel('Accuracy')
    plt.show()


def cross_validation_split(train_dataset, val_dataset, epochs, model, optimizer,
                           loss_function, regularization, ui_tch, threshold=False):
    print(f'----------------- CROSS VALIVATION -----------------\n')
    k_folds = ui_tch.spinBox_cross_val.value()
    kfold = KFold(n_splits=k_folds, shuffle=True)
    accuracy_list = []
    batch_size = 50

    loss_list = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        train_sampler = SubsetRandomSampler(train_idx)
        train_dl = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

        val_sampler = SubsetRandomSampler(val_idx)
        val_dl = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

        losses, val_losses = fit_split(epochs, model, optimizer, loss_function, regularization,
                                       train_dl, val_dl, ui_tch)
        loss_list.append(val_losses)

        accuracy, precision, recall, f1 = calculate_metrics(ui_tch, model, val_dl, threshold)
        print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
        accuracy_list.append(accuracy)

    flattened_list = [item for sublist in loss_list for item in sublist]
    print('Training Ended')
    print('Average Loss: {}\n'.format(np.mean(flattened_list)))

    plt.figure(figsize=(8, 6))
    bars = plt.bar(range(1, len(accuracy_list) + 1), accuracy_list, color='skyblue')
    for bar, value in zip(bars, accuracy_list):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=10)
    plt.title('Cross Validation')
    plt.xlabel('Folds')
    plt.ylabel('Accuracy')
    plt.show()


def nn_cross_val(ui_tch, data):
    # Подготовка данных
    if ui_tch.comboBox_dataset.currentText() == 'shuffle':
        input_cols = list(data.columns[2:])
        output_cols = ['mark']

        def dataframe_to_arrays(data):
            df = data.copy(deep=True)
            input_array = df[input_cols].to_numpy()
            input_scaler = StandardScaler()
            input_array = input_scaler.fit_transform(input_array)
            target_array = df[output_cols].to_numpy()
            return input_array, target_array

        inputs_array, targets_array = dataframe_to_arrays(data)
        inputs = torch.from_numpy(inputs_array).type(torch.float)
        targets = torch.from_numpy(targets_array).type(torch.float)

        dataset = TensorDataset(inputs, targets)
        train_ds, val_ds = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(10))
        batch_size = 50
        train_dl = DataLoader(train_ds, batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size)
        input_dim = len(train_ds[0][0])
        output_dim = 1
    elif ui_tch.comboBox_dataset.currentText() == 'well split':
        batch_size = 50
        list_param = data.columns[2:].to_list()
        training_sample_train, training_sample_test,\
            markup_train, markup_test = train_test_split_cvw(data,
                    [0, 1], 'mark', list_param, random_seed=ui.spinBox_seed.value(), test_size=0.2)
        train_dataset = CustomDataset(training_sample_train, markup_train)
        val_dataset = CustomDataset(training_sample_test, markup_test)
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for batch in train_dl:
            xb, yb = batch['data'], batch['label']
            break
        input_dim = xb.size(1)
        output_dim = 1

    # Параметры модели
    epochs = ui_tch.spinBox_epochs.value()
    learning_rate = ui_tch.doubleSpinBox_choose_lr.value()
    hidden_units = list(map(int, ui_tch.lineEdit_choose_layers.text().split()))
    dropout_rate = ui_tch.doubleSpinBox_choose_dropout.value()
    weight_decay = ui_tch.doubleSpinBox_choose_decay.value()
    regularization = ui_tch.doubleSpinBox_choose_reagular.value()

    if ui_tch.comboBox_activation_func.currentText() == 'ReLU':
        activation_function = nn.ReLU()
    elif ui_tch.comboBox_activation_func.currentText() == 'Sigmoid':
        activation_function = nn.Sigmoid()
    elif ui_tch.comboBox_activation_func.currentText() == 'Tanh':
        activation_function = nn.Tanh()

    model = Model(input_dim, output_dim, hidden_units, dropout_rate, activation_function)
    if ui_tch.comboBox_optimizer.currentText() == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif ui_tch.comboBox_optimizer.currentText() == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if ui_tch.comboBox_loss.currentText() == 'CrossEntropy':
        loss_function = nn.CrossEntropyLoss()
    elif ui_tch.comboBox_loss.currentText() == 'BCEWithLogitsLoss':
        loss_function = nn.BCEWithLogitsLoss()
    elif ui_tch.comboBox_loss.currentText() == 'BCELoss':
        loss_function = nn.BCELoss()

    threshold = False
    if ui_tch.checkBox_threshold.isChecked():
        threshold = True

    # Cross-validation
    if ui_tch.comboBox_dataset.currentText() == 'shuffle':
        cross_validation_shuffle(dataset, epochs, model, optimizer,
                                 loss_function, regularization, ui_tch, threshold=threshold)
    else:
        pass
        # cross_validation_split(train_dataset, val_dataset, epochs, model, optimizer,
        #                          loss_function, regularization, ui_tch, threshold=threshold)

def torch_save_classifier(pipeline, accuracy, list_params, text_model):
    model_name = 'torch_NN'
    result = QtWidgets.QMessageBox.question(
        MainWindow,
        'Сохранение модели',
        f'Сохранить модель {model_name}?',
        QtWidgets.QMessageBox.Yes,
        QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        # Сохранение модели в файл с помощью pickle
        path_model = f'models/classifier/{model_name}_{round(accuracy, 3)}_{datetime.datetime.now().strftime("%d%m%y")}.pkl'
        if os.path.exists(path_model):
            path_model = f'models/classifier/{model_name}_{round(accuracy, 3)}_{datetime.datetime.now().strftime("%d%m%y_%H%M%S")}.pkl'
        with open(path_model, 'wb') as f:
            pickle.dump(pipeline, f)

        new_trained_model = TrainedModelClass(
            analysis_id=get_MLP_id(),
            title=f'{model_name}_{round(accuracy, 3)}_{datetime.datetime.now().strftime("%d%m%y")}',
            path_model=path_model,
            list_params=json.dumps(list_params),
            except_signal=ui.lineEdit_signal_except.text(),
            except_crl=ui.lineEdit_crl_except.text(),
            comment=text_model
        )
        session.add(new_trained_model)
        session.commit()
        update_list_trained_models_class()
    else:
        pass


def draw_results_graphs(loss, epochs, learning_rate, hidden_units, weight_decay, dropout_rate, regular):
    fig, axs = plt.subplots(1, 1, figsize=(16, 8))
    epoch = list(range(1, epochs + 1))
    axs.plot(epoch, loss, marker='o', linestyle='-', label='Val Loss')
    axs.set_xlabel('Epochs')
    axs.set_ylabel('Loss')
    axs.legend()

    fig.suptitle(f'\nTrain Loss Plot: ' + '\nlearning_rate: ' + str(learning_rate) + '\nhidden units: ' + str(hidden_units) \
                         + '\nweight decay: ' + str(weight_decay) + '\ndropout rate: ' + str(dropout_rate) + \
                         '\nregularization: ' + str(regular))
    plt.subplots_adjust(top=0.8)
    plt.show()


class PyTorchClassifier(BaseEstimator):
    def __init__(self, model, input_dim, output_dim, hidden_units,
                            dropout_rate, activation_function,
                            loss_function, optimizer, learning_rate, weight_decay,
                            epochs, regular, early_stopping, patience, labels, batch_size=20):
        self.model = model(input_dim, output_dim, hidden_units,
                           dropout_rate, activation_function)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.activation_function = activation_function
        self.criterion = loss_function
        self.weight_decay = weight_decay
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regular = regular
        self.early_stopping = early_stopping
        self.patience = patience
        self.labels = labels
        self.batch_size = batch_size
        self.classes_ = list(labels.values())

    def get_params(self, deep=True):
        return {
            'model': self.model,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,

            'batch_size': self.batch_size,
            'hidden_units': self.hidden_units,
            'dropout_rate': self.dropout_rate,
            'activation_function': self.activation_function,
            'loss_function': self.criterion,
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'epochs': self.epochs,
            'regular': self.regular,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'labels': self.labels
        }

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X_train, y_train):
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
        X_train = torch.from_numpy(X_train).float()
        if np.issubdtype(y_train.dtype, np.str_):
            labels_dict = {value: key for key, value in self.labels.items()}
            y_train = np.array([labels_dict[m] for m in y_train if m in labels_dict])
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
        draw_results_graphs(losses, self.epochs, self.learning_rate, self.hidden_units,
                            self.weight_decay, self.dropout_rate, self.regular)

    def predict(self, X):
        predictions = []
        mark_pred = []
        X = torch.from_numpy(X).float()

        self.model.eval()
        with torch.no_grad():
            pred_batch = self.model(X) # вероятность
            predictions.extend([np.hstack((pred.numpy(), 1 - pred.numpy())) for pred in pred_batch])
            mark_pred.extend([pred.numpy() for pred in pred_batch])
        mark = [item for m in mark_pred for item in m]
        mark = np.where(np.array(mark) > 0.5, 1, 0)
        labels = self.labels
        label_mark = []
        label_mark.extend([labels[m] for m in mark if m in labels])
        return label_mark

    def predict_proba(self, X):
        self.model.eval()
        predictions = []
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            pred_batch = self.model(X)  # вероятность
            predictions.extend([np.hstack((pred.numpy(), 1 - pred.numpy())) for pred in pred_batch])
        return np.array(predictions)

    def score(self, X, y):
        labels_dict = {value: key for key, value in self.labels.items()}
        y = np.array([labels_dict[m] for m in y if m in labels_dict])
        y = torch.from_numpy(y).float()
        y_pred = self.predict(X)
        mark = []
        mark.extend([labels_dict[m] for m in y_pred if m in labels_dict])
        return accuracy_score(y, mark)

class PyTorchTuneClassifier(BaseEstimator):
    def __init__(self, model, input_dim, output_dim, hidden_units,
                            dropout_rate, activation_function,
                            loss_function, optimizer, learning_rate, weight_decay,
                            epochs, regular, early_stopping, patience, labels, batch_size=20):
        self.model = model(input_dim, output_dim, hidden_units,
                           dropout_rate, activation_function)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.activation_function = activation_function
        self.criterion = loss_function
        self.weight_decay = weight_decay
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regular = regular
        self.early_stopping = early_stopping
        self.patience = patience
        self.labels = labels
        self.batch_size = batch_size

    def classes_(self):
        return self.labels.values()

    def get_params(self, deep=True):
        return {
            'model': self.model,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,

            'batch_size': self.batch_size,
            'hidden_units': self.hidden_units,
            'dropout_rate': self.dropout_rate,
            'activation_function': self.activation_function,
            'loss_function': self.criterion,
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'epochs': self.epochs,
            'regular': self.regular,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'labels': self.labels
        }

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X_train, y_train):
        X_train = torch.from_numpy(X_train).float()
        if np.issubdtype(y_train.dtype, np.str_):
            labels_dict = {value: key for key, value in self.labels.items()}
            y_train = np.array([labels_dict[m] for m in y_train if m in labels_dict])
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
        predictions = []
        mark_pred = []
        X = torch.from_numpy(X).float()

        self.model.eval()
        with torch.no_grad():
            pred_batch = self.model(X) # вероятность
            predictions.extend([np.hstack((pred.numpy(), 1 - pred.numpy())) for pred in pred_batch])
            mark_pred.extend([pred.numpy() for pred in pred_batch])
        mark = [item for m in mark_pred for item in m]
        mark = np.where(np.array(mark) > 0.5, 1, 0)
        labels = self.labels
        label_mark = []
        label_mark.extend([labels[m] for m in mark if m in labels])
        return label_mark

    def predict_proba(self, X):
        predictions = []
        X = torch.from_numpy(X).float()
        with torch.no_grad():
            pred_batch = self.model(X)  # вероятность
            predictions.extend([np.hstack((pred.numpy(), 1 - pred.numpy())) for pred in pred_batch])
        return np.array(predictions)

    def score(self, X, y):
        labels_dict = {value: key for key, value in self.labels.items()}
        y = np.array([labels_dict[m] for m in y if m in labels_dict])
        y = torch.from_numpy(y).float()
        y_pred = self.predict(X)
        mark = []
        mark.extend([labels_dict[m] for m in y_pred if m in labels_dict])
        return accuracy_score(y, mark)


def draw_roc_curve(y_val, y_mark):
    fpr, tpr, thresholds = roc_curve(y_val, y_mark)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    return roc_auc

def classify_based_on_roc(y_val, y_mark, threshold_strategy="accuracy"):
    fpr, tpr, thresholds = roc_curve(y_val, y_mark)
    if threshold_strategy == "accuracy":
        accuracy = tpr + (1 - fpr)
        opt_idx = np.argmax(accuracy)
    elif threshold_strategy == "sensitivity":
        opt_idx = np.argmax(tpr)
    elif threshold_strategy == "specificity":
        tnr = 1 - fpr
        opt_idx = np.argmax(tnr)

    opt_threshold = thresholds[opt_idx]
    mark = np.where(np.array(y_mark) > opt_threshold, 1, 0)
    return opt_threshold, mark



def set_marks():
    list_cat = [i.title for i in session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).all()]
    labels = {}
    labels[list_cat[0]] = 1
    labels[list_cat[1]] = 0
    if len(list_cat) > 2:
        for index, i in enumerate(list_cat[2:]):
            labels[i] = index
    return labels

def str_to_interval(string):
    parts = string.split("-")
    result = [float(part.replace(",", ".")) for part in parts]
    return result

def add_features(ui_tch, text_model):
    pipe_steps = []
    text_scaler = ''
    if ui_tch.checkBox_stdscaler.isChecked():
        std_scaler = StandardScaler()
        pipe_steps.append(('std_scaler', std_scaler))
        text_scaler += '\nStandardScaler'
    if ui_tch.checkBox_robscaler.isChecked():
        robust_scaler = RobustScaler()
        pipe_steps.append(('robust_scaler', robust_scaler))
        text_scaler += '\nRobustScaler'
    if ui_tch.checkBox_mnmxscaler.isChecked():
        minmax_scaler = MinMaxScaler()
        pipe_steps.append(('minmax_scaler', minmax_scaler))
        text_scaler += '\nMinMaxScaler'
    if ui_tch.checkBox_mxabsscaler.isChecked():
        maxabs_scaler = MaxAbsScaler()
        pipe_steps.append(('maxabs_scaler', maxabs_scaler))
        text_scaler += '\nMaxAbsScaler'

    over_sampling, text_over_sample = 'none', ''
    if ui_tch.checkBox_smote.isChecked():
        over_sampling, text_over_sample = 'smote', '\nSMOTE'
    if ui_tch.checkBox_adasyn.isChecked():
        over_sampling, text_over_sample = 'adasyn', '\nADASYN'

    if ui_tch.checkBox_pca.isChecked():
        n_comp = ui_tch.spinBox_pca.value()
        pca = PCA(n_components=n_comp, random_state=0, svd_solver='auto')
        pipe_steps.append(('pca', pca))
    text_pca = f'\nPCA: n_components={n_comp}' if ui_tch.checkBox_pca.isChecked() else ''

    text_model += text_scaler
    text_model += text_over_sample
    text_model += text_pca

    return pipe_steps, text_model

def final_model_build(ui_tch, input_dim, output_dim, labels_dict):
    epochs = ui_tch.spinBox_epochs.value()
    learning_rate = ui_tch.doubleSpinBox_choose_lr.value()
    hidden_units = list(map(int, ui_tch.lineEdit_choose_layers.text().split()))
    dropout_rate = ui_tch.doubleSpinBox_choose_dropout.value()
    weight_decay = ui_tch.doubleSpinBox_choose_decay.value()
    regular = ui_tch.doubleSpinBox_choose_reagular.value()

    if ui_tch.comboBox_activation_func.currentText() == 'ReLU':
        activation_function = nn.ReLU()
    elif ui_tch.comboBox_activation_func.currentText() == 'Sigmoid':
        activation_function = nn.Sigmoid()
    elif ui_tch.comboBox_activation_func.currentText() == 'Tanh':
        activation_function = nn.Tanh()

    if ui_tch.comboBox_optimizer.currentText() == 'Adam':
        optimizer = torch.optim.Adam
    elif ui_tch.comboBox_optimizer.currentText() == 'SGD':
        optimizer = torch.optim.SGD

    if ui_tch.comboBox_loss.currentText() == 'CrossEntropy':
        loss_function = nn.CrossEntropyLoss()
    elif ui_tch.comboBox_loss.currentText() == 'BCEWithLogitsLoss':
        loss_function = nn.BCEWithLogitsLoss()
    elif ui_tch.comboBox_loss.currentText() == 'BCELoss':
        loss_function = nn.BCELoss()

    early_stopping = False
    patience = 0
    if ui_tch.checkBox_early_stop.isChecked():
        early_stopping = True

    model_name = 'torch_NN_ENSEMBLE_META_MODEL'
    text_model = model_name + ''

    final_model = PyTorchClassifier(Model, input_dim, output_dim, hidden_units,
                                          dropout_rate, activation_function,
                                          loss_function, optimizer, learning_rate, weight_decay,
                                          epochs, regular, early_stopping, patience, labels_dict, batch_size=20)
    return final_model, text_model



def torch_classifier_train():
    TorchClassifier = QtWidgets.QDialog()
    ui_tch = Ui_TorchClassifierForm()
    ui_tch.setupUi(TorchClassifier)
    TorchClassifier.show()
    TorchClassifier.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    data, list_param = build_table_train(True, 'mlp')
    list_param_mlp = data.columns.tolist()[2:]
    labels = set_marks()
    labels_dict = {value: key for key, value in labels.items()}
    data['mark'] = data['mark'].replace(labels)
    data = data.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, 2:], data['mark'], test_size=0.2, random_state=42)
    y_train = y_train.values
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    X_val = X_test.values
    y_val = torch.from_numpy(y_test.values).float()

    model_name = 'torch_NN_cls'

    def torch_classifier_lineup():
        input_dim = X_train.shape[1]
        output_dim = 1

        epochs = ui_tch.spinBox_epochs.value()
        learning_rate = ui_tch.doubleSpinBox_choose_lr.value()
        hidden_units = list(map(int, ui_tch.lineEdit_choose_layers.text().split()))
        dropout_rate = ui_tch.doubleSpinBox_choose_dropout.value()
        weight_decay = ui_tch.doubleSpinBox_choose_decay.value()
        regular = ui_tch.doubleSpinBox_choose_reagular.value()

        if ui_tch.comboBox_activation_func.currentText() == 'ReLU':
            activation_function = nn.ReLU()
        elif ui_tch.comboBox_activation_func.currentText() == 'Sigmoid':
            activation_function = nn.Sigmoid()
        elif ui_tch.comboBox_activation_func.currentText() == 'Tanh':
            activation_function = nn.Tanh()

        if ui_tch.comboBox_optimizer.currentText() == 'Adam':
            optimizer = torch.optim.Adam
        elif ui_tch.comboBox_optimizer.currentText() == 'SGD':
            optimizer = torch.optim.SGD

        if ui_tch.comboBox_loss.currentText() == 'CrossEntropy':
            loss_function = nn.CrossEntropyLoss()
        elif ui_tch.comboBox_loss.currentText() == 'BCEWithLogitsLoss':
            loss_function = nn.BCEWithLogitsLoss()
        elif ui_tch.comboBox_loss.currentText() == 'BCELoss':
            loss_function = nn.BCELoss()

        early_stopping = False
        patience = 0
        if ui_tch.checkBox_early_stop.isChecked():
            early_stopping = True
            patience = ui_tch.spinBox_stop_patience.value()

        text_model = model_name + ''
        pipe_steps, text_model = add_features(ui_tch, text_model)
        feature_union = FeatureUnion(pipe_steps)
        X_train_transformed = feature_union.fit_transform(X[:100])
        input_dim = X_train_transformed.shape[1]

        pipeline = Pipeline([
            ('features', feature_union),
            ('classifier', PyTorchClassifier(Model, input_dim, output_dim, hidden_units,
                                             dropout_rate, activation_function,
                                             loss_function, optimizer, learning_rate, weight_decay,
                                             epochs, regular, early_stopping, patience, labels_dict, batch_size=20))
        ])

        except_mlp = session.query(ExceptionMLP).filter_by(analysis_id=get_MLP_id()).first()
        new_lineup = LineupTrain(
            type_ml='cls',
            analysis_id=get_MLP_id(),
            list_param=json.dumps(list_param_mlp),
            list_param_short=json.dumps(list_param),
            except_signal=except_mlp.except_signal,
            except_crl=except_mlp.except_crl,
            text_model=text_model,
            model_name=model_name,
            over_sampling='none',
            pipe=pickle.dumps(pipeline),
            random_seed=ui.spinBox_seed.value(),
            cvw=False
        )
        session.add(new_lineup)
        session.commit()

        set_info(f'Модель {model_name} добавлена в очередь\n{text_model}', 'green')

    def train():
        input_dim = X_train.shape[1]
        output_dim = 1

        epochs = ui_tch.spinBox_epochs.value()
        learning_rate = ui_tch.doubleSpinBox_choose_lr.value()
        hidden_units = list(map(int, ui_tch.lineEdit_choose_layers.text().split()))
        dropout_rate = ui_tch.doubleSpinBox_choose_dropout.value()
        weight_decay = ui_tch.doubleSpinBox_choose_decay.value()
        regular = ui_tch.doubleSpinBox_choose_reagular.value()

        if ui_tch.comboBox_activation_func.currentText() == 'ReLU':
            activation_function = nn.ReLU()
        elif ui_tch.comboBox_activation_func.currentText() == 'Sigmoid':
            activation_function = nn.Sigmoid()
        elif ui_tch.comboBox_activation_func.currentText() == 'Tanh':
            activation_function = nn.Tanh()

        if ui_tch.comboBox_optimizer.currentText() == 'Adam':
            optimizer = torch.optim.Adam
        elif ui_tch.comboBox_optimizer.currentText() == 'SGD':
            optimizer = torch.optim.SGD

        if ui_tch.comboBox_loss.currentText() == 'CrossEntropy':
            loss_function = nn.CrossEntropyLoss()
        elif ui_tch.comboBox_loss.currentText() == 'BCEWithLogitsLoss':
            loss_function = nn.BCEWithLogitsLoss()
        elif ui_tch.comboBox_loss.currentText() == 'BCELoss':
            loss_function = nn.BCELoss()

        patience = 0
        early_stopping = False
        if ui_tch.checkBox_early_stop.isChecked():
            early_stopping = True
            patience = ui_tch.spinBox_stop_patience.value()

        X = X_train.astype(np.float64)
        y = y_train.astype(np.float64)
        if ui_tch.checkBox_smote.isChecked():
            smote = SMOTE(random_state=0)
            X, y = smote.fit_resample(X, y)
        if ui_tch.checkBox_adasyn.isChecked():
            adasyn = ADASYN(random_state=0)
            X, y = adasyn.fit_resample(X, y)

        text_model = model_name + ''
        pipe_steps, text_model = add_features(ui_tch, text_model)
        feature_union = FeatureUnion(pipe_steps)
        X_train_transformed = feature_union.fit_transform(X[:100])
        input_dim = X_train_transformed.shape[1]
        pipeline = Pipeline([
            ('features', feature_union),
            ('classifier', PyTorchClassifier(Model, input_dim, output_dim, hidden_units,
                                             dropout_rate, activation_function,
                                             loss_function, optimizer, learning_rate, weight_decay,
                                             epochs, regular, early_stopping, patience, labels_dict, batch_size=20))
        ])

        start_time = datetime.datetime.now()
        pipeline.fit(X, y)

        y_mark = pipeline.predict(X_val)
        mark = []
        mark.extend([labels[m] for m in y_mark if m in labels])
        y_res = pipeline.predict_proba(X_val)
        y_prob = [i[0] for i in y_res]
        accuracy = accuracy_score(y_val, mark)
        precision = precision_score(y_val, mark)
        recall = recall_score(y_val, mark)
        f1 = f1_score(y_val, mark)
        print('Accuracy: ', accuracy)
        roc_auc = draw_roc_curve(y_val, y_prob)
        end_time = datetime.datetime.now() - start_time
        print(end_time)

        if ui_tch.checkBox_save_model.isChecked():
            text_model_final = '*** TORCH NN *** \n' + text_model + 'test_accuray: ' + str(round(accuracy, 3)) \
                               + '\nroc_auc: ' + str(round(roc_auc, 3)) + '\nprecision: ' + str(round(precision, 3)) +\
                               '' + '\nrecall: ' + str(round(recall, 3)) + '\nf1: ' + str(round(f1, 3)) + '\nвремя обучения: ' \
                         + str(end_time) + '\nlearning_rate: ' + str(learning_rate) + '\nhidden units: ' + str(
                hidden_units) \
                         + '\nweight decay: ' + str(weight_decay) + '\ndropout rate: ' + str(dropout_rate) + \
                         '\nregularization: ' + str(regular)
            torch_save_classifier(pipeline, accuracy, list_param, text_model_final)
            print('Model saved')

    def tune_params():
        start_time = datetime.datetime.now()
        input_dim = X_train.shape[1]
        output_dim = 1

        epochs = ui_tch.spinBox_epochs.value()
        learning_rate = str_to_interval(ui_tch.lineEdit_tune_lr.text())
        dropout_rate = str_to_interval(ui_tch.lineEdit_tune_dropout.text())
        weight_decay = str_to_interval(ui_tch.lineEdit_tune_decay.text())
        regularization = ui_tch.doubleSpinBox_choose_reagular.value()
        hidden_units = str_to_interval(ui_tch.lineEdit_tune_layers.text())
        layers_num = ui_tch.spinBox_layers_num.value()
        patience = ui_tch.spinBox_stop_patience.value()

        if ui_tch.comboBox_activation_func.currentText() == 'ReLU':
            activation_function = nn.ReLU()
        elif ui_tch.comboBox_activation_func.currentText() == 'Sigmoid':
            activation_function = nn.Sigmoid()
        elif ui_tch.comboBox_activation_func.currentText() == 'Tanh':
            activation_function = nn.Tanh()

        if ui_tch.comboBox_optimizer.currentText() == 'Adam':
            optimizer = torch.optim.Adam
        elif ui_tch.comboBox_optimizer.currentText() == 'SGD':
            optimizer = torch.optim.SGD

        if ui_tch.comboBox_loss.currentText() == 'CrossEntropy':
            loss_function = nn.CrossEntropyLoss()
        elif ui_tch.comboBox_loss.currentText() == 'BCEWithLogitsLoss':
            loss_function = nn.BCEWithLogitsLoss()
        elif ui_tch.comboBox_loss.currentText() == 'BCELoss':
            loss_function = nn.BCELoss()

        early_stopping = False
        if ui_tch.checkBox_early_stop.isChecked():
            early_stopping = True


        def objective(trial):
            print("Trial number:", trial.number)
            op_learning_rate = trial.suggest_float('learning_rate', learning_rate[0], learning_rate[1], log=True)
            op_num_hidden_units = [trial.suggest_int(f'num_hidden_units_layer{i}',
                                                     hidden_units[0], hidden_units[1]) for i in range(1, layers_num)]
            op_dropout_rate = trial.suggest_float('dropout_rate', dropout_rate[0], dropout_rate[1], log=True)
            op_weight_decay = trial.suggest_float('weight_decay', weight_decay[0], weight_decay[1], log=True)
            op_regularization = trial.suggest_float('regularization', regularization, regularization + 0.1, log=True)

            X = X_train.astype(np.float64)
            y = y_train.astype(np.float64)
            if ui_tch.checkBox_smote.isChecked():
                smote = SMOTE(random_state=0)
                X, y = smote.fit_resample(X, y)
            if ui_tch.checkBox_adasyn.isChecked():
                adasyn = ADASYN(random_state=0)
                X, y = adasyn.fit_resample(X, y)
            text_model = model_name + ''
            pipe_steps, text_model = add_features(ui_tch, text_model)
            feature_union = FeatureUnion(pipe_steps)
            X_train_transformed = feature_union.fit_transform(X[:100])
            input_dim = X_train_transformed.shape[1]

            pipeline = Pipeline([
                ('features', feature_union),
                ('classifier', PyTorchClassifier(Model, input_dim, output_dim, op_num_hidden_units,
                                                 op_dropout_rate, activation_function,
                                                 loss_function, optimizer, op_learning_rate, op_weight_decay,
                                                 epochs, op_regularization, early_stopping, patience,
                                                    labels_dict, batch_size=20))
            ])

            pipeline.fit(X, y)

            y_mark = pipeline.predict(X_val)
            mark = []
            mark.extend([labels[m] for m in y_mark if m in labels])
            accuracy = accuracy_score(y_val, mark)
            print('Accuracy: ', accuracy)
            return accuracy

        num_trials = ui_tch.spinBox_trials.value()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize',
                                    pruner=optuna.pruners.MedianPruner(
                                        n_startup_trials=2, n_warmup_steps=20, interval_steps=1
                                    ), sampler=optuna.samplers.RandomSampler(seed=10))
        study.optimize(objective, n_trials=num_trials)

        print("Number of finished trials:", len(study.trials))
        print("Best trial:")
        trial = study.best_trial

        print("Value: ", trial.value)
        print("Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        best_learning_rate = trial.params['learning_rate']
        best_num_hidden_units = [trial.params[f'num_hidden_units_layer{i}'] for i in range(1, layers_num)]
        best_dropout_rate = trial.params['dropout_rate']
        best_weight_decay = trial.params['weight_decay']
        best_regularization = trial.params['regularization']

        epochs = ui_tch.spinBox_epochs.value()
        if ui_tch.comboBox_activation_func.currentText() == 'ReLU':
            activation_function = nn.ReLU()
        elif ui_tch.comboBox_activation_func.currentText() == 'Sigmoid':
            activation_function = nn.Sigmoid()
        elif ui_tch.comboBox_activation_func.currentText() == 'Tanh':
            activation_function = nn.Tanh()

        if ui_tch.comboBox_optimizer.currentText() == 'Adam':
            optimizer = torch.optim.Adam
        elif ui_tch.comboBox_optimizer.currentText() == 'SGD':
            optimizer = torch.optim.SGD

        if ui_tch.comboBox_loss.currentText() == 'BCEWithLogitsLoss':
            loss_function = torch.nn.BCEWithLogitsLoss()
        elif ui_tch.comboBox_loss.currentText() == 'CrossEntropy':
            loss_function = torch.nn.CrossEntropyLoss()
        elif ui_tch.comboBox_loss.currentText() == 'BCELoss':
            loss_function = torch.nn.BCELoss()

        early_stopping = False
        if ui_tch.checkBox_early_stop.isChecked():
            early_stopping = True

        X = X_train.astype(np.float64)
        y = y_train.astype(np.float64)
        if ui_tch.checkBox_smote.isChecked():
            smote = SMOTE(random_state=0)
            X, y = smote.fit_resample(X, y)
        if ui_tch.checkBox_adasyn.isChecked():
            adasyn = ADASYN(random_state=0)
            X, y = adasyn.fit_resample(X, y)
        text_model = model_name + ''
        pipe_steps, text_model = add_features(ui_tch, text_model)
        feature_union = FeatureUnion(pipe_steps)
        X_train_transformed = feature_union.fit_transform(X[:100])
        input_dim = X_train_transformed.shape[1]
        pipeline = Pipeline([
            ('features', feature_union),
            ('classifier', PyTorchClassifier(Model, input_dim, output_dim, best_num_hidden_units,
                                             best_dropout_rate, activation_function,
                                             loss_function, optimizer, best_learning_rate, best_weight_decay,
                                             epochs, best_regularization, early_stopping, patience,
                                             labels_dict, batch_size=20))
        ])

        pipeline.fit(X, y)
        y_mark = pipeline.predict(X_val)
        y_res = pipeline.predict_proba(X_val)
        y_prob = [i[0] for i in y_res]
        mark = []
        mark.extend([labels[m] for m in y_mark if m in labels])
        accuracy = accuracy_score(y_val, mark)
        precision = precision_score(y_val, mark)
        recall = recall_score(y_val, mark)
        f1 = f1_score(y_val, mark)
        roc_auc = draw_roc_curve(y_val, y_prob)
        print('Best Accuracy: ', accuracy)
        end_time = datetime.datetime.now() - start_time
        print(end_time)

        if ui_tch.checkBox_save_model.isChecked():
            text_model_final = '*** TORCH NN *** \n' + text_model + 'test_accuray: ' + str(round(accuracy, 3)) +\
                               '\nprecision ' +  str(round(precision, 3)) + '\nrecall ' + str(round(recall, 3)) \
                               + '\nf1 ' + str(round(f1, 3)) +  '\nroc_auc ' + str(round(roc_auc, 3)) + '\nвремя обучения: ' + str(end_time) + '\nlearning_rate: '\
                               + str(best_learning_rate) + '\nhidden units: ' + str(best_num_hidden_units) \
                         + '\nweight decay: ' + str(best_weight_decay) + '\ndropout rate: ' + str(best_dropout_rate) + \
                         '\nregularization: ' + str(best_regularization)
            torch_save_classifier(pipeline, accuracy, list_param, text_model_final)
            print('Model saved')

    def train_stack_vote():
        input_dim = X_train.shape[1]
        output_dim = 1
        # estimators -- base models
        # final_model -- meta_model
        # list_model -- names of models array

        # ESTIMATORS
        activation_func = [nn.ReLU(), nn.Sigmoid(), nn.Tanh()]
        optimizer = [torch.optim.Adam, torch.optim.SGD]
        loss_func = [torch.nn.BCEWithLogitsLoss(), torch.nn.CrossEntropyLoss(), torch.nn.BCELoss()]
        epochs = ui_tch.spinBox_epochs.value()
        learning_rate = str_to_interval(ui_tch.lineEdit_tune_lr.text())
        dropout_rate = str_to_interval(ui_tch.lineEdit_tune_dropout.text())
        weight_decay = str_to_interval(ui_tch.lineEdit_tune_decay.text())
        regularization = ui_tch.doubleSpinBox_choose_reagular.value()
        hidden_units = str_to_interval(ui_tch.lineEdit_tune_layers.text())
        layers_num = ui_tch.spinBox_layers_num.value()
        patience = ui_tch.spinBox_stop_patience.value()
        early_stopping = False
        if ui_tch.checkBox_early_stop.isChecked():
            early_stopping = True

        num_models = ui_tch.spinBox_models_num.value()
        estimators, list_model = [], []
        for i in range(num_models):
            model = PyTorchClassifier(Model, input_dim, output_dim,
                                      hidden_units=np.random.randint(hidden_units[0], hidden_units[1], layers_num),
                                      dropout_rate=random.uniform(dropout_rate[0], dropout_rate[1]),
                                      activation_function=activation_func[random.randint(0, len(activation_func) - 1)],
                                      loss_function=loss_func[random.randint(0, len(loss_func) - 1)],
                                      optimizer=optimizer[random.randint(0, len(optimizer) - 1)],
                                      learning_rate=random.uniform(learning_rate[0], learning_rate[1]),
                                      weight_decay=random.uniform(weight_decay[0], weight_decay[1]),
                                      epochs=random.randint(49, epochs),
                                      regular=random.uniform(0.000001, regularization),
                                      early_stopping=early_stopping,
                                      patience=patience,
                                      labels=labels_dict,
                                      batch_size=20)
            X = X_train.values.astype(np.float64)
            y = y_train.astype(np.float64)
            model.fit(X, y)
            estimators.append((f'model_{i}', model))
            print(f'model_{i}')

        final_model, final_text_model = final_model_build(ui_tch, input_dim, output_dim, labels_dict)
        if ui_tch.radioButton_voting.isChecked():
            # hard_voting = 'hard' if ui_cls.checkBox_voting_hard.isChecked() else 'soft'
            model_class = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
            text_model = f'**Voting**: -soft-\n({num_models} models)\n'
            model_name = 'VOT'
        elif ui_tch.radioButton_stacking.isChecked():
            model_class = StackingClassifier(estimators=estimators, final_estimator=final_model, n_jobs=-1)
            text_model = f'**Stacking**:\nFinal estimator: {final_text_model}\n + {num_models} Base models\n'
            model_name = 'STACK'

        X = X_train.astype(np.float64)
        y = y_train.astype(np.float64)
        if ui_tch.checkBox_smote.isChecked():
            smote = SMOTE(random_state=0)
            X, y = smote.fit_resample(X, y)
        if ui_tch.checkBox_adasyn.isChecked():
            adasyn = ADASYN(random_state=0)
            X, y = adasyn.fit_resample(X, y)
        text_model = model_name + text_model
        pipe_steps, text_model = add_features(ui_tch, text_model)
        feature_union = FeatureUnion(pipe_steps)
        X_train_transformed = feature_union.fit_transform(X[:100])
        input_dim = X_train_transformed.shape[1]

        pipeline = Pipeline([
            ('features', feature_union),
            ('classifier', model_class)])

        start_time = datetime.datetime.now()
        print('Start time: ', start_time)

        pipeline.fit(X, y)
        y_mark = pipeline.predict(X_val)
        mark = []
        mark.extend([labels[m] for m in y_mark if m in labels])
        y_res = pipeline.predict_proba(X_val)
        y_prob = [i[0] for i in y_res]
        accuracy = accuracy_score(y_val, mark)
        print('Accuracy: ', accuracy)
        draw_roc_curve(y_val, y_prob)

        end_time = datetime.datetime.now() - start_time
        print(end_time)

        if ui_tch.checkBox_save_model.isChecked():
            text_model_final = '*** TORCH NN *** \n' + text_model + 'test_accuray: ' + str(
                round(accuracy, 3)) + '\nвремя обучения: ' \
                               + str(end_time) + '\n'
            torch_save_classifier(pipeline, accuracy, list_param, text_model_final)
            print('Model saved')

    def class_bagging():
        input_dim = X_train.shape[1]
        output_dim = 1

        epochs = ui_tch.spinBox_epochs.value()
        learning_rate = ui_tch.doubleSpinBox_choose_lr.value()
        hidden_units = list(map(int, ui_tch.lineEdit_choose_layers.text().split()))
        dropout_rate = ui_tch.doubleSpinBox_choose_dropout.value()
        weight_decay = ui_tch.doubleSpinBox_choose_decay.value()
        regular = ui_tch.doubleSpinBox_choose_reagular.value()
        if ui_tch.comboBox_activation_func.currentText() == 'ReLU':
            activation_function = nn.ReLU()
        elif ui_tch.comboBox_activation_func.currentText() == 'Sigmoid':
            activation_function = nn.Sigmoid()
        elif ui_tch.comboBox_activation_func.currentText() == 'Tanh':
            activation_function = nn.Tanh()

        if ui_tch.comboBox_optimizer.currentText() == 'Adam':
            optimizer = torch.optim.Adam
        elif ui_tch.comboBox_optimizer.currentText() == 'SGD':
            optimizer = torch.optim.SGD

        if ui_tch.comboBox_loss.currentText() == 'CrossEntropy':
            loss_function = nn.CrossEntropyLoss()
        elif ui_tch.comboBox_loss.currentText() == 'BCEWithLogitsLoss':
            loss_function = nn.BCEWithLogitsLoss()
        elif ui_tch.comboBox_loss.currentText() == 'BCELoss':
            loss_function = nn.BCELoss()

        patience = 0
        early_stopping = False
        if ui_tch.checkBox_early_stop.isChecked():
            early_stopping = True
            patience = ui_tch.spinBox_stop_patience.value()
        num_estimators = ui_tch.spinBox_bagging.value()

        base_model = PyTorchClassifier(Model, input_dim, output_dim, hidden_units,
                          dropout_rate, activation_function,
                          loss_function, optimizer, learning_rate, weight_decay,
                          epochs, regular, early_stopping, patience, labels_dict, batch_size=20)
        X = np.array(X_train)
        y = np.array(y_train)
        base_model.fit(X, y)
        meta_classifier = BaggingClassifier(base_estimator=base_model, n_estimators=num_estimators, n_jobs=-1)
        print('type X meta', X.shape)
        X = X.reshape(-1, input_dim)
        print('type X reshape', X.shape)
        meta_classifier.fit(X, y)
        print("score ", meta_classifier.score(X_val, y_val))
        pass


    def choose():
        if ui_tch.checkBox_choose_param.isChecked():
            train()

        elif ui_tch.checkBox_tune_param.isChecked():
            tune_params()

        elif ui_tch.checkBox_stack_vote.isChecked():
            train_stack_vote()

        elif ui_tch.checkBox_bagging.isChecked():
            class_bagging()

    ui_tch.pushButton_lineup.clicked.connect(torch_classifier_lineup)
    ui_tch.pushButton_train.clicked.connect(choose)
    TorchClassifier.exec()

