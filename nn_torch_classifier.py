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


# def hidden_block(input_dim, output_dim, dropout_rate, activation_func):
#     return torch.nn.Sequential(
#         torch.nn.Linear(input_dim, output_dim),
#         activation_func,
#         torch.nn.BatchNorm1d(output_dim),
#         torch.nn.Dropout(dropout_rate),
#     )
#
#
# class Model(torch.nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_units, dropout_rate, activation_function):
#         super(Model, self).__init__()
#         self.input_layer = torch.nn.Linear(input_dim, hidden_units[0])
#         self.batch_norm1 = nn.BatchNorm1d(hidden_units[0])
#         self.hidden_layers = torch.nn.ModuleList(
#             [hidden_block(hidden_units[i], hidden_units[i + 1], dropout_rate, activation_function) for i in range(len(hidden_units) - 1)]
#         )
#         self.output_layer = torch.nn.Linear(hidden_units[-1], output_dim)
#         self.sigmoid = torch.nn.Sigmoid()
#
#     def forward(self, x):
#         out = torch.relu(self.input_layer(x))
#         out = self.batch_norm1(out)
#         for layer in self.hidden_layers:
#             out = self.activation(layer(out))
#             out = self.dropout(out)
#         out = self.output_layer(out)
#         out = self.sigmoid(out)
#         return out


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


def str_to_interval(string):
    parts = string.split("-")
    result = [float(part.replace(",", ".")) for part in parts]
    return result

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


class PyTorchClassifier:
    def __init__(self, model, input_dim, output_dim, hidden_units,
                            dropout_rate, activation_function,
                            loss_function, optimizer, learning_rate, weight_decay,
                            epochs, regular, batch_size=20):
        self.model = model(input_dim, output_dim, hidden_units,
                           dropout_rate, activation_function)
        self.criterion = loss_function
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.epochs = epochs
        self.regular = regular
        self.batch_size = batch_size

    def fit(self, X_train, y_train):
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        losses = []

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

            losses.append(running_loss / (X_train.shape[0] / self.batch_size))
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {running_loss / (X_train.shape[0] / self.batch_size)}')
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
        return mark

    def predict_proba(self, X):
        predictions = []
        mark_pred = []
        X = torch.from_numpy(X).float()
        with torch.no_grad():
            pred_batch = self.model(X)  # вероятность
            predictions.extend([np.hstack((pred.numpy(), 1 - pred.numpy())) for pred in pred_batch])
            mark_pred.extend([pred.numpy() for pred in pred_batch])
        mark = [item for m in mark_pred for item in m]
        return predictions


    def metrics(self, X_val, y_val, opt_threshold=0.5):
        all_targets = []
        all_predictions = []
        predictions = []
        X = torch.from_numpy(X_val).float()
        self.model.eval()
        with torch.no_grad():
            val_dl = DataLoader(TensorDataset(X, y_val), batch_size=20, shuffle=False)
            for xb, yb in val_dl:
                pred_batch = self.model(xb)
                all_targets.extend([y.numpy() for y in yb])
                all_predictions.extend([pred.numpy() for pred in pred_batch])
                predictions.extend([np.hstack((pred.numpy(), 1 - pred.numpy())) for pred in pred_batch])
        all_predictions = [1 if p >= opt_threshold else 0 for p in all_predictions]
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions)
        recall = recall_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions)
        return accuracy, precision, recall, f1

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


def nn_torch(ui_tch, data, list_param, labels, labels_mark):
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, 2:], data['mark'], test_size=0.2, random_state=42)
    y_train = y_train.values
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    X_val = X_test.values
    y_val = torch.from_numpy(y_test.values).float()

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

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('scaler', StandardScaler())
        ])),
        ('classifier', PyTorchClassifier(Model, input_dim, output_dim, hidden_units,
                            dropout_rate, activation_function,
                            loss_function, optimizer, learning_rate, weight_decay,
                            epochs, regular, batch_size=20))
    ])

    start_time = datetime.datetime.now()
    pipeline.fit(X, y)
    # y_mark = pipeline.named_steps['classifier'].predict_graphs(X_val, y_val)
    y_mark = pipeline.predict(X_val)
    mark = np.where(np.array(y_mark) > 0.5, 1, 0)
    accuracy = accuracy_score(y_val, mark)
    end_time = datetime.datetime.now() - start_time
    print('Accuracy: ', accuracy)
    print(end_time)
    draw_roc_curve(y_val, y_mark)

    if ui_tch.checkBox_save_model.isChecked():
        text_model = '*** TORCH NN *** \n' + 'test_accuray: ' + str(round(accuracy, 3)) + '\nвремя обучения: ' \
                     + str(end_time) + '\nlearning_rate: ' + str(learning_rate) + '\nhidden units: ' + str(hidden_units) \
                     + '\nweight decay: ' + str(weight_decay) + '\ndropout rate: ' + str(dropout_rate) + \
                     '\nregularization: ' + str(regular) + '\n'
        torch_save_classifier(pipeline, accuracy, list_param, text_model)
        print('Model saved')

def torch_classifier_train():
    TorchClassifier = QtWidgets.QDialog()
    ui_tch = Ui_TorchClassifierForm()
    ui_tch.setupUi(TorchClassifier)
    TorchClassifier.show()
    TorchClassifier.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    data, list_param = build_table_train(True, 'mlp')
    labels = {'empty': 0, 'bitum': 1, 'пусто': 0, 'нефть': 1, 'cold': 0, 'hot': 1}
    labels_dict = {value: key for key, value in labels.items()}
    ## todo Переделать в функцию
    # data['mark'] = data['mark'].replace({'empty': 0, 'bitum': 1})
    # data['mark'] = data['mark'].replace({'пусто': 0, 'нефть': 1})
    # data['mark'] = data['mark'].replace({'cold': 0, 'hot': 1})
    data['mark'] = data['mark'].replace(labels)
    data = data.fillna(0)

    def train():
        # if ui_tch.checkBox_choose_param.isChecked():
        #     nn_choose_params(ui_tch, data, list_param)
        #
        # if ui_tch.checkBox_tune_param.isChecked():
        nn_torch(ui_tch, data, list_param, labels_dict, labels)
            # nn_tune_params(ui_tch, data, list_param)
    def cv():
        nn_cross_val(ui_tch, data)

    ui_tch.pushButton_train.clicked.connect(train)
    ui_tch.pushButton_cross_val.clicked.connect(cv)
    TorchClassifier.exec()
