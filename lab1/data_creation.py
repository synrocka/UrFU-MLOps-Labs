import numpy as np
import os
import pandas as pd

data_length = 2000


def create_data(base_value, noise_value):
    """
    Генерация псевдослучайных данных
    :param base_value:  основная величина признака без шума
    :param noise_value: амплитуда шума
    :return: массив numpy размером data_length на 1
    """
    return np.array([int(base_value * x) +
                     int(np.random.choice([-1, 1]) * noise_value * np.random.rand())
                     for x in np.random.rand(data_length)])


def create_base_structure(test_path='./test', train_path='./train'):
    """
    Создание базовой структуры директорий (для тестовых и тренировочных данных)
    :param test_path: путь до тестовой директории
    :param train_path: путь до тренировочной директории
    """
    try:
        os.mkdir(test_path)
    except FileExistsError:
        print("Folder %s already exists" % test_path)

    try:
        os.mkdir(train_path)
    except FileExistsError:
        print("Folder %s already exists" % train_path)


def write_file(filename, np_data, is_target: bool, column_names, train_dir_path='./train', test_dir_path='./test'):
    """
    Запись данных в две директории (тестовую и тренировочную)
    :param filename: базовое наименование файла
    :param np_data: данные для записи
    :param is_target: логическое указание является ли файл таргетом или нет
    :param column_names: список наименований колонок
    :param train_dir_path: путь до тренировочной директории
    :param test_dir_path:  путь до тестовой директории
    :return: множество их двух строк, содержащие пути дос с=записанных файлов
    """
    if is_target:
        df_train = pd.DataFrame(np_data[:int(data_length * 0.8)])
        df_test = pd.DataFrame(np_data[int(data_length * 0.8):])
    else:
        df_train = pd.DataFrame(np_data[:int(data_length * 0.8), :])
        df_test = pd.DataFrame(np_data[int(data_length * 0.8):, :])

    df_train.columns = column_names
    df_test.columns = column_names
    train_path = os.path.join(train_dir_path, filename + "_train.csv")
    test_path = os.path.join(test_dir_path, filename + "_test.csv")

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    return train_path, test_path


create_base_structure()
print("--- Basic data structure created")

feature_1 = create_data(20, 4)
feature_2 = create_data(1000, 27)
feature_3 = create_data(10, 2)
feature_4 = create_data(90, 7)
print("--- Data generated")

feat_1_feat_2 = np.stack((feature_1, feature_2), axis=1)
feat_3_feat_4 = np.stack((feature_3, feature_4), axis=1)
target = create_data(5, 2)

files = [write_file("feat_1_feat_2", feat_1_feat_2, False, ['feat_1', 'feat_2']),
         write_file("feat_3_feat_4", feat_3_feat_4, False, ['feat_3', 'feat_4']),
         write_file("target", target, True, ['target'])]
print(f"--- Data written to files: {files}")
