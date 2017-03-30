import numpy as np
import pandas as pd

TRAIN_DATA_PATH = 'D:/data/korean_chess/records.csv'


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def load_train(data_path, validation_size=10000):
    data = pd.read_csv(data_path)
    # data = data.sample(frac=1).reset_index(drop=True)
    input_feature_maps = data.iloc[:, :3 * 10 * 9].values
    input_feature_maps = input_feature_maps.astype(np.float16)
    input_feature_maps = input_feature_maps.reshape(-1, 3, 10, 9)

    labels = data.iloc[:, 3 * 10 * 9:].values
    labels = labels.astype(np.uint8)
    labels = labels.reshape(-1, 2, 90)
    # labels = labels.reshape(-1, 2, 10, 9)

    validation_input_feature_maps = input_feature_maps[:validation_size]
    validation_labels = labels[:validation_size]

    train_input_feature_maps = input_feature_maps[validation_size:]
    train_labels = labels[validation_size:]

    return train_input_feature_maps, train_labels, validation_input_feature_maps, validation_labels
