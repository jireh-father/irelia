import json
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import shutil
import csv


class Dataset(object):
    DEF_TRAIN_FILE_NAME = "train_dataset.txt"
    DEF_TEST_FILE_NAME = "train_dataset.txt"

    def __init__(self, sess):
        self.sess = sess
        self.file = None
        self.csv_writer = None
        self.dataset = None

    def open(self, file_path, mode="w+"):
        self.file = open(file_path, mode=mode)
        self.csv_writer = csv.writer(self.file, delimiter=',')

    def close(self):
        if self.file is not None:
            self.file.close()

    def write(self, info, state_history, mcts_history):
        if self.csv_writer is None:
            return False

        values = {}
        if info["winner"] == 'b':
            values["b"] = 1
            values["r"] = -1
        else:
            values["b"] = -1
            values["r"] = 1
        for i in range(len(state_history)):
            if i % 2 == 0:
                value = values["b"]
            else:
                value = values["r"]
            self.csv_writer.writerow([value, json.dumps(state_history[i]), json.dumps(mcts_history[i])])

    def make_dataset(self, filenames, batch_size, shuffle_buffer_size=100, num_dataset_parallel=4):
        def decode_line(line):
            items = tf.decode_csv(line, [[""], [""], [""]], field_delim=",")
            return items

        if len(filenames) > 1:
            dataset = tf.data.Dataset.from_tensor_slices(filenames)

            dataset = dataset.flat_map(
                lambda filename: (
                    tf.data.TextLineDataset(filename).map(decode_line, num_dataset_parallel)))
        else:
            dataset = tf.data.TextLineDataset(filenames).map(decode_line, num_dataset_parallel)

        if shuffle_buffer_size > 0:
            dataset = dataset.shuffle(shuffle_buffer_size)
        self.dataset = dataset.batch(batch_size).make_initializable_iterator()

    def batch(self):
        value_data, state_data, policy_data = self.sess.run(self.dataset.get_next())
        state_data = np.array(list(map(lambda x: np.array(json.loads(x.decode("utf-8"))), state_data)))
        policy_data = np.array(list(map(lambda x: np.array(json.loads(x.decode("utf-8"))), policy_data)))
        value_data = np.array(list(map(lambda x: float(x.decode("utf-8")), value_data)))
        return state_data, policy_data, value_data

    def init_dataset(self):
        self.sess.run(self.dataset.initializer)


