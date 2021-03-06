import json
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import shutil
import csv
from util import common
import sys


class Dataset(object):
    DEF_TRAIN_FILE_NAME = "train_dataset.txt"
    DEF_TEST_FILE_NAME = "train_dataset.txt"

    def __init__(self, sess=None):
        self.sess = sess
        self.file = None
        self.csv_writer = None
        self.dataset = None
        self.dataset_iterator = None
        self.num_samples = 0
        self.get_next = None

    def open(self, file_path, mode="w"):
        if sys.version_info[0] == 3:
            self.file = open(file_path, mode=mode, newline='')
        else:
            self.file = open(file_path, mode=mode)
        self.csv_writer = csv.writer(self.file, delimiter=',')

    def close(self):
        if self.file is not None:
            self.file.close()
            self.csv_writer = None

    def write(self, info, state_history, mcts_history, num_state_history=7):
        if self.csv_writer is None:
            return False

        values = {}
        win_value = 1
        print(info)
        if info["over_limit_step"] or info["is_draw"]:
            win_value = info["score_diff"] / 73.5
        if info["winner"] == 'b':
            values["b"] = win_value
            values["r"] = -win_value
        else:
            values["b"] = -win_value
            values["r"] = win_value
        for i in range(len(state_history)):
            if i % 2 == 0:
                value = values["b"]
            else:
                value = values["r"]
            start_idx = 0 if i - num_state_history < 0 else i - num_state_history
            end_idx = i + 1
            history = state_history[start_idx:end_idx]
            new_state_history = common.convert_state_history_to_model_input(history, num_state_history)
            new_state_history = new_state_history.tolist()
            self.csv_writer.writerow([value, json.dumps(new_state_history), json.dumps(mcts_history[i])])

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

        self.dataset_iterator = dataset.batch(batch_size).make_initializable_iterator()
        self.num_samples = Dataset.get_number_of_items(filenames)
        self.get_next = self.dataset_iterator.get_next()

    @staticmethod
    def get_number_of_items(datset_files):
        nums = 0

        for file in datset_files:
            f = open(file)
            for line in f:
                nums += 1
            f.close()
        return nums

    def close_dataset(self):
        self.dataset.__init__()
        self.dataset = None
        self.dataset_iterator = None

    def batch(self):
        value_data, state_data, policy_data = self.sess.run(self.get_next)
        state_data = np.array(list(map(lambda x: np.array(json.loads(x.decode("utf-8"))), state_data)))
        policy_data = np.array(list(map(lambda x: np.array(json.loads(x.decode("utf-8"))), policy_data)))
        value_data = np.array(list(map(lambda x: float(x.decode("utf-8")), value_data)))
        return state_data, policy_data, value_data

    def init_dataset(self):
        self.sess.run(self.dataset_iterator.initializer)
