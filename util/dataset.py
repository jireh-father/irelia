import json
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import shutil
import csv


class Dataset(object):
    def __init__(self, sess, dataset_dir, backup=True):
        self.sess = sess
        self.dataset_dir = dataset_dir
        self.train_data_path = os.path.join(dataset_dir, "train_dataset.txt")
        self.test_data_path = os.path.join(dataset_dir, "test_dataset.txt")
        if os.path.exists(self.train_data_path) and backup:
            self.backup_dataset(self.train_data_path)
        if os.path.exists(self.test_data_path) and backup:
            self.backup_dataset(self.test_data_path)
        self.train_f = open(self.train_data_path, "a+")
        self.test_f = open(self.test_data_path, "a+")
        self.train_csv = csv.writer(self.train_f, delimiter=',')
        self.test_csv = csv.writer(self.test_f, delimiter=',')
        self.train_dataset = None
        self.test_dataset = None

    def close_files(self):
        self.train_f.close()
        self.test_f.close()

    def has_train_dataset_file(self):
        return os.path.getsize(self.train_data_path) != 0

    def has_test_dataset_file(self):
        return os.path.getsize(self.test_data_path) != 0

    def write_dataset(self, info, state_history, mcts_history, episode_step, train_games):
        if info["winner"]:
            if episode_step < train_games:
                self.write_train_data(info["winner"], state_history, mcts_history)
            else:
                self.write_test_data(info["winner"], state_history, mcts_history)

    def write_train_data(self, winner, state_history, mcts_history):
        print("write self-play data for train data")
        self.write_data(self.train_csv, winner, state_history, mcts_history)

    def write_test_data(self, winner, state_history, mcts_history):
        print("write self-play data for test data")
        self.write_data(self.test_csv, winner, state_history, mcts_history)

    def write_data(self, writer, winner, state_history, mcts_history):
        values = {}
        if winner == 'b':
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
            writer.writerow([value, json.dumps(state_history[i]), json.dumps(mcts_history[i])])

    def open_dataset(self, batch_size=64, shuffle=1000):
        if self.has_train_dataset_file():
            self.open_train_dataset(batch_size, shuffle)
            self.init_train()
        if self.has_test_dataset_file():
            self.open_test_dataset(batch_size, shuffle)
            self.init_test()

    def open_train_dataset(self, batch_size=64, shuffle=1000):
        self.train_dataset = self.get_dataset(self.train_data_path, batch_size, shuffle)

    def open_test_dataset(self, batch_size=64, shuffle=1000):
        self.test_dataset = self.get_dataset(self.test_data_path, batch_size, shuffle)

    def get_dataset(self, data_path, batch_size=64, shuffle=1000):
        def decode_line(line):
            items = tf.decode_csv(line, [[""], [""], [""]], field_delim=",")
            return items

        base_dataset = tf.data.TextLineDataset(data_path).map(decode_line)

        if shuffle:
            base_dataset = base_dataset.shuffle(shuffle)
        base_dataset = base_dataset.batch(batch_size).make_initializable_iterator()
        return base_dataset

    def get_train_batch(self):
        return self.get_batch(self.train_dataset)

    def get_test_batch(self):
        return self.get_batch(self.test_dataset)

    def get_batch(self, dataset):
        value_data, state_data, policy_data = self.sess.run(dataset.get_next())
        state_data = np.array(list(map(lambda x: np.array(json.loads(x.decode("utf-8"))), state_data)))
        policy_data = np.array(list(map(lambda x: np.array(json.loads(x.decode("utf-8"))), policy_data)))
        value_data = np.array(list(map(lambda x: float(x.decode("utf-8")), value_data)))
        return state_data, policy_data, value_data

    def init_train(self):
        self.sess.run(self.train_dataset.initializer)

    def init_test(self):
        self.sess.run(self.test_dataset.initializer)

    def backup_dataset(self, data_path):
        if not os.path.exists(data_path):
            return
        if os.path.getsize(data_path) == 0:
            print("empty dataset!! remove!!")
            os.remove(data_path)
            return
        print("backup dataset!")
        dt = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
        dataset_file_name = os.path.basename(data_path)
        bak_dir = os.path.join(self.dataset_dir, "dataset_" + dt)
        if not os.path.exists(bak_dir):
            os.makedirs(bak_dir)
        shutil.move(data_path, os.path.join(bak_dir, dataset_file_name))

    def reset(self):
        self.backup_dataset(self.train_data_path)
        self.backup_dataset(self.test_data_path)
        self.train_f = open(self.train_data_path, "w+")
        self.test_f = open(self.test_data_path, "w+")
        self.train_csv = csv.writer(self.train_f, delimiter=',')
        self.test_csv = csv.writer(self.test_f, delimiter=',')
