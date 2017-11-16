import json
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import shutil


def write_data(writer, winner, state_history, mcts_history):
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


def get_dataset(data_path, batch_size=64, shuffle=1000):
    def decode_line(line):
        items = tf.decode_csv(line, [[""], [""], [""]], field_delim=",")
        return items

    base_dataset = tf.data.TextLineDataset(data_path).map(decode_line)

    if shuffle:
        base_dataset = base_dataset.shuffle(shuffle)
    base_dataset = base_dataset.batch(batch_size).make_initializable_iterator()
    return base_dataset


def get_batch(sess, dataset):
    value_data, state_data, policy_data = sess.run(dataset.get_next())
    state_data = np.array(map(lambda x: np.array(json.loads(x.decode("utf-8"))), state_data))
    policy_data = np.array(map(lambda x: np.array(json.loads(x.decode("utf-8"))), policy_data))
    value_data = np.array(map(lambda x: float(x.decode("utf-8")), value_data))
    return state_data, policy_data, value_data


def initializer(sess, dataset):
    sess.run(dataset.initializer)


def backup_dataset(data_path):
    if not os.path.exists(data_path):
        return
    if os.path.getsize(data_path) != 0:
        print("empty dataset!! remove!!")
        os.remove(data_path)
        return
    print("backup dataset!")
    dt = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
    dataset_file_name = os.path.basename(data_path)
    save_dir = os.path.dirname(data_path)
    bak_dir = os.path.join(save_dir, "dataset_" + dt)
    if not os.path.exists(bak_dir):
        os.makedirs(bak_dir)
    shutil.move(data_path, os.path.join(bak_dir, dataset_file_name))
