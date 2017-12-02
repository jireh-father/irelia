import os
import shutil
import datetime
import time
import tensorflow as tf
import time
import numpy as np

num_opt_games = 1000
num_eval_games = 300
num_selfplay_games = 50
num_checkpoint_epochs = 1


def restore_model(save_dir, model_file_name, saver, sess, restore_pending=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint_path = os.path.join(save_dir, model_file_name) + ".ckpt"
    while True:
        if os.path.exists(checkpoint_path + ".data-00000-of-00001"):
            print("restore success!!")
            try:
                saver.restore(sess, checkpoint_path)
                break
            except Exception as e:
                print("restore exception", e)
        if not restore_pending:
            return False
        print("waiting for model...")
        time.sleep(0.5)
    return checkpoint_path


def log(msg):
    dt = now_date_str()
    print("[%s] %s" % (dt, msg))
    pass


def set_flags():
    tf.app.flags.DEFINE_string('save_dir',
                               os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "checkpoint"),
                               "save dir")

    tf.app.flags.DEFINE_string('model_file_name', "model", "model name to save")
    tf.app.flags.DEFINE_integer('max_step', 10, "max step in a episode")
    tf.app.flags.DEFINE_integer('max_episode', 1000000, "max episode")
    tf.app.flags.DEFINE_integer('max_simulation', 5, "max simulation count in a mcts search")
    tf.app.flags.DEFINE_integer('exploration_step', 20, "exploration step")
    tf.app.flags.DEFINE_integer('episode_interval_to_train', 2, "episode interval to train model")
    tf.app.flags.DEFINE_integer('epoch', 20, "epoch")
    tf.app.flags.DEFINE_integer('num_state_history', 7, "num_state_history")
    tf.app.flags.DEFINE_integer('num_model_layers', 20, "numbers of model layers")
    tf.app.flags.DEFINE_float('weight_decay', 0.0001, "weigh decay for weights l2 regularize")
    tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate")
    tf.app.flags.DEFINE_float('learning_rate_decay', 0.1, "learning rate decay")
    tf.app.flags.DEFINE_integer('learning_rate_decay_interval', 1000, "learning rate decay interval")
    tf.app.flags.DEFINE_integer('batch_size', 64, "batch size")
    tf.app.flags.DEFINE_float('c_puct', 0.01, "a constant determining the level of exploration")
    tf.app.flags.DEFINE_float('momentum', 0.9, "momentum for optimizer")
    tf.app.flags.DEFINE_boolean('print_mcts_history', False, "show mcts search history")
    tf.app.flags.DEFINE_boolean('print_mcts_tree', True, "show mcts search tree")
    tf.app.flags.DEFINE_boolean('use_color_print', False, "use color in printing state")
    tf.app.flags.DEFINE_boolean('use_cache', False, "use cache")
    tf.app.flags.DEFINE_boolean('reuse_mcts', False, "reuse mcts")


def convert_state_history_to_model_input(state_history, num_state_history):
    blue_history = []
    red_history = []
    if len(state_history) <= num_state_history:
        for i in range(num_state_history - len(state_history) + 1):
            blue_history.append([[0] * 9 for i in range(10)])
            red_history.append([[0] * 9 for i in range(10)])
    for state in state_history:
        blue_history.append(state[0])
        red_history.append(state[1])

    new_state_history = blue_history + red_history
    new_state_history = (np.array(new_state_history) / 7).tolist()
    new_state_history.append(state_history[-1][2])
    new_state_history = np.transpose(np.array(new_state_history), [1, 2, 0])
    return new_state_history


def now_date_str_nums():
    return datetime.datetime.now().strftime('%Y%m%d%H%M%S')


def now_date_str():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
