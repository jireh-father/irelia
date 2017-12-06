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
num_checkpoint_epochs = 5


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
    tf.app.flags.DEFINE_integer('max_step', 150, "max step in a episode")
    tf.app.flags.DEFINE_integer('max_episode', 1000000, "max episode")
    tf.app.flags.DEFINE_integer('max_simulation', 5, "max simulation count in a mcts search")
    tf.app.flags.DEFINE_integer('exploration_step', 20, "exploration step")
    tf.app.flags.DEFINE_integer('episode_interval_to_train', 2, "episode interval to train model")
    tf.app.flags.DEFINE_integer('epoch', 20, "epoch")
    tf.app.flags.DEFINE_integer('num_state_history', 7, "num_state_history")
    tf.app.flags.DEFINE_integer('num_model_layers', 20, "numbers of model layers")
    tf.app.flags.DEFINE_float('weight_decay', 0.0001, "weigh decay for weights l2 regularize")
    tf.app.flags.DEFINE_integer('batch_size', 64, "batch size")
    tf.app.flags.DEFINE_integer('shuffle_buffer_size', 100, "shuffle_buffer_size")
    tf.app.flags.DEFINE_float('c_puct', 0.01, "a constant determining the level of exploration")
    tf.app.flags.DEFINE_float('momentum', 0.9, "momentum for optimizer")
    tf.app.flags.DEFINE_boolean('print_mcts_history', False, "show mcts search history")
    tf.app.flags.DEFINE_boolean('print_mcts_tree', True, "show mcts search tree")
    tf.app.flags.DEFINE_boolean('print_mcts_search', True, "show mcts search")
    tf.app.flags.DEFINE_boolean('use_color_print', False, "use color in printing state")
    tf.app.flags.DEFINE_boolean('use_cache', True, "use cache")
    tf.app.flags.DEFINE_boolean('reuse_mcts', True, "reuse mcts")
    tf.app.flags.DEFINE_boolean('backup_dataset', False, "backup_dataset")
    tf.app.flags.DEFINE_string('dataset_dir', None, "dataset_dir")

    ######################
    # Optimization Flags #
    ######################
    tf.app.flags.DEFINE_string(
        'optimizer', 'rmsprop',
        'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
        '"ftrl", "momentum", "sgd" or "rmsprop".')

    tf.app.flags.DEFINE_float(
        'adadelta_rho', 0.95,
        'The decay rate for adadelta.')

    tf.app.flags.DEFINE_float(
        'adagrad_initial_accumulator_value', 0.1,
        'Starting value for the AdaGrad accumulators.')

    tf.app.flags.DEFINE_float(
        'adam_beta1', 0.9,
        'The exponential decay rate for the 1st moment estimates.')

    tf.app.flags.DEFINE_float(
        'adam_beta2', 0.999,
        'The exponential decay rate for the 2nd moment estimates.')

    tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

    tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                              'The learning rate power.')

    tf.app.flags.DEFINE_float(
        'ftrl_initial_accumulator_value', 0.1,
        'Starting value for the FTRL accumulators.')

    tf.app.flags.DEFINE_float(
        'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

    tf.app.flags.DEFINE_float(
        'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

    tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

    tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

    #######################
    # Learning Rate Flags #
    #######################

    tf.app.flags.DEFINE_string(
        'learning_rate_decay_type',
        'exponential',
        'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
        ' or "polynomial"')

    tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

    tf.app.flags.DEFINE_float(
        'end_learning_rate', 0.0001,
        'The minimal end learning rate used by a polynomial decay learning rate.')

    tf.app.flags.DEFINE_float(
        'label_smoothing', 0.0, 'The amount of label smoothing.')

    tf.app.flags.DEFINE_float(
        'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

    tf.app.flags.DEFINE_float(
        'num_epochs_per_decay', 2.0,
        'Number of epochs after which learning rate decays.')

    tf.app.flags.DEFINE_bool(
        'sync_replicas', False,
        'Whether or not to synchronize the replicas during training.')

    tf.app.flags.DEFINE_integer(
        'replicas_to_aggregate', 1,
        'The Number of gradients to collect before updating params.')

    tf.app.flags.DEFINE_float(
        'moving_average_decay', None,
        'The decay to use for the moving average.'
        'If left as None, then moving averages are not used.')

    tf.app.flags.DEFINE_boolean('cycle_learning_rate', True, "cycle")


def convert_state_history_to_model_input(state_history, num_state_history=7):
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
