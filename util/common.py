import os
import shutil
import datetime
import time
import tensorflow as tf
import time


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
            break
        time.sleep(0.5)
    return checkpoint_path


def log(msg):
    dt = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print("[%s] %s" % (dt, msg))
    pass


def set_flags():
    tf.app.flags.DEFINE_string('save_dir',
                               os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "checkpoint"),
                               "save dir")

    tf.app.flags.DEFINE_string('model_file_name', "model", "model name to save")
    tf.app.flags.DEFINE_integer('max_step', 200, "max step in a episode")
    tf.app.flags.DEFINE_integer('max_episode', 1000000, "max episode")
    tf.app.flags.DEFINE_integer('max_simulation', 5, "max simulation count in a mcts search")
    tf.app.flags.DEFINE_integer('exploration_step', 20, "exploration step")
    tf.app.flags.DEFINE_integer('episode_interval_to_train', 2, "episode interval to train model")
    tf.app.flags.DEFINE_integer('epoch', 20, "epoch")
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
