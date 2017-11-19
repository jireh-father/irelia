# coding=utf8
import tensorflow as tf
from game.game import Game
from core.model import Model
from util.dataset import Dataset
from util import common
import os
import glob
import datetime
import time
import glob

FLAGS = tf.app.flags.FLAGS

common.set_flags()

tf.app.flags.DEFINE_boolean('do_eval', False, "do eval")

data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
env = Game.make("KoreanChess-v1", {"use_check": False, "limit_step": FLAGS.max_step, "data_format": data_format,
                                   "print_mcts_history": FLAGS.print_mcts_history,
                                   "use_color_print": FLAGS.use_color_print})

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
model = Model(sess, weight_decay=FLAGS.weight_decay, momentum=FLAGS.momentum, num_layers=FLAGS.num_model_layers)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
learning_rate = FLAGS.learning_rate_decay

new_model_dir = os.path.join(FLAGS.save_dir, "new_models")
if not os.path.exists(new_model_dir):
    os.makedirs(new_model_dir)
common.restore_model(FLAGS.save_dir, "best_" + FLAGS.model_file_name, saver, sess)
# todo: check ready dataset and implement many files
ready_data_dir = os.path.join(FLAGS.save_dir, "ready")
while True:
    train_data_files = glob.glob(os.path.join(ready_data_dir, "train*"))
    if len(train_data_files) > 0:
        break


ds = Dataset(sess, FLAGS.save_dir, False, True)

""""""
"""train model"""
learning_rate = common.train_model(model, learning_rate, ds, FLAGS)
checkpoint_path = os.path.join(FLAGS.save_dir, FLAGS.model_file_name)
common.save_model(sess, saver, checkpoint_path)
if os.path.exists(checkpoint_path + ".index"):

    dt = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
    model_files = glob.glob(checkpoint_path + "*")
    for model_file in model_files:
        os.rename(model_file, os.path.join(new_model_dir, dt + os.path.splitext(model_file)[1]))
if FLAGS.do_eval:
    common.eval_model(model, ds)
# todo: remove or backup trained files
