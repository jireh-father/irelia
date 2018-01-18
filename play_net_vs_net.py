from core import play
import tensorflow as tf
from util import common
import os
from core.model_two_policy import Model
from game.game import Game

FLAGS = tf.app.flags.FLAGS

common.set_flags()
common.make_dirs(os.path.join(FLAGS.save_dir, "dataset_ready"))

env = Game.make("KoreanChess-v1",
                {"use_check": False, "limit_step": FLAGS.max_step, "use_color_print": FLAGS.use_color_print,
                 "interval": 0, "position_type": [0, 2]})

FLAGS.num_model_layers = 20
FLAGS.restore_model_path = "./checkpoint/new_model_20180118232921.ckpt"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
model = Model(sess, weight_decay=FLAGS.weight_decay, momentum=FLAGS.momentum, num_layers=FLAGS.num_model_layers,
              use_cache=FLAGS.use_cache, conf=FLAGS)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

if FLAGS.restore_model_path:
    common.restore_model(FLAGS.restore_model_path, None, saver, sess)
else:
    common.restore_model(FLAGS.save_dir, FLAGS.model_file_name, saver, sess)
info = play.self_play_only_net(env, model, FLAGS.max_step)

print(info)
