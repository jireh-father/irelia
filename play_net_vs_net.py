from core import play
import tensorflow as tf
from util import common
import os
from core.model import Model
from game.game import Game

FLAGS = tf.app.flags.FLAGS

common.set_flags()
common.make_dirs(os.path.join(FLAGS.save_dir, "dataset_ready"))

env = Game.make("KoreanChess-v1",
                {"use_check": False, "limit_step": FLAGS.max_step, "use_color_print": True,
                 "interval": 1})

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
model = Model(sess, weight_decay=FLAGS.weight_decay, momentum=FLAGS.momentum, num_layers=FLAGS.num_model_layers,
              use_cache=FLAGS.use_cache)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

common.restore_model("F:\data\irelia", "new_model_20171204045347", saver, sess)
info = play.self_play_only_net(env, model, FLAGS.max_step)

print(info)
