from core import play
import tensorflow as tf
from util import common
import os
from core.model import Model
from game.game import Game
from core import play

FLAGS = tf.app.flags.FLAGS

common.set_flags()
common.make_dirs(os.path.join(FLAGS.save_dir, "dataset_ ready"))

env = Game.make("KoreanChess-v1",
                {"use_check": False, "limit_step": FLAGS.max_step, "use_color_print": True})

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
model = Model(sess, weight_decay=FLAGS.weight_decay, momentum=FLAGS.momentum, num_layers=FLAGS.num_model_layers,
              use_cache=FLAGS.use_cache)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

common.restore_model(FLAGS.save_dir, FLAGS.model_file_name, saver, sess)

info = play.eval_play(env, model, model, 300, FLAGS.max_step, 0.1, print_mcts_search=True)
print(info)
