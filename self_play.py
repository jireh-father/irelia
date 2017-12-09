from core import play
import tensorflow as tf
from util import common
from util.dataset import Dataset
import os
from core.model import Model
from game.game import Game
import uuid
from util.common import log

FLAGS = tf.app.flags.FLAGS

common.set_flags()
common.make_dirs(os.path.join(FLAGS.save_dir, "dataset_ready"))

env = Game.make("KoreanChess-v1",
                {"use_check": False, "limit_step": FLAGS.max_step, "print_mcts_history": FLAGS.print_mcts_history,
                 "use_color_print": FLAGS.use_color_print, "use_cache": FLAGS.use_cache})

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
model = Model(sess, weight_decay=FLAGS.weight_decay, momentum=FLAGS.momentum, num_layers=FLAGS.num_model_layers,
              use_cache=FLAGS.use_cache, conf=FLAGS)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

ds = Dataset(sess)

while True:
    common.restore_model(FLAGS.save_dir, "best_model.ckpt", saver, sess)
    now = common.now_date_str_nums()
    dataset_path = os.path.join(FLAGS.save_dir, ("dataset_%s_%s.csv" % (now, uuid.uuid4())))
    ds.open(dataset_path)
    game_results = {"b": 0, "r": 0, "d": 0}
    episode = 0
    while True:
        """"""
        """self-play"""
        log("self-play episode %d" % episode)
        info, state_history, mcts_history = play.self_play(env, model, FLAGS.max_simulation, FLAGS.max_step,
                                                           FLAGS.c_puct, FLAGS.exploration_step, FLAGS.reuse_mcts,
                                                           FLAGS.print_mcts_tree, FLAGS.num_state_history,
                                                           use_reward_mcts=FLAGS.use_reward_mcts)

        if info["winner"]:
            game_results[info["winner"]] += 1
        else:
            game_results["d"] += 1
        common.log(
            "Blue wins : %d, Red wins : %d, Draws : %d" % (game_results["b"], game_results["r"], game_results["d"]))
        """"""
        """save self-play data"""
        if info["winner"]:
            ds.write(info, state_history, mcts_history, FLAGS.num_state_history)
        if game_results["b"] + game_results["r"] == common.num_selfplay_games:
            break
        episode += 1
    ds.close()
    os.rename(dataset_path, os.path.join(FLAGS.save_dir, "dataset_ready", os.path.basename(dataset_path)))
