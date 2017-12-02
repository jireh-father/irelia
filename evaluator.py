from core import play
import tensorflow as tf
from util import common
import os
from core.model import Model
from game.game import Game
import glob
import time
from util.common import log

FLAGS = tf.app.flags.FLAGS

common.set_flags()
common.make_dirs(FLAGS.save_dir)

env = Game.make("KoreanChess-v1",
                {"use_check": False, "limit_step": FLAGS.max_step, "print_mcts_history": FLAGS.print_mcts_history,
                 "use_color_print": FLAGS.use_color_print, "use_cache": FLAGS.use_cache})

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
new_model_g = tf.Graph()
with new_model_g.as_default():
    new_model_sess = tf.Session(config=config)
    new_model = Model(new_model_sess, weight_decay=FLAGS.weight_decay, momentum=FLAGS.momentum,
                      num_layers=FLAGS.num_model_layers, use_cache=FLAGS.use_cache)
    new_model_sess.run(tf.global_variables_initializer())

best_model_g = tf.Graph()
with best_model_g.as_default():
    best_model_sess = tf.Session(config=config)
    best_model = Model(best_model_sess, weight_decay=FLAGS.weight_decay, momentum=FLAGS.momentum,
                       num_layers=FLAGS.num_model_layers, use_cache=FLAGS.use_cache)
    best_model_sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

while True:
    new_model_files = glob.glob(os.path.join(FLAGS.save_dir, "new_model_*.ckpt.index"))
    if len(new_model_files) < 1:
        log("waiting new model checkpoint...")
        time.sleep(10)
        continue
    for new_model_file in new_model_files:
        new_model_file = new_model_file[:-6]
        common.restore_model(FLAGS.save_dir, "best_model.ckpt", saver, best_model_sess, restore_pending=True)
        if not common.restore_model(FLAGS.save_dir, new_model_file, saver, new_model_sess):
            log("failed to restore new model checkpoint : %s" % new_model_file)
            continue
        game_results = {"new_model": 0, "best_model": 0, "d": 0}
        for episode in common.num_eval_games:
            log("eval-play episode %d" % episode)
            if episode % 2 == 0:
                print("blue : new model, red : best model")
                info = play.eval_play(env, new_model, best_model, FLAGS.max_simulation, FLAGS.max_step, FLAGS.c_puct,
                                      FLAGS.reuse_mcts, FLAGS.print_mcts_tree, FLAGS.num_state_history)
            else:
                print("blue : best model, red : new model")
                info = play.eval_play(env, best_model, new_model, FLAGS.max_simulation, FLAGS.max_step, FLAGS.c_puct,
                                      FLAGS.reuse_mcts, FLAGS.print_mcts_tree, FLAGS.num_state_history)

            if info["winner"]:
                who = {"r": "new_model", "b": "best_model"}
                if episode % 2 == 0:
                    who["b"] = "new_model"
                    who["r"] = "best_model"
                game_results[who[info["winner"]]] += 1
            else:
                game_results["d"] += 1
            common.log("New model wins : %d, Best model wins : %d, Draws : %d" % (
                game_results["new_model"], game_results["best_model"], game_results["d"]))
        total_wins = game_results["new_model"] + game_results["best_model"]
        if game_results["new_model"] > total_wins * 0.55:
            # change best_model
            best_model_f = glob.glob(os.path.join(FLAGS.save_dir, "best_model.ckpt*"))
            for best_f in best_model_f:
                os.rename(best_f, os.path.join(FLAGS.save_dir, "best_model_bak", os.path.basename(best_f)))

            new_model_f = glob.glob(os.path.join(FLAGS.save_dir, new_model_file + "*"))
            for new_f in new_model_f:
                os.rename(new_f,
                          os.path.join(FLAGS.save_dir, "best_model.ckpt.%s" % new_f.split(".")[-1]))
