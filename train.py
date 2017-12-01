# coding=utf8
import tensorflow as tf
from game.game import Game
from core.model import Model
from core import optimizer
from util.dataset import Dataset
from util import common
import os
from core import play

FLAGS = tf.app.flags.FLAGS

common.set_flags()

env = Game.make("KoreanChess-v1",
                {"use_check": False, "limit_step": FLAGS.max_step, "print_mcts_history": FLAGS.print_mcts_history,
                 "use_color_print": FLAGS.use_color_print, "use_cache": FLAGS.use_cache})

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
model = Model(sess, weight_decay=FLAGS.weight_decay, momentum=FLAGS.momentum, num_layers=FLAGS.num_model_layers,
              use_cache=FLAGS.use_cache)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
learning_rate = FLAGS.learning_rate

common.restore_model(FLAGS.save_dir, FLAGS.model_file_name, saver, sess)

dataset_path = os.path.join(FLAGS.save_dir, "dataset.csv")
ds = Dataset(sess)
ds.open(dataset_path)
game_results = {"b": 0, "r": 0, "d": 0}

for episode in range(FLAGS.max_episode):
    """"""
    """self-play"""
    print("self-play episode %d" % episode)
    info, state_history, mcts_history = play.self_play(env, model, FLAGS.max_simulation, FLAGS.max_step,
                                                            FLAGS.c_puct, FLAGS.exploration_step, FLAGS.reuse_mcts,
                                                            FLAGS.print_mcts_tree, FLAGS.num_state_history)

    if info["winner"]:
        game_results[info["winner"]] += 1
    else:
        game_results["d"] += 1
    common.log("Blue wins : %d, Red winds : %d, Draws : %d" % (game_results["b"], game_results["r"], game_results["d"]))
    """"""
    """save self-play data"""
    ds.write(info, state_history, mcts_history, FLAGS.num_state_history)
    """"""
    """train model"""
    if episode > 0 and episode % FLAGS.episode_interval_to_train == 0 and os.path.getsize(dataset_path) > 0:
        ds.close()
        ds.make_dataset([dataset_path], FLAGS.batch_size)
        optimizer.train_model(model, learning_rate, ds, FLAGS.epoch, FLAGS.learning_rate_decay,
                              FLAGS.learning_rate_decay_interval)
        saver.save(sess, os.path.join(FLAGS.save_dir, "%s_%d.ckpt" % (FLAGS.model_file_name, episode)))
        # todo : evaluate best player

        ds.open(dataset_path)
