# coding=utf8
import tensorflow as tf
from game.game import Game
from core.model import Model
from core.mcts import Mcts
from util.dataset import Dataset
from util import common
import sys, traceback
import os
import uuid

FLAGS = tf.app.flags.FLAGS

common.set_flags()

tf.app.flags.DEFINE_integer('episode_interval_to_save_data', 10, "episode interval to save data")
tf.app.flags.DEFINE_boolean('restore_pending', False, "restore pending")

data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
env = Game.make("KoreanChess-v1", {"use_check": False, "limit_step": FLAGS.max_step, "data_format": data_format,
                                   "print_mcts_history": FLAGS.print_mcts_history,
                                   "use_color_print": FLAGS.use_color_print})

if not os.path.exists(os.path.join(FLAGS.save_dir, "ready")):
    os.makedirs(os.path.join(FLAGS.save_dir, "ready"))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
model = Model(sess, weight_decay=FLAGS.weight_decay, momentum=FLAGS.momentum, num_layers=FLAGS.num_model_layers)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
model_file_name = "best_" + FLAGS.model_file_name

checkpoint_path = common.restore_model(FLAGS.save_dir, model_file_name, saver, sess)
ds = Dataset(sess, FLAGS.save_dir, False, file_name_suffix=uuid.uuid4())

game_results = {"b": 0, "r": 0, "d": 0}
train_games = int(FLAGS.episode_interval_to_train * FLAGS.train_fraction)
i_episode = 0
while True:
    """"""
    """self-play"""
    state = env.reset()
    mcts = Mcts(state, env, model, FLAGS.max_simulation, c_puct=FLAGS.c_puct)
    state_history = [state.tolist()]
    mcts_history = []
    temperature = 1
    info = []
    for step in range(FLAGS.max_step):
        common.log("episode: %d, step: %d" % (i_episode, step))
        if step >= FLAGS.exploration_step:
            common.log("temperature down")
            temperature = 0
        actions = env.get_all_actions()
        search_action_probs, action = mcts.search(temperature)
        if FLAGS.print_mcts_tree:
            mcts.print_tree()
        try:
            state, reward, done, info = env.step(action)
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            continue

        if len(actions) != len(search_action_probs):
            sys.exit("error!!! action count!!")
        mcts_history.append(env.convert_action_probs_to_policy_probs(actions, search_action_probs))

        if done:
            if info["winner"]:
                game_results[info["winner"]] += 1
            else:
                game_results["d"] += 1
            break
        state_history.append(state.tolist())
    common.log("Blue wins : %d, Red winds : %d, Draws : %d" % (game_results["b"], game_results["r"], game_results["d"]))
    """"""
    """save self-play data"""
    ds.write_dataset(info, state_history, mcts_history, FLAGS.max_episode % FLAGS.episode_interval_to_train,
                     train_games)

    if i_episode != 0 and i_episode % FLAGS.episode_interval_to_save_data == 0:
        train_data_path, test_data_path = ds.close_files()
        if os.path.exists(train_data_path):
            train_file_name = os.path.basename(train_data_path)
            os.rename(train_data_path, os.path.join(FLAGS.save_dir, "ready", train_file_name))
        if os.path.exists(test_data_path):
            test_file_name = os.path.basename(test_data_path)
            os.rename(test_data_path, os.path.join(FLAGS.save_dir, "ready", test_file_name))
        ds.reset(file_name_suffix=uuid.uuid4())
        common.restore_model(FLAGS.save_dir, model_file_name, saver, sess, FLAGS.restore_pending)
    i_episode += 1
