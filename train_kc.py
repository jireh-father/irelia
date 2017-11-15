# coding=utf8
import tensorflow as tf
from game.game import Game
import os
from core.model import Model
from core.mcts import Mcts
from util import dataset
import csv

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('save_dir', os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoint"),
                           "save dir")
tf.app.flags.DEFINE_string('model_file_name', "model.ckpt", "model name to save")
tf.app.flags.DEFINE_integer('max_step', 200, "max step in a episode")
tf.app.flags.DEFINE_integer('max_episode', 100000, "max episode")
tf.app.flags.DEFINE_integer('max_simulation', 3, "max simulation count in a mcts search")
tf.app.flags.DEFINE_integer('exploration_step', 20, "exploration step")
tf.app.flags.DEFINE_integer('batch_interval_to_save', 1, "batch interval to save model")
tf.app.flags.DEFINE_integer('episode_interval_to_train', 3, "episode interval to train model")
tf.app.flags.DEFINE_integer('epoch', 3, "epoch")
tf.app.flags.DEFINE_float('weight_decay', 0.0001, "weigh decay for weights l2 regularize")
tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate")
tf.app.flags.DEFINE_float('learning_rate_decay', 0.1, "learning rate decay")
tf.app.flags.DEFINE_integer('learning_rate_decay_interval', 1000, "learning rate decay interval")
tf.app.flags.DEFINE_integer('batch_size', 64, "batch size")
tf.app.flags.DEFINE_float('c_puct', 0.5, "a constant determining the level of exploration")
tf.app.flags.DEFINE_float('train_fraction', 0.7, "train dataset fraction")
tf.app.flags.DEFINE_float('momentum', 0.9, "momentum for optimizer")

if not os.path.exists(FLAGS.save_dir):
    os.mkdir(FLAGS.save_dir)

data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

env = Game.make("KoreanChess-v1", {"use_check": False, "limit_step": FLAGS.max_step, "data_format": data_format})

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
model = Model(sess, weight_decay=FLAGS.weight_decay, momentum=FLAGS.momentum)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
learning_rate = FLAGS.learning_rate_decay

if not os.path.exists(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)
checkpoint_path = os.path.join(FLAGS.save_dir, FLAGS.model_file_name)
if os.path.exists(checkpoint_path + ".index"):
    saver.restore(sess, checkpoint_path)
tf.summary.FileWriter(FLAGS.save_dir, sess.graph)
train_data_path = os.path.join(FLAGS.save_dir, "train_dataset.txt")
test_data_path = os.path.join(FLAGS.save_dir, "test_dataset.txt")
if os.path.exists(train_data_path):
    os.remove(train_data_path)
if os.path.exists(test_data_path):
    os.remove(test_data_path)
train_f = open(train_data_path, "w+")
test_f = open(test_data_path, "w+")
train_csv = csv.writer(train_f, delimiter=',')
test_csv = csv.writer(test_f, delimiter=',')

game_results = {"b": 0, "r": 0, "d": 0}
train_games = int(FLAGS.episode_interval_to_train * FLAGS.train_fraction)

for i_episode in range(FLAGS.max_episode):
    """self-play"""
    state = env.reset()
    mcts = Mcts(state, env, model, FLAGS.max_simulation, c_puct=FLAGS.c_puct)
    state_history = [state.tolist()]
    mcts_history = []
    temperature = 1
    winner = None
    done = False
    for step in range(FLAGS.max_step):
        print("episode: %d, step: %d" % (i_episode, step))
        if step >= FLAGS.exploration_step:
            print("temperature down")
            temperature = 0
        search_action_probs = mcts.search(temperature)
        actions = env.get_all_actions()
        action = actions[search_action_probs.argmax()]
        try:
            state, reward, done, info = env.step(action)
            mcts_history.append(search_action_probs.tolist())
        except Exception as e:
            print(e)
            continue

        if done:
            if info["over_limit_step"] or info["is_draw"]:
                # todo: check winner 점수계산?
                game_results["d"] += 1
            else:
                winner = 'b' if env.current_turn == 'r' else 'b'
                game_results[env.next_turn] += 1
            break
        state_history.append(state.tolist())
    print("Blue wins : %d, Red winds : %d, Draws : %d" % (game_results["b"], game_results["r"], game_results["d"]))

    """save self-play data"""
    if done and winner:

        episode_step = FLAGS.max_episode % FLAGS.episode_interval_to_train
        if episode_step > train_games:
            print("write self-play data for test data")
            # save test data
            dataset.write_data(test_csv, winner, state_history, mcts_history)
        else:
            print("write self-play data for train data")
            # save train data
            dataset.write_data(train_csv, winner, state_history, mcts_history)

    """train model"""
    if i_episode > 0 and i_episode % FLAGS.episode_interval_to_train == 0:
        # read saved data
        train_dataset = dataset.get_dataset(train_data_path, FLAGS.batch_size)
        test_dataset = dataset.get_dataset(test_data_path, FLAGS.batch_size)
        batch_step = 0
        print("train!")
        for epoch in range(FLAGS.epoch):
            while True:
                print("epoch: %d, step: %d, episode: %d" % (epoch, batch_step, i_episode))
                try:
                    train_batch_state, train_batch_policy, train_batch_value = dataset.get_batch(sess, train_dataset)
                    model.train(train_batch_state, train_batch_policy, train_batch_value, learning_rate)
                    # save model
                    if batch_step > 0 and batch_step % FLAGS.batch_interval_to_save == 0:
                        test_batch_state, test_batch_policy, test_batch_value = dataset.get_batch(sess, test_dataset)
                        cost = model.eval(test_batch_state, test_batch_policy, test_batch_value)
                        print("eval cost", cost)
                        print("save model")
                        result = saver.save(sess, checkpoint_path)
                        # todo : evaluate best player
                    if batch_step > 0 and batch_step % FLAGS.learning_rate_decay_interval == 0:
                        print("decay learning rate")
                        learning_rate = learning_rate * FLAGS.learning_rate_decay
                    batch_step += 1
                except tf.errors.OutOfRangeError:
                    break

        train_f.close()
        test_f.close()
        os.remove(train_data_path)
        os.remove(test_data_path)
        train_f = open(train_data_path, "w+")
        test_f = open(test_data_path, "w+")
        train_csv = csv.writer(train_f, delimiter=',')
        test_csv = csv.writer(test_f, delimiter=',')
