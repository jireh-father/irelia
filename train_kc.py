import tensorflow as tf
from game.game import Game
import os
from core.model import Model
from core.mcts import Mcts

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('save_dir', os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoint"),
                           "save dir")
tf.app.flags.DEFINE_string('model_file_name', "model.ckpt", "model name to save")
tf.app.flags.DEFINE_integer('max_step', 200, "max step in a episode")
tf.app.flags.DEFINE_integer('max_episode', 100000, "max episode to train")
tf.app.flags.DEFINE_integer('max_simulation', 500, "max simulation count in a mcts search")
tf.app.flags.DEFINE_integer('exploration_step', 20, "exploration step")
tf.app.flags.DEFINE_integer('batch_interval_to_save', 200, "batch interval to save model")
tf.app.flags.DEFINE_integer('episode_interval_to_train', 100, "episode interval to train model")
tf.app.flags.DEFINE_integer('whole train_steps', 50, "train steps of whole data")
tf.app.flags.DEFINE_float('weight_decay', 0.0001, "weigh decay for weights l2 regularize")
tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate")
tf.app.flags.DEFINE_float('learning_rate_decay', 0.1, "learning rate decay")
tf.app.flags.DEFINE_integer('learning_rate_decay_interval', 10000, "learning rate decay interval")
tf.app.flags.DEFINE_integer('batch_size', 64, "batch size")
tf.app.flags.DEFINE_float('c_puct', 0.5, "a constant determining the level of exploration")

if FLAGS.checkpoint and not os.path.exists(os.path.dirname(FLAGS.checkpoint)):
    os.mkdir(os.path.dirname(FLAGS.checkpoint))

env = Game.make("KoreanChess-v1", {"use_check": False, "limit_step": FLAGS.max_step})

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
model = Model(sess, weight_decay=FLAGS.weight_decay)
saver = tf.train.Saver()

if not os.path.exists(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)
checkpoint_path = os.path.join(FLAGS.save_dir, FLAGS.model_file_name)
if os.path.exists(checkpoint_path + ".index"):
    saver.restore(sess, checkpoint_path)
tf.summary.FileWriter(os.path.dirname(FLAGS.save_dir), sess.graph)
output = open(os.path.join(os.path.dirname(FLAGS.save_dir), "self_play_data.txt"), "a+")

game_results = {"b": 0, "r": 0, "d": 0}

for i_episode in range(FLAGS.max_episode):
    state = env.reset()
    mcts = Mcts(state, env, model, FLAGS.max_simulation, c_puct=FLAGS.c_puct)
    state_history = [state]
    # todo : temperature change
    temperature = 1
    for step in range(FLAGS.max_step):
        if step >= FLAGS.exploration_step:
            temperature = 0
        search_action_probs = mcts.search(temperature)
        actions = env.get_all_actions()
        action = actions[search_action_probs.argmax()]
        try:
            state, reward, done, info = env.step(action)
            state_history.append(state)
        except Exception as e:
            continue

        if done:
            if info["over_limit_step"] or info["is_draw"]:
                # todo: check winner 점수계산?
                game_results["d"] += 1
            else:
                game_results[env.next_turn] += 1
            break
    print("Blue wins : %d, Red winds : %d, Draws : %d" % (game_results["b"], game_results["r"], game_results["d"]))
    # save history data to train
    output.write(json.dumps({"action": action_history, "state": state_history}) + "\n")
    # save data
    if i_episode > 0 and i_episode % FLAGS.episode_interval_to_train == 0:
        # read saved data
        train_steps = data_cnt / FLAGS.batch_size * FLAGS.whole_train_steps
        for i in range(train_steps):
            data = get_batch()
            model.train(data)  # todo: learning rate decay
            # save model
            if i > 0 and i % FLAGS.batch_interval_to_save == 0:
                result = saver.save(sess, checkpoint_path)
                # todo : evaluate best player
output.close()
