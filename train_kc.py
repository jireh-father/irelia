import tensorflow as tf
from game.game import Game
import os
from core.model import Model
from core.mcts import Mcts

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('save_dir', os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoint"),
                           "save dir")
tf.app.flags.DEFINE_string('model_name', "model.ckpt", "model name to save")
tf.app.flags.DEFINE_integer('max_step', 200, "max step in a episode")
tf.app.flags.DEFINE_integer('max_episode', 100000, "max episode to train")
tf.app.flags.DEFINE_integer('max_simulation', 500, "max simulation count in a mcts search")
tf.app.flags.DEFINE_integer('exploration_step', 20, "exploration step")
tf.app.flags.DEFINE_float('gpu_memory_fraction', 0.3, "gpu memory fraction to use")

if FLAGS.checkpoint and not os.path.exists(os.path.dirname(FLAGS.checkpoint)):
    os.mkdir(os.path.dirname(FLAGS.checkpoint))

env = Game.make("KoreanChess-v1", {"use_check": False, "limit_step": FLAGS.max_step})

model = Model()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

if FLAGS.checkpoint_path:
    if os.path.exists(FLAGS.checkpoint_path + ".index"):
        saver.restore(sess, FLAGS.checkpoint_path)
    tf.summary.FileWriter(os.path.dirname(FLAGS.checkpoint_path), sess.graph)
    output = open(os.path.join(os.path.dirname(FLAGS.checkpoint_path), "self_play_data.txt"), "a+")

blue_wins = 0
red_wins = 0
draws = 0

for i_episode in range(FLAGS.max_episode):
    s_blue_ = env.reset()
    mcts = Mcts(s_blue_, env, model, FLAGS.max_simulation)
    reward_history = []
    state_history = [s_blue_.tolist()]
    action_history = []
    r_red = None
    s_blue = None
    s_red = None
    # todo : temperature change
    temperature = 1
    while True:
        mcts.search(temperature)

        """

        old code


        """
        """ blue: get a action """
        a_blue = actor.choose_action(s_blue_, env)
        action_history.append(a_blue)

        """ blue: step """
        s_red_, r_blue, done, info = env.step(a_blue)
        print("reward", r_blue)
        reward_history.append(r_blue)
        # blue : encode action for train
        a_blue = actor_critic.encode_action(a_blue)
        state_history.append(s_red_.tolist())
        """ red: reverse step for same training """
        s_red_ = actor_critic.reverse_state(s_red_)

        """ red: train"""
        if r_red is not None:
            td_error = critic.learn(s_red, r_red - r_blue, s_red_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(s_red, a_red[0], a_red[1], td_error)  # true_gradient = grad[logPi(s,a) * td_error]

        """ blue: if win ( done ) """
        if done:
            if info["over_limit_step"]:
                draws += 1
            else:
                blue_wins += 1
            """ blue : train """
            td_error = critic.learn(s_blue, r_blue, s_blue_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(s_blue, a_blue[0], a_blue[1], td_error)  # true_gradient = grad[logPi(s,a) * td_error]
            print_episode(reward_history)

            break

        """ blue : back up old state """
        s_blue = s_blue_

        """ red : get a action """
        a_red = actor.choose_action(s_red_, env)
        action_history.append(a_red)

        """ red : step """
        s_blue_, r_red, done, info = env.step(a_red)
        print("reward", r_red)
        reward_history.append(r_red)
        # red: encode action for train
        a_red = env.reverse_action(a_red)
        a_red = actor_critic.encode_action(a_red)
        state_history.append(s_blue_.tolist())

        """ blue : train """
        td_error = critic.learn(s_blue, r_blue - r_red, s_blue_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s_blue, a_blue[0], a_blue[1], td_error)  # true_gradient = grad[logPi(s,a) * td_error]

        """ red: if win ( done ) """
        if done:
            if info["over_limit_step"]:
                draws += 1
            else:
                red_wins += 1
            """ red: train """
            td_error = critic.learn(s_red, r_red, s_red_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(s_red, a_red[0], a_red[1], td_error)  # true_gradient = grad[logPi(s,a) * td_error]
            print_episode(reward_history)
            break

        """ red : back up old state """
        s_red = s_red_
    print("Blue wins : %d, Red winds : %d, Draws : %d" % (blue_wins, red_wins, draws))
    if checkpoint_path:
        output.write(json.dumps({"action": action_history, "state": state_history}) + "\n")
        if i_episode != 0 and i_episode % 100 == 0:
            result = saver.save(sess, checkpoint_path)
if checkpoint_path:
    output.close()
    result = saver.save(sess, checkpoint_path)
    print("Model saved in file: %s" % result)
