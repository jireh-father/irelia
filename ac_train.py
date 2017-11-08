# coding=utf8
"""
source reference : https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/8_Actor_Critic_Advantage/AC_CartPole.py
"""

import numpy as np
import tensorflow as tf
from game.game import Game
from model import resnet
import json
import os
import copy
from model import actor_critic

FLAGS = tf.app.flags.FLAGS

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters
MAX_EPISODE = 100000
MAX_EP_STEPS = 200  # maximum time step in one episode
GAMMA = 0.9  # reward discount in TD error
LR_A = 0.001  # learning rate for actor
LR_C = 0.01  # learning rate for critic

checkpoint_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoint", "model.ckpt")
if checkpoint_path and not os.path.exists(os.path.dirname(checkpoint_path)):
    os.mkdir(os.path.dirname(checkpoint_path))

env = Game.make("KoreanChess-v1", {"use_check": False, "limit_step": MAX_EP_STEPS})

N_F = 9 * 10  # env.observation_space.shape[0]
N_A = 9 * 10  # env.action_space.n


def print_episode(track_r):
    ep_rs_sum = sum(track_r)

    # if 'running_reward' not in globals():
    #     running_reward = ep_rs_sum
    # else:
    #     running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
    print("episode:", i_episode, "  reward:", ep_rs_sum)


sess = tf.Session()
ph_state = tf.placeholder(tf.float32, [1, 10, 9, 3], "state")
conv_logits = resnet.model(ph_state, blocks=20, data_format="channels_last")
conv_logits = tf.reshape(conv_logits, [-1, N_F], name="reshape")
actor = actor_critic.Actor(sess, input=conv_logits, input_ph=ph_state, n_actions=N_A, lr=LR_A)
critic = actor_critic.Critic(sess, input=conv_logits, input_ph=ph_state,
                             lr=LR_C)  # we need a good teacher, so the teacher should learn faster than the actor

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

if checkpoint_path:
    if os.path.exists(checkpoint_path + ".index"):
        saver.restore(sess, checkpoint_path)
    tf.summary.FileWriter(os.path.dirname(checkpoint_path), sess.graph)
    output = open(os.path.join(os.path.dirname(checkpoint_path), "history.txt"), "w+")

sess.run(init_op)

for i_episode in range(MAX_EPISODE):
    s_blue_ = env.reset()
    track_r = []
    state_list = [s_blue_.tolist()]
    action_list = []
    r_red = None
    s_blue = None
    s_red = None
    while True:
        """ blue: get a action """
        a_blue = actor.choose_action(s_blue_, env)
        action_list.append(a_blue)

        """ blue: step """
        s_red_, r_blue, done, _ = env.step(a_blue)
        print("reward", r_blue)
        track_r.append(r_blue)
        # blue : encode action for train
        a_blue = actor_critic.encode_action(a_blue)
        state_list.append(s_red_.tolist())
        """ red: reverse step for same training """
        s_red_ = actor_critic.reverse_state(s_red_)

        """ red: train"""
        if r_red is not None:
            td_error = critic.learn(s_red, r_red - r_blue, s_red_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(s_red, a_red[0], a_red[1], td_error)  # true_gradient = grad[logPi(s,a) * td_error]

        """ blue: if win ( done ) """
        if done:
            """ blue : train """
            td_error = critic.learn(s_blue, r_blue, s_blue_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(s_blue, a_blue[0], a_blue[1], td_error)  # true_gradient = grad[logPi(s,a) * td_error]
            print_episode(track_r)

            break

        """ blue : back up old state """
        s_blue = s_blue_

        """ red : get a action """
        a_red = actor.choose_action(s_red_, env)
        action_list.append(a_red)

        """ red : step """
        s_blue_, r_red, done, _ = env.step(a_red)
        print("reward", r_red)
        track_r.append(r_red)
        # red: encode action for train
        a_red = env.reverse_action(a_red)
        a_red = actor_critic.encode_action(a_red)
        state_list.append(s_blue_.tolist())

        """ blue : train """
        td_error = critic.learn(s_blue, r_blue - r_red, s_blue_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s_blue, a_blue[0], a_blue[1], td_error)  # true_gradient = grad[logPi(s,a) * td_error]

        """ red: if win ( done ) """
        if done:
            """ red: train """
            td_error = critic.learn(s_red, r_red, s_red_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(s_red, a_red[0], a_red[1], td_error)  # true_gradient = grad[logPi(s,a) * td_error]
            print_episode(track_r)
            break

        """ red : back up old state """
        s_red = s_red_
    if checkpoint_path:
        output.write(json.dumps({"action": action_list, "state": state_list}) + "\n")
        if i_episode != 0 and i_episode % 100 == 0:
            result = saver.save(sess, checkpoint_path)
if checkpoint_path:
    output.close()
    result = saver.save(sess, checkpoint_path)
    print("Model saved in file: %s" % result)
