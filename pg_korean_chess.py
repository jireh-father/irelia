# coding=utf8
"""
source reference : https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/8_Actor_Critic_Advantage/AC_CartPole.py
"""

import numpy as np
import tensorflow as tf
from game.game import Game
from model import resnet
import json


np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters
OUTPUT_GRAPH = True
MAX_EPISODE = 1
MAX_EP_STEPS = 400  # maximum time step in one episode
GAMMA = 0.9  # reward discount in TD error
LR_A = 0.001  # learning rate for actor
LR_C = 0.01  # learning rate for critic

env = Game.make("KoreanChess-v1", {"use_check": False, "limit_step": MAX_EP_STEPS})

N_F = 9 * 10  # env.observation_space.shape[0]
N_A = 9 * 10  # env.action_space.n


def print_episode(track_r):
    ep_rs_sum = sum(track_r)

    # if 'running_reward' not in globals():
    #     running_reward = ep_rs_sum
    # else:
    #     running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
    print("episode:", i_episode, "  reward:", int(ep_rs_sum))


def reverse_state(state):
    np.flipud(np.fliplr(state[1]))
    return np.array([np.flipud(np.fliplr(state[1])), np.flipud(np.fliplr(state[0])), state[2]])


def encode_action(action):
    action_from = action["from_y"] * 9 + action["from_x"]
    action_to = action["to_y"] * 9 + action["to_x"]
    return [action_from, action_to]


class Actor(object):
    def __init__(self, sess, input, input_ph, n_actions, lr=0.001):
        self.sess = sess
        self.s = input_ph
        self.a_from = tf.placeholder(tf.int32, None, "act_from")
        self.a_to = tf.placeholder(tf.int32, None, "act_to")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            # l1 = tf.layers.dense(
            #     inputs=input,
            #     units=20,  # number of hidden units
            #     activation=tf.nn.relu,
            #     kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            #     bias_initializer=tf.constant_initializer(0.1),  # biases
            #     name='l1'
            # )
            self.acts_prob = tf.layers.dense(
                inputs=input,
                units=n_actions,  # output units
                activation=tf.nn.softmax,  # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

            with tf.variable_scope('exp_v'):
                log_prob = tf.log(self.acts_prob[0, self.a_from] + self.acts_prob[0, self.a_to])
                self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

            with tf.variable_scope('train'):
                self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a_from, a_to, td):
        s = np.transpose(s, [1, 2, 0])
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a_from: a_from, self.a_to: a_to, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = np.transpose(s, [1, 2, 0])
        s = s[np.newaxis, :]

        probs = self.sess.run(self.acts_prob, {self.s: s})  # get probabilities for all actions

        # 현재 상태에서 가능한 모든 액션 가져오기
        actions = env.get_all_actions()
        # 모든 액션을 인코하여 새로운 리스트로 만듦 to : [[from, to]]
        encoded_actions = []
        for action in actions:
            encoded_actions.append(encode_action(action))
        # 모든 액션리스트의 인덱스와 같은 리스트를 만들어서 확률값 저장장
        actions_probs = []
        for encoded_action in encoded_actions:
            actions_probs.append(probs[0, encoded_action[0]] + probs[0, encoded_action[1]])
        # 확률 최대 1값이 되도록 정규화
        total_p = sum(actions_probs)

        for i, p in enumerate(actions_probs):
            actions_probs[i] = 1 * (p / total_p)
        rand_idx = np.random.choice(np.arange(len(actions_probs)), p=actions_probs)  # return a int

        return actions[rand_idx]


class Critic(object):
    def __init__(self, sess, input, input_ph, lr=0.01):
        self.sess = sess

        self.s = input_ph
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=input,
                units=64,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

            with tf.variable_scope('squared_TD_error'):
                self.td_error = self.r + GAMMA * self.v_ - self.v
                self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
            with tf.variable_scope('train'):
                self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s = np.transpose(s, [1, 2, 0])
        s_ = np.transpose(s_, [1, 2, 0])
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.s: s, self.v_: v_, self.r: r})
        return td_error


sess = tf.Session()
ph_state = tf.placeholder(tf.float32, [1, 10, 9, 3], "state")
conv_logits = resnet.model(ph_state, 1, data_format="channels_last")
conv_logits = tf.reshape(conv_logits, [-1, N_F], name="reshape")
actor = Actor(sess, input=conv_logits, input_ph=ph_state, n_actions=N_A, lr=LR_A)
critic = Critic(sess, input=conv_logits, input_ph=ph_state,
                lr=LR_C)  # we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

output = open("train_history.txt", "w+")

for i_episode in range(MAX_EPISODE):
    s_blue_ = env.reset()
    track_r = []
    state_list = [s_blue_.tolist()]
    r_red = None
    s_blue = None
    s_red = None
    while True:

        a = actor.choose_action(s_blue_)

        s_red_, r_blue, done, _ = env.step(a)

        a = encode_action(a)
        state_list.append(s_red_.tolist())

        if done:
            track_r.append(r_blue)

            td_error = critic.learn(s_blue, r_blue, s_blue_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(s_blue, a[0], a[1], td_error)  # true_gradient = grad[logPi(s,a) * td_error]
            print_episode(track_r)
            break

        if s_blue is not None:
            r = r_red - r_blue
            track_r.append(r)

            td_error = critic.learn(s_blue, r, s_blue_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(s_blue, a[0], a[1], td_error)  # true_gradient = grad[logPi(s,a) * td_error]

        s_blue = s_blue_

        a = actor.choose_action(s_red_)
        s_red_ = reverse_state(s_red_)
        s_blue_, r_red, done, _ = env.step(a)
        a = encode_action(a)
        state_list.append(s_blue_.tolist())
        if s_red is not None:
            r = r_blue - r_red
            track_r.append(r)

            td_error = critic.learn(s_red, r, s_red_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(s_red, a[0], a[1], td_error)  # true_gradient = grad[logPi(s,a) * td_error]

        if done:
            print_episode(track_r)
            break

        s_red = s_red_
    output.write(json.dumps(state_list) + "\n")
output.close()
