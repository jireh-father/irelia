import tensorflow as tf
import numpy as np
import copy

GAMMA = 0.9  # reward discount in TD error


def encode_action(action):
    action_from = action["from_y"] * 9 + action["from_x"]
    action_to = action["to_y"] * 9 + action["to_x"]
    return [action_from, action_to]


def reverse_state(state):
    state = copy.deepcopy(state)
    return np.array([np.flipud(np.fliplr(state[1])), np.flipud(np.fliplr(state[0])), state[2]])


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
        print("actor loss(-) : %f" % exp_v)
        return exp_v

    def choose_action(self, s, env):
        s = np.transpose(s, [1, 2, 0])
        s = s[np.newaxis, :]

        probs = self.sess.run(self.acts_prob, {self.s: s})  # get probabilities for all actions
        probs = np.flipud(np.fliplr(probs))
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

        print("choose action")
        print("action p", actions_probs)
        print("actions", actions)
        print("choose action idx", rand_idx, actions[rand_idx])

        return actions[rand_idx]


class Critic(object):
    def __init__(self, sess, input, input_ph, lr=0.01):
        self.sess = sess

        self.s = input_ph
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')
        self.merged = None

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
        td_error, _, loss = self.sess.run(
            [self.td_error, self.train_op, self.loss],
            {self.s: s, self.v_: v_, self.r: r})
        print("critic td_error : %f, loss : %f" % (td_error, loss))
        return td_error
