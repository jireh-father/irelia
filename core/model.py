# coding=utf8
"""reference source: https://github.com/tensorflow/models/tree/master/official/resnet """
import tensorflow as tf
import numpy as np

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


class Model(object):
    def __init__(self, sess, input_shape=[10, 9, 17], num_layers=20, num_classes=10 * 9, weight_decay=0.01,
                 momentum=0.9, use_cache=False):
        self.sess = sess
        self.is_training = tf.placeholder(tf.bool, shape=(), name="is_training")
        self.inputs = None
        self.policy_network = None
        self.value_network = None
        self.policy_label = None
        self.value_label = None
        self.learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
        self.cost = None
        self.train_op = None
        self.merged = None
        self.momentum = momentum
        self.use_cache = use_cache
        self.build_model(input_shape, num_layers, num_classes, weight_decay)
        self.inference_cache = {}

    def inference(self, state):
        # cache_key = str(state)
        # if self.use_cache and cache_key in self.inference_cache:
        #     return self.inference_cache[cache_key][0], self.inference_cache[cache_key][1]
        input_state = state[np.newaxis, :]
        policy, value = self.sess.run([self.policy_network, self.value_network],
                                      feed_dict={self.inputs: input_state, self.is_training: False})
        # if self.use_cache:
        #     self.inference_cache[cache_key] = [policy[0], value[0]]
        return policy[0], value[0]

    def train(self, state, policy, value, learning_rate):
        return self.sess.run([self.train_op, self.cost, self.merged],
                             feed_dict={self.inputs: state, self.is_training: True, self.policy_label: policy,
                                        self.value_label: value, self.learning_rate: learning_rate})

    def eval(self, state, policy, value):
        return self.sess.run([self.cost],
                             feed_dict={self.inputs: state, self.is_training: False, self.policy_label: policy,
                                        self.value_label: value})

    def build_model(self, input_shape, num_layers, num_classes, weight_decay):
        self.inputs = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], input_shape[2]],
                                     "inputs")
        self.policy_label = tf.placeholder(tf.float32, [None, num_classes], "policy_label")
        self.value_label = tf.placeholder(tf.float32, [None], "value_label")
        inputs = self.inputs

        network = self.conv2d_fixed_padding(
            inputs=inputs, filters=256, kernel_size=3, strides=1, name="start_conv")

        network = self.block_layer(inputs=network, filters=256, blocks=num_layers, strides=1)

        value_network = self.conv2d_fixed_padding(inputs=network, filters=1, kernel_size=1, strides=1,
                                                  name="value_conv")
        value_network = tf.reshape(value_network, [-1, num_classes], name="value_reshape")
        value_network = tf.layers.dense(inputs=value_network, units=64, name="value_dense1")

        value_network = tf.nn.relu(value_network, name="value_relu")

        value_network = tf.layers.dense(inputs=value_network, units=1, name="value_dense2")

        value_network = tf.nn.tanh(value_network, name="value_tanh")
        # value_net_inputs = tf.reshape(value_net_inputs, [-1], name="value_scalar_reshape")

        policy_network = self.conv2d_fixed_padding(inputs=network, filters=2, kernel_size=1, strides=1,
                                                   name="policy_conv")
        policy_network = tf.reshape(policy_network, [-1, num_classes * 2], name="policy_reshape")
        policy_network = tf.layers.dense(inputs=policy_network, units=num_classes, name="policy_dense")
        tf.summary.image(tensor=tf.reshape(policy_network, [-1, 10, 9, 1]), max_outputs=100, name="policy")
        self.policy_network = policy_network
        self.value_network = value_network

        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        regularizer = 0
        for weight in weights:
            regularizer += tf.nn.l2_loss(weight)
        regularizer *= weight_decay

        #     l = (z − v) 2 − πT log p + c||θ||2
        # self.cost = tf.reduce_mean(tf.pow(self.value_label - value_network, 2)) - tf.reduce_mean(
        #     self.policy_label * tf.log(policy_network)) + (0.5 * l2_regularizer)
        value_loss = tf.reduce_mean(tf.pow(self.value_label - tf.reshape(value_network, [-1]), 2))
        policy_loss = tf.reduce_mean(
            tf.nn.softmax(self.policy_label) * tf.log(tf.nn.softmax(policy_network)))
        self.cost = value_loss - policy_loss + regularizer

        tf.summary.scalar('value_loss', value_loss)
        tf.summary.scalar('policy_loss', policy_loss)
        tf.summary.scalar('total_loss', self.cost)
        tf.summary.scalar('regularizer', regularizer)
        # value_loss = tf.losses.mean_squared_error(self.value_label, value_network)
        # policy_loss = tf.losses.softmax_cross_entropy()

        self.train_op = tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.cost)
        self.merged = tf.summary.merge_all()

        # todo: accuracy! legal action probabilities and value scalar

    def batch_norm_relu(self, inputs, name):
        inputs = tf.layers.batch_normalization(
            inputs=inputs, axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
            scale=True, training=self.is_training, fused=True, name=name + "/batch")

        inputs = tf.nn.relu(inputs, name=name + "/relu")
        return inputs

    def batch_norm(self, inputs, name):
        inputs = tf.layers.batch_normalization(
            inputs=inputs, axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
            scale=True, training=self.is_training, fused=True, name=name + "/batch")
        return inputs

    def fixed_padding(self, inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
        return padded_inputs

    def conv2d_fixed_padding(self, inputs, filters, kernel_size, strides, name=None, relu=True):
        if strides > 1:
            inputs = self.fixed_padding(inputs, kernel_size)

        inputs = tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
            padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(), name=name)
        if relu:
            return self.batch_norm_relu(inputs, name)
        else:
            return self.batch_norm(inputs, name)

    def building_block(self, inputs, filters, strides, name=None):
        shortcut = inputs

        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=strides,
            name=name + "/block_start")

        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=1, relu=False, name=name + "/block_end")

        inputs = tf.add(inputs, shortcut, name=name + "/shortcut_add")

        return tf.nn.relu(inputs, name=name + "/relu")

    def block_layer(self, inputs, filters, blocks, strides):
        for i, _ in enumerate(range(0, blocks)):
            inputs = self.building_block(inputs, filters, strides, name="block_" + str(i))
            tf.summary.histogram('block_activations_%d' % i, inputs)

        return inputs

    def filter_action_probs(self, action_probs, legal_actions, env):
        legal_action_probs = []
        for legal_action in legal_actions:
            legal_action = env.encode_action(legal_action)
            legal_action_probs.append(action_probs[legal_action[0]] + action_probs[legal_action[0]])

        legal_action_probs = np.array(legal_action_probs)
        if (legal_action_probs == 0).all():
            legal_action_probs = np.array([1. / len(legal_action_probs)] * len(legal_action_probs))
        else:
            legal_action_probs = legal_action_probs / legal_action_probs.sum()
        return legal_action_probs

    def get_action_idx(self, action_probs, temperature):
        if temperature == 0:
            arg_max_list = np.argwhere(action_probs == np.amax(action_probs)).flatten()
            print("Max score:%f" % arg_max_list[0])
            if len(arg_max_list) > 1:
                action_idx = np.random.choice(arg_max_list, 1)[0]
            else:
                action_idx = action_probs.argmax()
        else:
            action_idx = np.random.choice(len(action_probs), 1, p=action_probs)[0]
        print("choice action idx %d" % action_idx)
        return action_idx
