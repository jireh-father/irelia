"""reference source: https://github.com/tensorflow/models/tree/master/official/resnet """
import tensorflow as tf
import math

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


class Model(object):
    def __init__(self, sess, input_shape=[10, 9, 3], num_layers=20, num_classes=10 * 9 + 1, weight_decay=0.01):
        self.sess = sess
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.inputs = None
        self.policy_network = None
        self.value_network = None
        self.policy_label = None
        self.value_label = None
        self.learning_rate = None
        self.build_model(input_shape, num_layers, num_classes, weight_decay)

    def inference(self, state):
        # todo: state 다수로 구조 변경
        return self.sess.run([], feed_dict={self.inputs: state, self.is_training: False})

    def train(self, state):
        return self.sess.run([], feed_dict={self.inputs: state, self.is_training: True, self.policy_label: True,
                                            self.value_label: True})

    def build_model(self, input_shape, num_layers, num_classes, weight_decay):
        self.inputs = tf.placeholder(tf.float32, [-1, input_shape[0], input_shape[1], input_shape[2]],
                                     "inputs")
        self.policy_label = tf.placeholder(tf.float32, [-1, num_classes], "policy_label")
        self.value_label = tf.placeholder(tf.float32, [-1, 1], "value_label")
        inputs = self.inputs
        data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

        if data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 3, 1, 2], name="channel_axis_change")

        network = self.conv2d_fixed_padding(
            inputs=inputs, filters=256, kernel_size=3, strides=1, data_format=data_format, name="start_conv")

        network = self.block_layer(inputs=network, filters=256, blocks=num_layers, strides=1, data_format=data_format)

        value_network = self.conv2d_fixed_padding(inputs=network, filters=1, kernel_size=1, strides=1,
                                                  data_format=data_format, name="value_conv")
        value_network = tf.reshape(value_network, [-1, num_classes], name="value_reshape")
        value_network = tf.layers.dense(inputs=value_network, units=64, name="value_dense1")

        value_network = tf.nn.relu(value_network, name="value_relu")

        value_network = tf.layers.dense(inputs=value_network, units=1, name="value_dense2")

        value_network = tf.nn.tanh(value_network, name="value_tanh")
        # value_net_inputs = tf.reshape(value_net_inputs, [-1], name="value_scalar_reshape")

        policy_network = self.conv2d_fixed_padding(inputs=network, filters=2, kernel_size=1, strides=1,
                                                   data_format=data_format, name="policy_conv")
        policy_network = tf.reshape(policy_network, [-1, num_classes * 2], name="policy_reshape")
        policy_network = tf.layers.dense(inputs=policy_network, units=num_classes, name="policy_dense")

        self.policy_network = policy_network
        self.value_network = value_network

        l2_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)

        #     l = (z − v) 2 − πT log p + c||θ||2
        loss = tf.reduce_mean(tf.pow(self.value_label - value_network, 2)) - tf.reduce_mean(
            self.policy_label * tf.log(policy_network)) + l2_regularizer
        # value_loss = tf.losses.mean_squared_error(self.value_label, value_network)
        # policy_loss = tf.losses.softmax_cross_entropy()

    def batch_norm_relu(self, inputs, data_format, name):
        inputs = tf.layers.batch_normalization(
            inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
            momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
            scale=True, training=self.is_training, fused=True, name=name + "/batch")

        inputs = tf.nn.relu(inputs, name=name + "/relu")
        return inputs

    def batch_norm(self, inputs, data_format, name):
        inputs = tf.layers.batch_normalization(
            inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
            momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
            scale=True, training=self.is_training, fused=True, name=name + "/batch")
        return inputs

    def fixed_padding(self, inputs, kernel_size, data_format):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        if data_format == 'channels_first':
            padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                            [pad_beg, pad_end], [pad_beg, pad_end]])
        else:
            padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                            [pad_beg, pad_end], [0, 0]])
        return padded_inputs

    def conv2d_fixed_padding(self, inputs, filters, kernel_size, strides, data_format, name=None, relu=True):
        if strides > 1:
            inputs = self.fixed_padding(inputs, kernel_size, data_format)

        inputs = tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
            padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
            data_format=data_format, name=name)
        if relu:
            return self.batch_norm_relu(inputs, data_format, name)
        else:
            return self.batch_norm(inputs, data_format, name)

    def building_block(self, inputs, filters, strides, data_format, name=None):
        shortcut = inputs

        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=strides,
            data_format=data_format, name=name + "/block_start")

        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=1,
            data_format=data_format, relu=False, name=name + "/block_end")

        inputs = tf.add(inputs, shortcut, name=name + "/shortcut_add")

        return tf.nn.relu(inputs, name=name + "/relu")

    def block_layer(self, inputs, filters, blocks, strides, data_format):
        for i, _ in enumerate(range(0, blocks)):
            inputs = self.building_block(inputs, filters, strides, data_format, name="block_" + str(i))

        return inputs
