"""reference source: https://github.com/tensorflow/models/tree/master/official/resnet """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs, is_training, data_format, name):
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=is_training, fused=True, name=name + "/batch")
    inputs = tf.nn.relu(inputs, name=name + "/relu")
    return inputs


def batch_norm(inputs, is_training, data_format, name):
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=is_training, fused=True, name=name + "/batch")
    return inputs


def fixed_padding(inputs, kernel_size, data_format):
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


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format, is_training, name=None, relu=True):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    inputs = tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format, name=name)
    if relu:
        return batch_norm_relu(inputs, is_training, data_format, name)
    else:
        return batch_norm(inputs, is_training, data_format, name)


def building_block(inputs, filters, is_training, strides, data_format, name=None):
    shortcut = inputs

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format, is_training=is_training, name=name + "/block_start")

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=1,
        data_format=data_format, is_training=is_training, relu=False, name=name + "/block_end")

    inputs = tf.add(inputs, shortcut, name=name + "/shortcut_add")

    return tf.nn.relu(inputs, name=name + "/relu")


def block_layer(inputs, filters, blocks, strides, is_training, data_format):
    for i, _ in enumerate(range(0, blocks)):
        inputs = building_block(inputs, filters, is_training, strides, data_format, name="block_" + str(i))

    return inputs


def model(inputs, is_training=True, blocks=2, num_classes=90, data_format=None):
    if data_format is None:
        data_format = (
            'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2], name="channel_axis_change")

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=256, kernel_size=3, strides=1,
        data_format=data_format, is_training=is_training, name="start_conv")

    inputs = block_layer(
        inputs=inputs, filters=256, blocks=blocks,
        strides=1, is_training=is_training,
        data_format=data_format)

    value_net_inputs = conv2d_fixed_padding(
        inputs=inputs, filters=1, kernel_size=1, strides=1,
        data_format=data_format, is_training=is_training, name="value_conv")
    value_net_inputs = tf.reshape(value_net_inputs, [-1, num_classes], name="value_reshape")
    value_net_inputs = tf.layers.dense(inputs=value_net_inputs, units=64, name="value_dense1")

    value_net_inputs = tf.nn.relu(value_net_inputs, name="value_relu")

    value_net_inputs = tf.layers.dense(inputs=value_net_inputs, units=1, name="value_dense2")

    value_net_inputs = tf.nn.tanh(value_net_inputs, name="value_tanh")

    # value_net_inputs = tf.reshape(value_net_inputs, [-1], name="value_scalar_reshape")

    policy_net_inputs = conv2d_fixed_padding(
        inputs=inputs, filters=2, kernel_size=1, strides=1,
        data_format=data_format, is_training=is_training, name="policy_conv")
    policy_net_inputs = tf.reshape(policy_net_inputs, [-1, num_classes * 2], name="policy_reshape")
    policy_net_inputs = tf.layers.dense(inputs=policy_net_inputs, units=num_classes, name="policy_dense")

    return value_net_inputs, policy_net_inputs


# inputs = tf.random_uniform([1, 10, 9, 2])
# value_net_inputs, policy_net_inputs = model(inputs)
# print(value_net_inputs, policy_net_inputs)
#
# # g = tf.get_default_graph()
# # print(g.get_operations())
# with tf.Session() as sess:
#     # # `sess.graph` provides access to the graph used in a `tf.Session`.
#     writer = tf.summary.FileWriter("./test", sess.graph)
#     #
#     #     # Perform your computation...
#     #     # for i in range(1000):
#     #     sess.run(model)
#     #     # ...
#     #
#     writer.close()
