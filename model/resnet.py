"""reference source: https://github.com/tensorflow/models/tree/master/official/resnet """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs, is_training, data_format):
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=is_training, fused=True)
    inputs = tf.nn.relu(inputs)
    return inputs


def batch_norm(inputs, is_training, data_format):
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=is_training, fused=True)
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


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format, is_training, relu=True):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    inputs = tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)
    if relu:
        return batch_norm_relu(inputs, is_training, data_format)
    else:
        return batch_norm(inputs, is_training, data_format)


def building_block(inputs, filters, is_training, projection_shortcut, strides,
                   data_format):
    shortcut = inputs

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format, is_training=is_training)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=1,
        data_format=data_format, is_training=is_training, relu=False)

    inputs = inputs + shortcut

    return tf.nn.relu(inputs)


def block_layer(inputs, filters, blocks, strides, is_training, name,
                data_format):
    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=1, strides=strides,
            data_format=data_format, is_training=is_training)

    for _ in range(0, blocks):
        inputs = building_block(inputs, filters, is_training, projection_shortcut, 1, data_format)

    return tf.identity(inputs, name)


def model(inputs, is_training=True, blocks=5, num_classes=90, data_format=None):
    if data_format is None:
        data_format = (
            'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=256, kernel_size=3, strides=1,
        data_format=data_format, is_training=is_training)

    inputs = block_layer(
        inputs=inputs, filters=256, blocks=blocks,
        strides=1, is_training=is_training, name='block_layer1',
        data_format=data_format)

    inputs = tf.reshape(inputs,
                        [-1, 512])
    inputs = tf.layers.dense(inputs=inputs, units=num_classes)
    inputs = tf.identity(inputs, 'final_dense')
    return inputs


inputs = tf.random_uniform([1, 10, 9, 2])
output = model(inputs)
print(output)

# g = tf.get_default_graph()
# print(g.get_operations())
with tf.Session() as sess:
    # # `sess.graph` provides access to the graph used in a `tf.Session`.
    writer = tf.summary.FileWriter("./test", sess.graph)
    #
    #     # Perform your computation...
    #     # for i in range(1000):
    #     sess.run(model)
    #     # ...
    #
    writer.close()
