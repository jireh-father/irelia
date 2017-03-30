import tensorflow as tf

num_output_filter = 2
dtype = tf.float16


def decode_shape(tensor, data_format=None):
    input_shape = list(tensor.get_shape())
    if data_format is 'NCHW':
        height = input_shape[2]
        width = input_shape[3]
        num_input_feature = input_shape[1]
    else:
        height = input_shape[1]
        width = input_shape[2]
        num_input_feature = input_shape[3]
    return height, width, num_input_feature


def sl_policy_network(inputs=None, num_repeat_layers=11, num_filters=192,
                      filter_initializer=tf.contrib.layers.xavier_initializer(), data_format=None):
    height, width, num_input_feature = decode_shape(inputs, data_format)
    end_points = {}

    # layer 1
    with tf.variable_scope("conv1"):
        filters = tf.get_variable("filter1", shape=[5, 5, num_input_feature, num_filters], dtype=dtype,
                                  initializer=filter_initializer)

        net = tf.nn.conv2d(inputs, filters, strides=[1, 1, 1, 1], padding='SAME', data_format=data_format)
        net = tf.nn.relu(net)
        end_points['conv1'] = net

    # repeat layers
    for layer_no in range(num_repeat_layers):
        layer_no += 2
        with tf.variable_scope("conv" + str(layer_no)):
            filters = tf.get_variable("filter" + str(layer_no), shape=[3, 3, num_filters, num_filters], dtype=dtype,
                                      initializer=filter_initializer)

            net = tf.nn.conv2d(net, filters, strides=[1, 1, 1, 1], padding='SAME', data_format=data_format)
            net = tf.nn.relu(net)
            end_points['conv' + str(layer_no)] = net

    # layer 13 (output)
    with tf.variable_scope("conv13"):
        filters = tf.get_variable("filter13", shape=[1, 1, num_filters, num_output_filter], dtype=dtype,
                                  initializer=filter_initializer)
        if data_format is 'NCHW':
            biases = tf.get_variable("biases", [num_output_filter, height, width],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=dtype)
        else:
            biases = tf.get_variable("biases", [height, width, num_output_filter],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=dtype)
        net = tf.nn.conv2d(net, filters, strides=[1, 1, 1, 1], padding='SAME', data_format=data_format)
        logits = net + biases

        if data_format is 'NCHW':
            end_points['OriginalLogits'] = logits
        else:
            # 64, 10, 9, 2
            end_points['OriginalLogits'] = tf.transpose(logits, perm=[0, 3, 1, 2])

        logits = tf.reshape(logits, [-1, 2, 90])
        end_points['Logits'] = logits
        end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')

    return logits, end_points


def rl_policy_network():
    pass


def value_network():
    pass
