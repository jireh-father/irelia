from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from util import neural_network as nn
from core import game
import numpy as np
import operator


def sampling_action(state_key, color, checkpoint_path, data_format='NCHW', choice_best=False):
    width = 9
    height = 10
    num_input_feature = 3

    if data_format == 'NCHW':
        inputs = tf.placeholder(tf.float16, [None, num_input_feature, height, width], name='inputs')
    else:
        inputs = tf.placeholder(tf.float16, [None, height, width, num_input_feature], name='inputs')

    logits, end_points = nn.sl_policy_network(inputs, 11, 192, data_format=data_format)

    argmax = tf.argmax(end_points['Predictions'], 2)

    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)

    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    x_train = game.convert_state_feature_map(state_key, color, data_format)
    result, pred = sess.run([argmax, end_points['Predictions']], {inputs: x_train})

    from_list = np.argsort(-pred[0][0])
    to_list = np.argsort(-pred[0][1])
    actions_dict = game.get_actions_hash_map(state_key, color)
    result_dict = {}
    for after_position in to_list:
        to_value = pred[0][1][after_position]
        to_x, to_y = game.convert_one_dim_pos_to_two_dim_pos(after_position)
        for from_position in from_list:
            if from_list[from_position] is False:
                continue
            from_value = pred[0][0][from_position]
            x, y = game.convert_one_dim_pos_to_two_dim_pos(from_position)
            pos_key = game.build_pos_key(x, y, to_x, to_y)
            if pos_key in actions_dict:
                result_dict[pos_key] = (from_value + to_value) / 2
                from_list[from_position] = False
                break
    if len(result_dict) < 1:
        return False
    result_dict = sorted(result_dict.items(), key=operator.itemgetter(1))
    result_dict.reverse()
    if choice_best:
        for key in result_dict:
            return key
    key_list = [x[0] for x in result_dict]
    value_list = [x[1] for x in result_dict]

    np_value_list = np.array(value_list)
    sum = np.sum(np_value_list)
    probabilities = np_value_list / sum

    sample = np.random.choice(key_list, 1, p=probabilities)

    sess.close()

    return sample[0]
