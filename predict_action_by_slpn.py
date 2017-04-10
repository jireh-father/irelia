import tensorflow as tf
from util import neural_network as nn
import numpy as np
import os
from util import gibo_csv_reader as reader
from core import game
import operator

F = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_repeat_layers', 11, 'The number of cnn repeat layers.')
tf.app.flags.DEFINE_integer('num_filters', 192, 'The number of cnn filters.')
tf.app.flags.DEFINE_string('data_format', 'NCHW', 'cnn data format')
tf.app.flags.DEFINE_string('checkpoint_path', '/home/igseo/data/korean_chess/train_log/sl_policy_network.ckpt',
                           'cnn data format')
tf.app.flags.DEFINE_string('state_key',
                           'r6,r4,r2,r3,1,r3,r2,r4,r6,4,r7,5,r5,5,r5,1,r1,1,r1,1,r1,1,r1,1,r1,18,b1,1,b1,1,b1,1,b1,1,b1,1,b5,5,b5,5,b7,4,b6,b2,b4,b3,1,b3,b4,b2,b6',
                           'current state')

tf.app.flags.DEFINE_string('color', 'b', 'current turn')

width = 9
height = 10
num_input_feature = 3

if F.data_format is 'NCHW':
    inputs = tf.placeholder(tf.float16, [None, num_input_feature, height, width], name='inputs')
else:
    inputs = tf.placeholder(tf.float16, [None, height, width, num_input_feature], name='inputs')

logits, end_points = nn.sl_policy_network(inputs, F.num_repeat_layers, F.num_filters,
                                          data_format=F.data_format)

argmax = tf.argmax(end_points['Predictions'], 2)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

saver = tf.train.Saver()
saver.restore(sess, F.checkpoint_path)

x_train = game.convert_state_feature_map(F.state_key, F.color)
if F.data_format is not 'NCHW':
    x_train = np.transpose(x_train, (0, 2, 3, 1))
result, pred = sess.run([argmax, end_points['Predictions']], {inputs: x_train})

from_list = np.argsort(-pred[0][0])
to_list = np.argsort(-pred[0][1])
print(from_list)
print(to_list)
print(pred[0][0][from_list])
print(pred[0][1][to_list])
actions_dict = game.get_actions_hash_map(F.state_key, F.color)
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

result_dict = sorted(result_dict.items(), key=operator.itemgetter(1))
result_dict.reverse()
print(result)
print(result_dict)
key_list = [x[0] for x in result_dict]
value_list = [x[1] for x in result_dict]

# e = np.exp(np.array(value_list) / 1.0)
# probabilities = e / np.sum(e)
np_value_list = np.array(value_list)
sum = np.sum(np_value_list)
probabilities = np_value_list / sum

sample = np.random.choice(key_list, 1, p=probabilities)
print(value_list)
print(np.sum(np.array(value_list)))
print(probabilities)
print(np.sum(probabilities))
print(sample[0])
