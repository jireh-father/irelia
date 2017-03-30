import tensorflow as tf
from util import neural_network as nn
import numpy as np
import os
from util import gibo_csv_reader as reader

F = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_epoch', 10, 'Max Epoch.')
tf.app.flags.DEFINE_integer('batch_size', 64, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('num_filters', 192, 'The number of cnn filters.')
tf.app.flags.DEFINE_integer('num_repeat_layers', 11, 'The number of cnn repeat layers.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning_rate.')
tf.app.flags.DEFINE_string('data_path', '/home/igseo/data/korean_chess/records.csv', 'training data path')
tf.app.flags.DEFINE_string('data_format', 'NCHW', 'cnn data format')
tf.app.flags.DEFINE_string('checkpoint_path', '/home/igseo/data/korean_chess/train/sl_policy_network.ckpt',
                           'cnn data format')
tf.app.flags.DEFINE_integer('save_interval_epoch', 1, 'Save Interval by Epoch.')
tf.app.flags.DEFINE_integer('print_interval_steps', 10, 'Print Interval by steps.')
tf.app.flags.DEFINE_integer('validation_interval_steps', 30, 'Validation Interval by steps.')

width = 9
height = 10
num_input_feature = 3

if F.data_format is 'NCHW':
    inputs = tf.placeholder(tf.float16, [None, num_input_feature, height, width], name='inputs')
    labels = tf.placeholder(tf.float16, [None, 2, height * width], name='labels')
else:
    inputs = tf.placeholder(tf.float16, [None, height, width, num_input_feature], name='inputs')
    labels = tf.placeholder(tf.float16, [None, height * width * 2], name='labels')

logits, end_points = nn.sl_policy_network(inputs, F.num_repeat_layers, F.num_filters,
                                          data_format=F.data_format)

with tf.variable_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(cross_entropy)

# train
with tf.name_scope('train'):
    optimizer = tf.train.RMSPropOptimizer(F.learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
    # train = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(cost)
    train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

saver = tf.train.Saver()
if os.path.isfile(F.checkpoint_path):
    saver.restore(sess, F.checkpoint_path)

if not os.path.isdir(os.path.dirname(F.checkpoint_path)):
    os.makedirs(os.path.dirname(F.checkpoint_path))

train_inputs, train_labels, valid_inputs, valid_labels = reader.load_train(F.data_path)
train_cnt = len(train_labels)

train_indices = np.arange(train_cnt)
valid_indices = np.arange(len(valid_labels))

train_cnt = len(train_labels)
steps = train_cnt // F.batch_size
# valid_cnt
for epoch in range(F.max_epoch):
    for i in range(steps):
        rand_train_indices = np.random.choice(train_indices, size=F.batch_size)
        x_train = train_inputs[rand_train_indices]
        y_train = train_labels[rand_train_indices]
        if F.data_format is not 'NCHW':
            x_train = np.transpose(x_train, (0, 2, 3, 1))
            y_train = np.transpose(y_train, (0, 2, 3, 1))

        curr_loss, curr_logits, _, pred = sess.run(
            [loss, logits, train, end_points['Predictions']], {inputs: x_train, labels: y_train})

        if i % F.print_interval_steps is 0:
            print("train loss: %s" % curr_loss)
            print(curr_logits[0])
            print(pred[0])
        if i % F.validation_interval_steps is 0:
            rand_valid_indices = np.random.choice(valid_indices, size=F.batch_size)
            x_valid = valid_inputs[rand_valid_indices]
            y_valid = valid_labels[rand_valid_indices]
            if F.data_format is not 'NCHW':
                x_valid = np.transpose(x_valid, (0, 2, 3, 1))
                y_valid = np.transpose(y_valid, (0, 2, 3, 1))

            valid_loss, valid_logits, _ = sess.run(
                [loss, logits, train], {inputs: x_valid, labels: y_valid})
            print("validation loss: %s" % valid_loss)
        if epoch < 1 and i % F.validation_interval_steps is 0:
            saver.save(sess, F.checkpoint_path)
    if epoch > 0 and epoch % F.save_interval_epoch is 0:
        saver.save(sess, F.checkpoint_path)
        # before = curr_logits[0, :, :, 0].flatten()
        # after = curr_logits[0, :, :, 1].flatten()
        # print(before)
        # sort_key = before.argsort()[-10:]
        # print(before[sort_key])
        # sort_key = after.argsort()[-10:]
        # print(after[sort_key])
        # print(curr_logits[0, :, :, 1])
        # print(curr_logits[0][:, 1])

# x_train = [[[[.6], [.4], [.2], [.3], [0], [.3], [.4], [.2], [.6]],
#             [[0], [0], [0], [0], [1], [0], [0], [0], [0]],
#             [[0], [.5], [0], [0], [0], [0], [0], [.5], [0]],
#             [[.1], [0], [.1], [0], [.1], [0], [.1], [0], [.1]],
#             [[0], [0], [0], [0], [0], [0], [0], [0], [0]],
#             [[0], [0], [0], [0], [0], [0], [0], [0], [0]],
#             [[.1], [0], [.1], [0], [.1], [0], [.1], [0], [.1]],
#             [[0], [.5], [0], [0], [0], [0], [0], [.5], [0]],
#             [[0], [0], [0], [0], [1], [0], [0], [0], [0]],
#             [[.6], [.4], [.2], [.3], [0], [.3], [.4], [.2], [.6]]]]
#
# y_train = [[[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
#             [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
#             [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
#             [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
#             [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
#             [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
#             [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [1, 0]],
#             [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
#             [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
#             [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]]]
x_train = [[[[.6, .4, .2, .3, 0, .3, .4, .2, .6],
             [0, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, .5, 0, 0, 0, 0, 0, .5, 0],
             [.1, 0, .1, 0, .1, 0, .1, 0, .1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [.1, 0, .1, 0, .1, 0, .1, 0, .1],
             [0, .5, 0, 0, 0, 0, 0, .5, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0],
             [.6, .4, .2, .3, 0, .3, .4, .2, .6]], ], ]

y_train = [[[[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]], ], ]
