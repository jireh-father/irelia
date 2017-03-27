import tensorflow as tf
from util import neural_network as nn
import bottleneck

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 16, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('num_filters', 192, 'The number of cnn filters.')
tf.app.flags.DEFINE_integer('num_repeat_layers', 2, 'The number of cnn repeat layers.')
tf.app.flags.DEFINE_integer('max_number_of_steps', 100, 'The maximum number of training steps.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning_rate.')

width = 9
height = 10
num_input_feature = 1

inputs = tf.placeholder(tf.float16, [None, num_input_feature, height, width], name='inputs')
labels = tf.placeholder(tf.float16, [None, 2, height, width], name='labels')

logits, end_points = nn.sl_policy_network(inputs, FLAGS.num_repeat_layers, FLAGS.num_filters, data_format='NCHW')

with tf.variable_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(cross_entropy)

# train
with tf.name_scope('train'):
    optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
    # train = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(cost)
    train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

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

for i in range(FLAGS.max_number_of_steps):
    curr_loss, curr_logits, _ = sess.run([loss, logits, train], {inputs: x_train, labels: y_train})

    print("loss: %s " % curr_loss)
    print(curr_logits)
    # before = curr_logits[0, :, :, 0].flatten()
    # after = curr_logits[0, :, :, 1].flatten()
    # print(before)
    # sort_key = before.argsort()[-10:]
    # print(before[sort_key])
    # sort_key = after.argsort()[-10:]
    # print(after[sort_key])
    # print(curr_logits[0, :, :, 1])
    # print(curr_logits[0][:, 1])
