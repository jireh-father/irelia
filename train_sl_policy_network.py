import tensorflow as tf
from util import neural_network as nn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 16, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('num_filters', 192, 'The number of cnn filters.')
tf.app.flags.DEFINE_integer('num_repeat_layers', 11, 'The number of cnn repeat layers.')

width = 9
height = 10
num_input_feature = 1

inputs = tf.placeholder(tf.float16, [None, height, width, num_input_feature], name='inputs')
labels = tf.placeholder(tf.float16, [None, height, width, 2], name='labels')

logits, end_points = nn.sl_policy_network(inputs, FLAGS.num_repeat_layers, FLAGS.num_filters)
print(logits)

with tf.variable_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    loss = tf.reduce_mean(cross_entropy)

# train
with tf.name_scope('train'):
    optimizer = tf.train.RMSPropOptimizer(0.1, decay=0.9, momentum=0.9, epsilon=1.0)
    # train = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(cost)
    train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

x_train = [[[[.6], [.4], [.2], [.3], [0], [.3], [.4], [.2], [.6]],
            [[0], [0], [0], [0], [1], [0], [0], [0], [0]],
            [[0], [.5], [0], [0], [0], [0], [0], [.5], [0]],
            [[.1], [0], [.1], [0], [.1], [0], [.1], [0], [.1]],
            [[0], [0], [0], [0], [0], [0], [0], [0], [0]],
            [[0], [0], [0], [0], [0], [0], [0], [0], [0]],
            [[.1], [0], [.1], [0], [.1], [0], [.1], [0], [.1]],
            [[0], [.5], [0], [0], [0], [0], [0], [.5], [0]],
            [[0], [0], [0], [0], [1], [0], [0], [0], [0]],
            [[.6], [.4], [.2], [.3], [0], [.3], [.4], [.2], [.6]]]]

y_train = [[[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [1, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]]]

for i in range(1000):
    curr_loss, curr_logits, _ = sess.run([loss, logits, train], {inputs: x_train, labels: y_train})
    print("loss: %s %s" % (curr_loss, str(curr_logits)))
