import tensorflow as tf
from util import neural_network as nn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 16, 'The number of samples in each batch.')

width = 9
height = 10
num_input_feature = 3
num_filters = 192

inputs = tf.get_variable("inputs", shape=[FLAGS.batch_size, height, width, num_input_feature], dtype=tf.float16,
                         initializer=tf.contrib.layers.xavier_initializer())

print(inputs)
logits, end_points = nn.sl_policy_network(inputs)
print(logits, end_points)

Y = tf.placeholder(tf.float16, [FLAGS.batch_size, height, width, 2], name='label')

with tf.variable_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, Y)
    print(cross_entropy)
    cost = tf.reduce_mean(cross_entropy)
    print(cost)

# train
with tf.name_scope('train'):
    optimizer = tf.train.RMSPropOptimizer(0.1, decay=0.9, momentum=0.9, epsilon=1.0)
    # train = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(cost)
    train_op = optimizer.minimize(cost)
    print(train_op)
