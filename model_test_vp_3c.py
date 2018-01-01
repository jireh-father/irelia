import tensorflow as tf
from core.model import Model
import numpy as np


def configure_optimizer(learning_rate, conf):
    if conf["optimizer"] == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=conf["adadelta_rho"],
            epsilon=conf["opt_epsilon"])
    elif conf["optimizer"] == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=conf["adagrad_initial_accumulator_value"])
    elif conf["optimizer"] == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=conf["adam_beta1"],
            beta2=conf["adam_beta2"],
            epsilon=conf["opt_epsilon"])
    elif conf["optimizer"] == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=conf["ftrl_learning_rate_power"],
            initial_accumulator_value=conf["ftrl_initial_accumulator_value"],
            l1_regularization_strength=conf["ftrl_l1"],
            l2_regularization_strength=conf["ftrl_l2"])
    elif conf["optimizer"] == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=conf["momentum"],
            name='Momentum')
    elif conf["optimizer"] == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=conf["rmsprop_decay"],
            momentum=conf["rmsprop_momentum"],
            epsilon=conf["opt_epsilon"])
    elif conf["optimizer"] == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', conf["optimizer"])
    return optimizer


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

input_shape = [10, 9, 3]
num_layers = 1
num_classes = 10 * 9
weight_decay = 0.01
momentum = 0.9
use_bias = True
learning_rate = 0.01


inputs = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], input_shape[2]],
                        "inputs")
policy_label = tf.placeholder(tf.float32, [None, num_classes], "policy_label")
value_label = tf.placeholder(tf.float32, [None], "value_label")
# policy_label = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], input_shape[2]], "policy_label")

net = tf.layers.conv2d(
    inputs=inputs, filters=256, kernel_size=3, strides=1,
    padding='SAME', use_bias=use_bias,
    kernel_initializer=tf.variance_scaling_initializer())

# _BATCH_NORM_DECAY = 0.997
# _BATCH_NORM_EPSILON = 1e-5
# net = tf.layers.batch_normalization(
#     inputs=net, axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
#     scale=True, training=True, fused=True)

net = tf.nn.relu(net)

# net = tf.layers.conv2d(
#     inputs=net, filters=1, kernel_size=1, strides=1,
#     padding='SAME', use_bias=use_bias,
#     kernel_initializer=tf.variance_scaling_initializer())

# net = tf.nn.relu(net)
value_network = tf.layers.conv2d(
    inputs=net, filters=1, kernel_size=1, strides=1,
    padding='SAME', use_bias=use_bias,
    kernel_initializer=tf.variance_scaling_initializer())
value_network = tf.reshape(value_network, [-1, num_classes], name="value_reshape")
value_network = tf.layers.dense(inputs=value_network, units=64, name="value_dense1")

value_network = tf.nn.relu(value_network, name="value_relu")

value_network = tf.layers.dense(inputs=value_network, units=1, name="value_dense2")

value_network = tf.nn.tanh(value_network, name="value_tanh")
# value_net_inputs = tf.reshape(value_net_inputs, [-1], name="value_scalar_reshape")

policy_network = tf.layers.conv2d(
    inputs=net, filters=2, kernel_size=1, strides=1,
    padding='SAME', use_bias=use_bias,
    kernel_initializer=tf.variance_scaling_initializer())
policy_network = tf.reshape(policy_network, [-1, num_classes * 2], name="policy_reshape")
policy_network = tf.layers.dense(inputs=policy_network, units=num_classes, name="policy_dense")
policy_network = tf.nn.softmax(policy_network)

weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
regularizer = 0
for weight in weights:
    regularizer += tf.nn.l2_loss(weight)
regularizer *= weight_decay

value_loss = tf.reduce_mean(tf.square(value_label - tf.reshape(value_network, [-1])))
# value_loss = tf.reduce_mean(tf.square(value_label - value_network))
# policy_loss = -tf.reduce_mean(tf.nn.softmax(self.policy_label) * tf.log(self.policy_network))
# policy_loss = -tf.reduce_mean(tf.transpose(self.policy_label) * tf.log(self.policy_network))
policy_loss = -tf.reduce_mean(tf.reduce_sum(policy_label * tf.log(policy_network), axis=1))
cost = value_loss + policy_loss + regularizer
# cost = value_loss + policy_loss

conf = {"optimizer": "sgd"}
conf['adadelta_rho'] = 0.95
conf['adagrad_initial_accumulator_value'] = 0.1
conf['adam_beta1'] = 0.9
conf['adam_beta2'] = 0.999
conf['opt_epsilon'] = 1.0
conf['ftrl_learning_rate_power'] = -0.5
conf['ftrl_initial_accumulator_value'] = 0.1
conf['ftrl_l1'] = 0.0
conf['ftrl_l2'] = 0.0
conf['rmsprop_momentum'] = 0.9
conf['rmsprop_decay'] = 0.9

optimizer = configure_optimizer(learning_rate, conf)
train_op = optimizer.minimize(cost)

sess.run(tf.global_variables_initializer())

epoch = 23

KING = 1.0
# CAR = 0.9
# PHO = 0.8
# MA = 0.7
# SA = 0.6
# SANG = 0.5
# JJOL = 0.4
CAR = 0.66
PHO = 0.33
MA = 0.
SA = -0.33
SANG = -0.66
JJOL = -1.

state = np.array([[[
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [JJOL, 0, JJOL, 0, JJOL, 0, JJOL, 0, JJOL],
    [0, PHO, 0, 0, 0, 0, 0, PHO, 0],
    [0, 0, 0, 0, KING, 0, 0, 0, 0],
    [CAR, SANG, MA, SA, 0, SA, SANG, MA, CAR],
],[
    [CAR, MA, SANG, SA, 0, SA, MA, SANG, CAR],
    [0, 0, 0, 0, KING, 0, 0, 0, 0],
    [0, PHO, 0, 0, 0, 0, 0, PHO, 0],
    [JJOL, 0, JJOL, 0, JJOL, 0, JJOL, 0, JJOL],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
],[
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
]],
    [[
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [JJOL, 0, JJOL, 0, JJOL, 0, JJOL, 0, JJOL],
        [0, PHO, 0, 0, 0, 0, 0, PHO, 0],
        [0, 0, 0, 0, KING, 0, 0, 0, 0],
        [CAR, SANG, MA, SA, 0, SA, SANG, MA, CAR],
    ], [
        [CAR, MA, SANG, SA, 0, SA, MA, SANG, CAR],
        [0, 0, 0, 0, KING, 0, 0, 0, 0],
        [0, PHO, 0, 0, 0, 0, 0, PHO, 0],
        [JJOL, 0, JJOL, 0, JJOL, 0, JJOL, 0, JJOL],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], [
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
    ]],
])
policy = np.array([0.] * 90)
policy[35] = 0.5
policy[34] = 0.5
policy2 = np.array([0.] * 90)
policy2[62] = 0.5
policy2[61] = 0.5
# state = np.array([np.transpose(state, [1, 2, 0])])
state = np.array(np.transpose(state, [0, 2, 3, 1]))
policy = np.array([policy, policy2])
value = np.array([1., -1.])
# policy = np.array([np.transpose(np.reshape(policy, [1,10,9]), [1,2,0])])

for i in range(epoch):
    print(i)
    _, cost_result, policy_result, value_result, value_loss_result, policy_loss_result = sess.run(
        [train_op, cost, policy_network, value_network, value_loss, policy_loss],
        feed_dict={inputs: state, policy_label: policy,
                   value_label: value})

    print(policy_result)
    # policy_result = np.transpose(policy_result[0], [2, 0, 1])
    # policy_result = np.reshape(policy_result, [90])
    print(policy_result.shape)
    print(policy_result[0][35], policy_result[0][34])
    print(policy_result[1][62], policy_result[1][61])
    print(value_result)
    print(cost_result)
    print(value_loss_result)
    print(policy_loss_result)
