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

input_shape = [10, 9, 1]
num_layers = 1
num_classes = 10 * 9
weight_decay = 0.01
momentum = 0.9
use_bias = False
learning_rate = 0.1
epoch = 100

inputs = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], input_shape[2]],
                        "inputs")
policy_label = tf.placeholder(tf.float32, [None, num_classes], "policy_label")

net = tf.layers.conv2d(
    inputs=inputs, filters=256, kernel_size=3, strides=1,
    padding='SAME', use_bias=use_bias,
    kernel_initializer=tf.variance_scaling_initializer())

net = tf.layers.conv2d(
    inputs=net, filters=1, kernel_size=1, strides=1,
    padding='SAME', use_bias=use_bias,
    kernel_initializer=tf.variance_scaling_initializer())

policy_network = tf.reshape(net, [-1, num_classes * 2], )
policy_network = tf.layers.dense(inputs=policy_network, units=num_classes)

cost = -tf.reduce_mean(tf.reduce_sum(policy_label * tf.log(policy_network), axis=1))
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

CAR = 0.9
SA = 0.6
KING = 1.
PHO = 0.8
JJOL = 0.4
MA = 0.7
SANG = 0.5
state = np.array([[
    [CAR, MA, SANG, SA, 0, SA, MA, SANG, CAR],
    [0, 0, 0, 0, KING, 0, 0, 0, 0],
    [0, PHO, 0, 0, 0, 0, 0, PHO, 0],
    [JJOL, 0, JJOL, 0, JJOL, 0, JJOL, 0, JJOL],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [JJOL, 0, JJOL, 0, JJOL, 0, JJOL, 0, JJOL],
    [0, PHO, 0, 0, 0, 0, 0, PHO, 0],
    [0, 0, 0, 0, KING, 0, 0, 0, 0],
    [CAR, SANG, MA, SA, 0, SA, SANG, MA, CAR],
]])
policy = np.array([0.] * 90)
policy[35] = 0.5
policy[34] = 0.5
print(state)
print(policy)
sys.eixt()

for i in range(epoch):
    _, cost_result, policy_result = sess.run([train_op, cost, policy_network],
                                             feed_dict={inputs: state, policy_label: policy})
    print(cost_result)
    print(policy_result)