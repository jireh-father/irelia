from util.dataset import Dataset
import tensorflow as tf
from util import common
import numpy as np
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset_path', None, "dataset_dir")

common.set_flags()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
ds = Dataset(sess)
FLAGS.dataset_path = "checkpoint/dataset_20171209190123_209a6fbd-34b2-4e3a-9039-a8341bda9755.csv"
ds.make_dataset([FLAGS.dataset_path], FLAGS.batch_size)

ds.init_dataset()
while True:
    try:
        train_batch_state, train_batch_policy, train_batch_value = ds.batch()
        print(train_batch_state.shape)
        print(train_batch_policy.shape)
        print(train_batch_value.shape)
        for i in range(len(train_batch_state)):
            state_data = train_batch_state[i]
            state_data = np.transpose(state_data, [2,0,1])
            print(state_data.shape)
            print(state_data[:8][0])
            print((state_data[:8][0] == 0).all())
            print(state_data[8:16].shape)

            color = 'b' if state_data[16][0] == 1 else 'r'
            print(color)
            break
        break
    except tf.errors.OutOfRangeError:
        break
