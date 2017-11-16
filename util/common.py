import os
import shutil
import datetime
import time
from util import dataset
import tensorflow as tf


def save_model(sess, saver, checkpoint_path):
    print("save model")
    model_file_name = os.path.basename(checkpoint_path)
    save_dir = os.path.dirname(checkpoint_path)
    if os.path.exists(checkpoint_path + ".index"):
        print("backup model")
        dt = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
        bak_dir = os.path.join(save_dir, "model_" + dt)
        os.makedirs(bak_dir)
        shutil.move(checkpoint_path + ".index", os.path.join(bak_dir, model_file_name + ".index"))
        shutil.move(checkpoint_path + ".data-00000-of-00001",
                    os.path.join(bak_dir, model_file_name + ".data-00000-of-00001"))
        shutil.move(checkpoint_path + ".meta", os.path.join(bak_dir, model_file_name + ".meta"))
    saver.save(sess, checkpoint_path)


def eval_mode(sess, model, test_dataset):
    if test_dataset is not None:
        while (True):
            try:
                test_batch_state, test_batch_policy, test_batch_value = dataset.get_batch(sess,
                                                                                          test_dataset)
                cost = model.eval(test_batch_state, test_batch_policy, test_batch_value)
                print("eval cost", cost)
                break
            except tf.errors.OutOfRangeError:
                dataset.initializer(sess, test_dataset)
