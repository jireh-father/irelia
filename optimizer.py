from core import optimizer
from util import common
import tensorflow as tf
import os
from core.model import Model
from util.dataset import Dataset
import glob
import time
from util.common import log

FLAGS = tf.app.flags.FLAGS

common.set_flags()
tf.app.flags.DEFINE_boolean('pending_dataset', False, "pending dataset")
common.make_dirs(os.path.join(FLAGS.save_dir, "dataset_ready"))
common.make_dirs(os.path.join(FLAGS.save_dir, "dataset_bak"))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
writer = tf.summary.FileWriter(FLAGS.save_dir + '/summary', sess.graph)
model = Model(sess, weight_decay=FLAGS.weight_decay, momentum=FLAGS.momentum, num_layers=FLAGS.num_model_layers,
              use_cache=FLAGS.use_cache)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

ds = Dataset(sess)

while True:
    files = glob.glob(os.path.join(FLAGS.save_dir, "dataset_ready", "dataset*.csv"))
    if FLAGS.pending_dataset:
        if len(files) < common.num_opt_games / common.num_selfplay_games:
            log("waiting for dataset... now %d games" % (len(files) * common.num_selfplay_games))
            time.sleep(10)
            continue
    log("load dataset %d files" % len(files))
    learning_rate = FLAGS.learning_rate
    ds.make_dataset(files, FLAGS.batch_size)

    # common.restore_model(FLAGS.save_dir, "best_model.ckpt", saver, sess, restore_pending=True)

    for epoch in range(FLAGS.epoch):
        optimizer.train_model_epoch(model, learning_rate, ds, writer)

        if epoch > 0 and epoch % FLAGS.learning_rate_decay_interval == 0:
            log("decay learning rate")
            learning_rate = learning_rate * FLAGS.learning_rate_decay

        if epoch > 0 and epoch % common.num_checkpoint_epochs == 0:
            now = common.now_date_str_nums()
            saver.save(sess, os.path.join(FLAGS.save_dir, "new_model_%s.ckpt" % now))
            log("save model")
    ds.close_dataset()
    for file in files:
        os.rename(file, os.path.join(FLAGS.save_dir, "dataset_bak", os.path.basename(file)))
        # os.remove(file)
