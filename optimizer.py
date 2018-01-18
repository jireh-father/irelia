from core import optimizer2 as optimizer
from util import common
import tensorflow as tf
import os
from core.model_two_policy import Model
from util.dataset2 import Dataset
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

model = Model(sess, weight_decay=FLAGS.weight_decay, momentum=FLAGS.momentum, num_layers=FLAGS.num_model_layers,
              use_cache=FLAGS.use_cache, conf=FLAGS)
writer = tf.summary.FileWriter(FLAGS.save_dir + '/summary', sess.graph)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
FLAGS.dataset_dir = "./"
FLAGS.num_model_layers = 1
FLAGS.batch_size = 32
ds = Dataset(sess)
while True:
    if FLAGS.dataset_dir:
        dataset_dir = FLAGS.dataset_dir
    else:
        dataset_dir = os.path.join(FLAGS.save_dir, "dataset_ready")
    files = glob.glob(os.path.join(dataset_dir, "dataset*.csv"))
    if FLAGS.pending_dataset:
        if len(files) < common.num_opt_games / common.num_selfplay_games:
            log("waiting for dataset... now %d games" % (len(files) * common.num_selfplay_games))
            time.sleep(10)
            continue

    log("load dataset %d files" % len(files))

    if FLAGS.restore_model_path:
        common.restore_model(FLAGS.restore_model_path, None, saver, sess, restore_pending=False)

    for epoch in range(FLAGS.epoch):
        print("epoch %d" % epoch)

        ds.make_dataset(files, FLAGS.batch_size, shuffle_buffer_size=FLAGS.shuffle_buffer_size)
        optimizer.train_model_epoch(model, ds, FLAGS.batch_size, writer)

        ds.close_dataset()
        if (epoch == 0 and common.num_checkpoint_epochs == 1) or (
          epoch > 0 and epoch % common.num_checkpoint_epochs == 0):
            now = common.now_date_str_nums()
            saver.save(sess, os.path.join(FLAGS.save_dir, "new_model_%s.ckpt" % now))
            log("save model")

    if FLAGS.backup_dataset:
        for file in files:
            os.rename(file, os.path.join(FLAGS.save_dir, "dataset_bak", os.path.basename(file)))
            # os.remove(file)
