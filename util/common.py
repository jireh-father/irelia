import os
import shutil
import datetime
import time
import tensorflow as tf


def save_model(sess, saver, checkpoint_path):
    print("save model")
    model_file_name = os.path.basename(checkpoint_path)
    save_dir = os.path.dirname(checkpoint_path)
    if os.path.exists(checkpoint_path + ".index"):
        print("backup model")
        dt = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
        bak_dir = os.path.join(save_dir, "model_" + dt)
        if not os.path.exists(bak_dir):
            os.makedirs(bak_dir)
        shutil.move(checkpoint_path + ".index", os.path.join(bak_dir, model_file_name + ".index"))
        shutil.move(checkpoint_path + ".data-00000-of-00001",
                    os.path.join(bak_dir, model_file_name + ".data-00000-of-00001"))
        shutil.move(checkpoint_path + ".meta", os.path.join(bak_dir, model_file_name + ".meta"))
    saver.save(sess, checkpoint_path)


def train_model(i_episode, ds, model, learning_rate, sess, saver, checkpoint_path, flags):
    if i_episode > 0 and i_episode % flags.episode_interval_to_train == 0 and ds.has_train_dataset_file():
        ds.open_dataset(flags.batch_size)
        batch_step = 0
        log("train!")
        for epoch in range(flags.epoch):
            while True:
                log("epoch: %d, step: %d, episode: %d" % (epoch, batch_step, i_episode))
                try:
                    train_batch_state, train_batch_policy, train_batch_value = ds.get_train_batch()
                    _, train_cost = model.train(train_batch_state, train_batch_policy, train_batch_value,
                                                learning_rate)
                    log("trained! cost:", train_cost)

                    if batch_step > 0 and batch_step % flags.learning_rate_decay_interval == 0:
                        log("decay learning rate")
                        learning_rate = learning_rate * flags.learning_rate_decay
                    batch_step += 1
                except tf.errors.OutOfRangeError:
                    log("out of range dataset! init!!")
                    ds.init_train()
                    break
        # save model
        eval_model(model, ds)
        save_model(sess, saver, checkpoint_path)
        # todo : evaluate best player
    return learning_rate


def eval_model(model, ds):
    if ds.test_dataset is not None:
        while True:
            try:
                test_batch_state, test_batch_policy, test_batch_value = ds.get_test_batch()
                cost = model.eval(test_batch_state, test_batch_policy, test_batch_value)
                print("eval cost", cost)
            except tf.errors.OutOfRangeError:
                print("all evaluation finished")
                break


def restore_model(save_dir, model_file_name, saver, sess):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint_path = os.path.join(save_dir, model_file_name)
    if os.path.exists(checkpoint_path + ".index"):
        print("restore success!!")
        saver.restore(sess, checkpoint_path)
    return checkpoint_path


def log(msg):
    dt = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
    print("[%s] %s" % (dt, msg))
    pass


def set_flags():
    tf.app.flags.DEFINE_string('save_dir', os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoint"),
                               "save dir")

    tf.app.flags.DEFINE_string('model_file_name', "model.ckpt", "model name to save")
    tf.app.flags.DEFINE_integer('max_step', 200, "max step in a episode")
    tf.app.flags.DEFINE_integer('max_episode', 1000000, "max episode")
    tf.app.flags.DEFINE_integer('max_simulation', 5, "max simulation count in a mcts search")
    tf.app.flags.DEFINE_integer('exploration_step', 20, "exploration step")
    tf.app.flags.DEFINE_integer('episode_interval_to_train', 2, "episode interval to train model")
    tf.app.flags.DEFINE_integer('epoch', 20, "epoch")
    tf.app.flags.DEFINE_integer('num_model_layers', 20, "numbers of model layers")
    tf.app.flags.DEFINE_float('weight_decay', 0.0001, "weigh decay for weights l2 regularize")
    tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate")
    tf.app.flags.DEFINE_float('learning_rate_decay', 0.1, "learning rate decay")
    tf.app.flags.DEFINE_integer('learning_rate_decay_interval', 1000, "learning rate decay interval")
    tf.app.flags.DEFINE_integer('batch_size', 64, "batch size")
    tf.app.flags.DEFINE_float('c_puct', 0.5, "a constant determining the level of exploration")
    tf.app.flags.DEFINE_float('train_fraction', 0.7, "train dataset fraction")
    tf.app.flags.DEFINE_float('momentum', 0.9, "momentum for optimizer")
    tf.app.flags.DEFINE_boolean('print_mcts_history', False, "show mcts search history")
    tf.app.flags.DEFINE_boolean('use_color_print', False, "use color in printing state")
