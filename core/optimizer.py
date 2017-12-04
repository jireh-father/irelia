from util.common import log
import tensorflow as tf


def train_model(model, learning_rate, ds, train_epoch, learning_rate_decay, learning_rate_decay_interval, writer):
    batch_step = 0
    log("train!")
    for epoch in range(train_epoch):
        ds.init_dataset()
        while True:
            log("epoch: %d, step: %d" % (epoch, batch_step))
            try:
                train_batch_state, train_batch_policy, train_batch_value = ds.batch()
                _, train_cost, summary = model.train(train_batch_state, train_batch_policy, train_batch_value,
                                                     learning_rate)
                log("trained! cost: %f" % train_cost)
                if batch_step != 0 and batch_step % 20 == 0:
                    writer.add_summary(summary, batch_step)
                if batch_step > 0 and batch_step % learning_rate_decay_interval == 0:
                    log("decay learning rate")
                    learning_rate = learning_rate * learning_rate_decay
                batch_step += 1
            except tf.errors.OutOfRangeError:
                log("out of range dataset! init!!")
                break


def train_model_epoch(model, ds, writer):
    batch_step = 0
    log("train!")
    ds.init_dataset()
    while True:
        log("step: %d" % batch_step)
        try:
            train_batch_state, train_batch_policy, train_batch_value = ds.batch()
            _, train_cost, summary = model.train(train_batch_state, train_batch_policy, train_batch_value,
                                                 ds.num_samples)
            log("trained! cost: %f" % train_cost)
            if batch_step != 0 and batch_step % 20 == 0:
                writer.add_summary(summary, batch_step)
            batch_step += 1
        except tf.errors.OutOfRangeError:
            log("out of range dataset! init!!")
            break
