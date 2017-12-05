from util.common import log
import tensorflow as tf


def train_model(model, ds, train_epoch, batch_size, writer):
    batch_step = 0
    log("train!")
    total_steps = ds.num_samples // batch_size
    if ds.num_samples % batch_size > 0:
        total_steps += 1
    for epoch in range(train_epoch):
        ds.init_dataset()
        while True:
            log("epoch: %d, step: %d/%d " % (epoch, batch_step % total_steps, total_steps))
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


def train_model_epoch(model, ds, batch_size, writer):
    batch_step = 0
    log("train!")
    total_steps = ds.num_samples // batch_size
    if ds.num_samples % batch_size > 0:
        total_steps += 1
    ds.init_dataset()
    while True:
        log("step: %d/%d" % (batch_step, total_steps))
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
