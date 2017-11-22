import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime

image_size = 13
bn_decay = 0.997
bn_epsilon = 1e-5
deconv_layers = 7
deconv_filters = [9, 9, 11, 11, 13, 13, 15]
batch_size = 32
epochs = 100
num_summary_image = 64
log_dir = "deconv_classify_log"
filter_size = 256
strides = 1

x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_")
inputs = tf.reshape(x, [-1, 28, 28, 1], name="reshape_hwc")

net = tf.image.resize_images(inputs, [image_size, image_size])

for i in range(deconv_layers):
    net = tf.layers.conv2d_transpose(net, filter_size, deconv_filters[i], strides=strides, padding='valid',
                                     kernel_initializer=tf.variance_scaling_initializer(), name="deconv" + str(i))

    net = tf.layers.batch_normalization(inputs=net, axis=3, momentum=bn_decay, epsilon=bn_epsilon,
                                        center=True, scale=True, training=True, fused=True, name="batch" + str(i))

    net = tf.nn.relu(net, name="relu" + str(i))
    tf.summary.histogram('activations_%d' % i, net)
gen_image_size = net.get_shape()[1]
gen_y_ = tf.image.resize_images(inputs, [gen_image_size, gen_image_size])
gen_y_ = tf.reshape(gen_y_, [-1, gen_image_size * gen_image_size], name="gen_y_vectorize")
net = tf.layers.conv2d(net, 1, 1, 1, padding='same', kernel_initializer=tf.variance_scaling_initializer,
                       name="last_conv")
gen_x = tf.nn.sigmoid(net, name="gen_sigmoid")
tf.summary.image(tensor=gen_x, max_outputs=num_summary_image, name="gen_x")
gen_x = tf.reshape(gen_x, [-1, gen_image_size * gen_image_size], "gen_x_vectorize")

net = tf.reshape(net, [-1, gen_image_size * gen_image_size], name="net_vectorize")
net = tf.layers.dense(net, 10, name="dense")

class_loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=net, name="class_loss"),
                               name="class_loss_reduce")
gen_loss_op = tf.log(
    tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gen_y_, logits=gen_x, name="gen_loss"),
                   name="gen_loss_reduce"), name="gen_loss_log")
total_loss_op = tf.add(class_loss_op, gen_loss_op, name="total_loss")
tf.summary.scalar('class_loss', class_loss_op)
tf.summary.scalar('gen_loss', gen_loss_op)
tf.summary.scalar('total_loss', total_loss_op)
train_op = tf.train.AdamOptimizer(learning_rate=0.01, name="optimizer").minimize(total_loss_op, name="minimize")
accuracy_op = tf.reduce_mean(
    tf.cast(tf.equal(tf.argmax(net, 1, name="argmax_y"), tf.argmax(y_, 1, name="argmax_y_"), name="equal_y"),
            tf.float32, name="equal_y_cast"), name="accuracy_mean")
tf.summary.scalar('accuracy', accuracy_op)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/test')
sess.run(tf.global_variables_initializer())

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
num_batch = len(mnist.train.images) // batch_size
num_test_batch = len(mnist.test.images) // batch_size
for epoch in range(epochs):
    for train_step in range(num_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, class_loss, gen_loss, total_loss, summary = sess.run(
            [train_op, class_loss_op, gen_loss_op, total_loss_op, merged], feed_dict={x: batch_xs, y_: batch_ys})
        now = datetime.now().strftime('%m/%d %H:%M:%S')
        print("[%s TRAIN %d epoch, %d/%d step] total_loss: %f, class_loss: %f, gen_loss: %f" % (
            now, epoch, train_step, num_batch, total_loss, class_loss, gen_loss))
        train_writer.add_summary(summary, train_step + num_batch * epoch)
    total_accuracy = 0
    for test_step in range(num_test_batch):
        test_xs, test_ys = mnist.test.next_batch(batch_size)
        accuracy, class_loss, gen_loss, total_loss, summary = sess.run(
            [accuracy_op, class_loss_op, gen_loss_op, total_loss_op, merged], feed_dict={x: test_xs, y_: test_ys})
        total_accuracy += accuracy
        now = datetime.now().strftime('%m/%d %H:%M:%S')
        print("[%s TEST %d epoch, %d/%d step] accuracy: %f, total_loss: %f, class_loss: %f, gen_loss: %f" % (
            now, epoch, test_step, num_test_batch, accuracy, total_loss, class_loss, gen_loss))
        test_writer.add_summary(summary, test_step + num_test_batch * epoch)
    print("Total Accuracy : %f" % (total_accuracy / num_test_batch))
