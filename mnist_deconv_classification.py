import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

image_size = 13
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
deconv_layers = 7
deconv_filters = [9, 9, 11, 11, 13, 13, 15]

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_")
inputs = tf.reshape(x, [-1, 28, 28, 1], name="reshape_hwc")

gen_image_y = tf.image.resize_images(inputs, [87, 87])
gen_image_y = tf.reshape(gen_image_y, [-1, 87 * 87])
inputs = tf.image.resize_images(inputs, [image_size, image_size])

for i in range(deconv_layers):
    inputs = tf.layers.conv2d_transpose(inputs, 256, deconv_filters[i], strides=1, padding='valid',
                                        kernel_initializer=tf.variance_scaling_initializer(), name="conv" + str(i))

    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=True, fused=True, name="batch" + str(i))

    inputs = tf.nn.relu(inputs, name="relu" + str(i))
inputs = tf.layers.conv2d(inputs, 1, 1, 1, padding='same', kernel_initializer=tf.variance_scaling_initializer)
gen_image = tf.nn.sigmoid(inputs, name="gen_sigmoid")
gen_image = tf.reshape(gen_image, [-1, 87 * 87], "gen_image_reshape")
inputs = tf.reshape(inputs, [-1, 87 * 87], name="reshape_fc")
inputs = tf.layers.dense(inputs, 10, name="dense")

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=inputs))
gen_image_loss = tf.log(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gen_image_y, logits=gen_image)))
total_loss = loss + gen_image_loss
train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(total_loss)
correct_prediction = tf.equal(tf.argmax(inputs, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(64)
    _, loss_result, gen_loss_result, total_loss_result = sess.run([train_op, loss, gen_image_loss, total_loss], feed_dict={x: batch_xs, y_: batch_ys})
    print(i, loss_result, gen_loss_result, total_loss_result)
    if i != 0 and i % 20 == 0:
        for j in range(10):
            test_xs, test_ys = mnist.test.next_batch(64)
            print(sess.run([accuracy, loss, gen_image_loss, total_loss], feed_dict={x: test_xs, y_: test_ys}))
