import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# weights = tf.Variable(tf.random_normal(shape=[784, 10],
#                                        mean=0,
#                                        stddev=1,
#                                        dtype=tf.float32))
# bias = tf.Variable(tf.random_normal(shape=[10],
#                                     mean=0,
#                                     stddev=1,
#                                     dtype=tf.float32))

weights = tf.Variable(np.zeros([784, 10]), dtype=tf.float32)
bias = tf.Variable(np.zeros([10]), dtype=tf.float32)

X = tf.placeholder(dtype=tf.float32,
                   shape=[None, 784],
                   name='input')
Y = tf.placeholder(dtype=tf.float32,
                   shape=[None, 10],
                   name='label')

layer_1_matmul = tf.matmul(X, weights, name='matmul-1')
layer_1_add_bias = tf.add(layer_1_matmul, bias, name='add_bias-1')

softmax_out = tf.nn.softmax(layer_1_add_bias)

learning_rate = 0.01
loss = tf.nn.softmax_cross_entropy_with_logits(logits=softmax_out, labels=Y)
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for i in range(1000):
        batch = mnist.train.next_batch(50)
        sess.run(train, feed_dict={X: batch[0], Y: batch[1]})

    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(softmax_out, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    results = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})

    print("Test accuracy : ", results)