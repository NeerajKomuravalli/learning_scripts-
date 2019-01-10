import tensorflow as tf
import numpy as np


weights = tf.Variable(tf.random_normal(shape=[10, 2],
                                       mean=0,
                                       stddev=1,
                                       dtype=tf.float32))
bias = tf.Variable(tf.random_normal(shape=[10, 1],
                                    mean=0,
                                    stddev=1,
                                    dtype=tf.float32))

input = tf.placeholder(dtype=tf.float32,
                       shape=[None, 10],
                       name='input')

input_signal = np.random.rand(1024, 10)

with tf.Session() as sess:
    relu = tf.nn.relu(input,
               name='activation1')
    # Using eval method
    print(relu.eval(feed_dict={input: input_signal}))
    # Using sess.run method
    print(sess.run(relu, feed_dict={input: input_signal}))