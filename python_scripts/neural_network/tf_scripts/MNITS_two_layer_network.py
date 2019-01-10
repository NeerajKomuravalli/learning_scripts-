import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def init_weights(shape):
    initial = tf.truncated_normal(dtype=tf.float32,
                               shape=shape,
                               stddev=0.1)
    return tf.Variable(initial)


def init_bias(shape):
    initial = tf.constant(0.1,
                       shape=shape)
    return tf.Variable(initial)


def max_pool_2_2(x):
    return tf.nn.max_pool(x,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')


def conv2d(x, W):
    return tf.nn.conv2d(x,
                        W,
                        strides=[1, 1, 1, 1],
                        padding='SAME')


def batch_norm_wrapper(inputs, is_training, decay=0.999):

    epsilon = 0.00000001
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    mean = tf.cond(is_training, lambda: tf.nn.moments(inputs, [0])[0], lambda: tf.ones(inputs.get_shape()[-1])*pop_mean)
    var = tf.cond(is_training, lambda: tf.nn.moments(inputs, [0])[1], lambda: tf.ones(inputs.get_shape()[-1])*pop_var)
    train_mean = tf.cond(is_training, lambda: tf.assign(pop_mean, pop_mean*decay+mean*(1-decay)), lambda: tf.zeros(1))
    train_var = tf.cond(is_training, lambda: tf.assign(pop_var, pop_var*decay+var*(1-decay)), lambda: tf.zeros(1))

    with tf.control_dependencies([train_mean, train_var]):
        return tf.nn.batch_normalization(inputs, mean, var, beta, scale, epsilon)


# Train data declaration
X = tf.placeholder(dtype=tf.float32,
                   shape=[None, 784],
                   name='input')
Y = tf.placeholder(dtype=tf.float32,
                   shape=[None, 10],
                   name='output')

training = tf.placeholder_with_default(True, shape=())

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Reshaping the X into the form accepted by the network
X_image = tf.reshape(X, [-1, 28, 28, 1])

# Declaring layer 1
W_conv_layer1 = init_weights([5, 5, 1, 32])
b_conv_layer1 = init_bias([32])

conv_out_1 = conv2d(X_image, W_conv_layer1) + b_conv_layer1
# norm_value_1 = batch_norm_wrapper(conv_out_1, training)
# norm_value_1 += b_conv_layer1
relu_layer1 = tf.nn.relu(conv_out_1)
max_pool_layer1 = max_pool_2_2(relu_layer1)

# Declaring layer 2
W_conv_layer2 = init_weights([5, 5, 32, 64])
b_conv_layer2 = init_bias([64])

conv_out_2 = conv2d(max_pool_layer1, W_conv_layer2) + b_conv_layer2
# norm_value_2 = batch_norm_wrapper(conv_out_2, training)
# norm_value_2 += b_conv_layer2
relu_layer2 = tf.nn.relu(conv_out_2)
max_pool_layer2 = max_pool_2_2(relu_layer2)


x_flatten = tf.reshape(max_pool_layer2, [-1, 7 * 7 * 64])

# Declaring fully connected layer
full_connected1 = init_weights([7 * 7 * 64, 1024])
b_fully_connected1 = init_bias([1024])

fc1_out = tf.matmul(x_flatten, full_connected1)
norm_fc_1 = batch_norm_wrapper(fc1_out, training)
norm_fc_1 += b_fully_connected1
relu_fc1 = tf.nn.relu(norm_fc_1)

# Adding dropout layer
keep_prob = tf.placeholder(tf.float32)
dropout_layer_out = tf.nn.dropout(relu_fc1, keep_prob)

full_connected2 = init_weights([1024, 10])
b_fully_connected2 = init_bias([10])

fc2_out = tf.matmul(dropout_layer_out, full_connected2)
norm_fc_2 = batch_norm_wrapper(fc2_out, training)
norm_fc_2 += b_fully_connected2
relu_fc2 = tf.nn.relu(norm_fc_2)

softmax_out = tf.nn.softmax(relu_fc2)

loss = tf.nn.softmax_cross_entropy_with_logits(logits=softmax_out, labels=Y)
train = tf.train.AdamOptimizer(0.0001).minimize(loss)

correct_preds = tf.equal(tf.argmax(Y, 1), tf.argmax(softmax_out, 1))
accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for i in range(1000):
        print("Running iteration : ", i)
        batch = mnist.train.next_batch(50)
        sess.run(train, feed_dict={X: batch[0], Y: batch[1], keep_prob: 0.5})

    results = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})

    print("Test accuracy : ", results)





