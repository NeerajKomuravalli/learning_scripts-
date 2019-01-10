import tensorflow as tf

p = tf.placeholder(tf.float32, shape=[None, 105])
logit_q = tf.placeholder(tf.float32, shape=[None, 105])
q = tf.nn.sigmoid(logit_q)

feed_dict = {
    p: [[1, 0, 0, 0, 0] + [0] * 100, [0, 0, 1, 1, 0] + [0] * 100],
    logit_q: [[0.6, 0, 0, 0, 1] + [0] * 100, [0, 0, 0.5, 0, 1] + [0] * 100]
}

with tf.Session() as sess:
    print("Sigmoid entropy")
    sigmoid_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=p, logits=logit_q)
    mean_of_sigmoid_entropy = tf.reduce_mean(sigmoid_entropy)

    print(sigmoid_entropy.eval(feed_dict))
    print(mean_of_sigmoid_entropy.eval(feed_dict))

    print("Softmax entropy")
    softmax_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=p, logits=logit_q)
    p = tf.stop_gradient(p)
    mean_of_softmax_entropy = tf.reduce_mean(softmax_entropy)

    print(softmax_entropy.eval(feed_dict))
    print(mean_of_softmax_entropy.eval(feed_dict))

    print("Custom made loss")
    custom_loss = tf.reduce_mean(tf.matmul(p, tf.transpose(-1 * tf.log(tf.nn.sigmoid(logit_q)) + 1e-9)) +
                                 tf.matmul(1 - p, tf.transpose(-1 * tf.log(1 - tf.nn.sigmoid(logit_q) + 1e-9))))
    print(custom_loss.eval(feed_dict))
