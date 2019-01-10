# https://cs230-stanford.github.io/tensorflow-input-data.html
import tensorflow as tf


if __name__ == '__main__':
    dataset = tf.data.TextLineDataset("file.txt")

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        for i in range(3):
            print(sess.run(next_element))