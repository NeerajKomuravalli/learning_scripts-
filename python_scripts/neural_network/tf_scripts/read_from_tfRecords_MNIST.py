import tensorflow as tf
from tensorflow.examples.tutorials.mnist import mnist


def decode(serialized_example):
    """Parses an image and label from the given `serialized_example`."""
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    # mnist.IMAGE_PIXELS
    image.set_shape((mnist.IMAGE_PIXELS))
    # We can reshape it to whatever shape we want
    image = tf.reshape(image, (28, 28, 1))
    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return image, label


def augment(image, label):
    """Placeholder for data augmentation."""
    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.
    return image, label


def resize(image, label):
    """resize the image data because we want the original data to be intact"""
    IMAGE_WIDTH = 128
    IMAGE_HEIGHT = 128
    image = tf.image.resize_images(image, [IMAGE_WIDTH, IMAGE_HEIGHT])
    return image, label

def normalize(image, label):
    """Convert `image` from [0, 255] -> [-0.5, 0.5] floats."""
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return image, label


batch_size = 32
num_epochs = 100
dataset = tf.data.TFRecordDataset('./train.tfrecords')
# dataset = tf.data.TFRecordDataset.range(10)
dataset = dataset.map(decode)
dataset = dataset.map(augment)
# dataset = dataset.map(resize)
dataset = dataset.map(normalize)
dataset = dataset.shuffle(1000 + 3 * batch_size)
dataset = dataset.repeat(num_epochs)
dataset = dataset.batch(batch_size)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

# # Testing
# dataset = tf.data.Dataset.range(10)
#
# iterator = dataset.make_one_shot_iterator()
# next_element = iterator.get_next()
#
# with tf.Session() as sess:
#     for i in range(10):
#         add = tf.multiply(next_element, next_element)
#         value = sess.run(add)
#         print(value)