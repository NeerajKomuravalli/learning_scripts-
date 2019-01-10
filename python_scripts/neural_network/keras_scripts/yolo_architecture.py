from keras.models import Sequential
from keras import backend as K
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D, Dense, Flatten


class yolo_model:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model.add(Conv2D(64, (7, 7), padding="same", input_shape=inputShape, strides=(2, 2)))
        model.add(MaxPooling2D((2, 2), padding="same", input_shape=inputShape, strides=(2, 2)))

        model.add(Conv2D(192, (3, 3), padding="same"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

        model.add(Conv2D(128, (1, 1), padding='same'))
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Conv2D(256, (1, 1), padding='same'))
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same', strides=(2, 2)))

        model.add(Conv2D(256, (1, 1), padding='same'))
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Conv2D(256, (1, 1), padding='same'))
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Conv2D(256, (1, 1), padding='same'))
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Conv2D(256, (1, 1), padding='same'))
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Conv2D(512, (1, 1), padding='same'))
        model.add(Conv2D(1024, (3, 3), padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same', strides=(2, 2)))

        model.add(Conv2D(512, (1, 1), padding='same'))
        model.add(Conv2D(1024, (3, 3), padding='same'))
        model.add(Conv2D(512, (1, 1), padding='same'))
        model.add(Conv2D(1024, (3, 3), padding='same'))
        model.add(Conv2D(1024, (3, 3), padding='same'))
        model.add(Conv2D(1024, (3, 3), padding='same', strides=(2, 2)))

        model.add(Conv2D(1024, (3, 3), padding='same'))
        model.add(Conv2D(1024, (3, 3), padding='same'))

        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Conv2D(4096, (1, 1), padding='same'))
        model.add(Dense(30))

        return model