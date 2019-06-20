import tensorflow as tf
from tensorflow.keras.activations import linear, relu, softmax
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, Input, MaxPool2D, ReLU,
                                     Reshape, Softmax)
from tensorflow.keras.models import Model


def cnn_trad_fpool3():

    INPUT_SHAPE = (36, 40, 1)

    inputs = Input(shape=INPUT_SHAPE, dtype=tf.float32)
    x = Conv2D(filters=64,
               kernel_size=(24, 10),
               strides=(1, 1),
               padding="SAME",
               name="conv_1a")(inputs)
    x = ReLU(name="relu_1a")(x)
    x = BatchNormalization(name="bnorm_1a")(x)
    x = Dropout(rate=0.2, name="dropout_1a")(x)
    x = MaxPool2D(pool_size=(1, 3), strides=(1, 3), name="mpool_1a")(x)

    x = Conv2D(kernel_size=(12, 5),
               filters=64,
               padding="SAME",
               strides=(1, 1),
               name="conv_2a")(x)
    x = ReLU(name="relu_2a")(x)
    x = BatchNormalization(name="bnorm_2a")(x)
    x = Dropout(rate=0.2, name="dropout_2a")(x)
    x = Flatten(name="flatten_2a")(x)

    x = Dense(units=32, activation=linear, name="low_rank_3a")(x)
    x = BatchNormalization(name="bnorm_3a")(x)
    x = Dropout(rate=0.2, name="dropout_3a")(x)

    x = Dense(units=128, activation=relu, name="non_linear_4a")(x)
    x = BatchNormalization(name="bnorm_4a")(x)
    x = Dropout(rate=0.2, name="dropout_4a")(x)

    x = Dense(units=128, activation=relu, name="non_linear_5a")(x)
    x = BatchNormalization(name="bnorm_5a")(x)
    x = Dropout(rate=0.2, name="dropout_5a")(x)

    x = Dense(units=128, activation=relu, name="non_linear_6a")(x)
    x = BatchNormalization(name="bnorm_6a")(x)

    x = Dense(units=10, name="output_map_7a")(x)
    x = Softmax(name="softmax_7a")(x)

    return Model(inputs=inputs, outputs=x, name="cnn-trad-fpool3")
