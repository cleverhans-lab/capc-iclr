import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Activation,
    AveragePooling2D,
    Flatten,
    Convolution2D,
    MaxPooling2D,
    Reshape,
)


def cryptonets_relu_model(input):
    y = Conv2D(
        filters=5,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding="same",
        use_bias=True,
        input_shape=(28, 28, 1),
        activation='relu',
        name="conv2d_1",
    )(input)
    y = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(y)
    y = Conv2D(
        filters=50,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding="same",
        use_bias=True,
        name="conv2d_2",
    )(y)

    y = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(y)
    y = Flatten()(y)
    y = Dense(100, use_bias=True, activation="relu", name="fc_1")(y)
    y = Dense(10, use_bias=True, name="fc_2")(y)

    return y


def cryptonets_relu_model_squashed(input, conv1_weights, squashed_weights,
                                   fc2_weights):

    print("conv1_weights", conv1_weights[0].shape, conv1_weights[1].shape)
    print("squashed_weights", squashed_weights[0].shape,
          squashed_weights[1].shape)
    print("fc2_weights", fc2_weights[0].shape, fc2_weights[1].shape)

    y = Conv2D(
        filters=5,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding="same",
        use_bias=True,
        kernel_initializer=tf.compat.v1.constant_initializer(conv1_weights[0]),
        bias_initializer=tf.compat.v1.constant_initializer(conv1_weights[1]),
        input_shape=(28, 28, 1),
        activation="relu",
        name="convd1_1",
    )(input)

    # Using Keras model API with Flatten results in split ngraph at Flatten() or Reshape() op.
    # Use tf.reshape instead
    y = tf.reshape(y, [-1, 5 * 14 * 14])

    # Flatten() results in split Keras graph
    y = Dense(
        100,
        use_bias=True,
        activation="relu",
        name="squash_fc_1",
        kernel_initializer=tf.compat.v1.constant_initializer(
            squashed_weights[0]),
        bias_initializer=tf.compat.v1.constant_initializer(squashed_weights[1]),
    )(y)
    y = Dense(
        10,
        use_bias=True,
        kernel_initializer=tf.compat.v1.constant_initializer(fc2_weights[0]),
        bias_initializer=tf.compat.v1.constant_initializer(fc2_weights[1]),
        name="output",
    )(y)

    return y