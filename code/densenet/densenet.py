from abc import ABC

import tensorflow as tf


# https://github.com/calmisential/DenseNet_TensorFlow2

# https://github.com/tensorflow/addons/issues/632
def unpool(value, name="unpool"):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = tf.reshape(value, [-1] + sh[-dim:])
        for i in range(dim, 0, -1):
            out = tf.concat([out, tf.zeros_like(out)], i)
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out


def downsample(filters=3, size=3, input_shape=[], apply_batchnorm=True):

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())
    initializer = tf.random_normal_initializer(0.0, 0.02)
    result.add(
        tf.keras.layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())
    result.add(
        tf.keras.layers.Conv2D(
            filters,
            size,
            strides=1,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    shortcut = tf.keras.layers.Conv2D(
        filters,
        size,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        use_bias=False,
    )
    inputs = tf.keras.layers.Input(input_shape)
    x = inputs
    x = result(x)
    y = inputs
    y = shortcut(y)

    return tf.keras.Model(inputs=inputs, outputs=x + y)


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, growth_rate, drop_rate):
        super(BottleNeck, self).__init__()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=4 * growth_rate, kernel_size=(1, 1), strides=1, padding="same"
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            filters=growth_rate, kernel_size=(3, 3), strides=1, padding="same"
        )
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, inputs, training=None, **kwargs):

        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.dropout(x, training=training)
        return x


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_layers, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate
        self.features_list = []

    def _make_layer(self, x, training):
        y = BottleNeck(growth_rate=self.growth_rate, drop_rate=self.drop_rate)(
            x, training=training
        )
        self.features_list.append(y)
        y = tf.concat(self.features_list, axis=-1)
        return y

    def call(self, inputs, training=None, **kwargs):
        self.features_list.append(inputs)
        x = self._make_layer(inputs, training=training)
        for i in range(1, self.num_layers):
            x = self._make_layer(x, training=training)
        self.features_list.clear()
        return x


class TransitionLayerdown(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super(TransitionLayerdown, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.conv = tf.keras.layers.Conv2D(
            filters=out_channels, kernel_size=(1, 1), strides=1, padding="same"
        )
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, inputs, training=None, **kwargs):
        x = self.bn(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        return x


class TransitionLayerup(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super(TransitionLayerup, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.conv = tf.keras.layers.Conv2D(
            filters=out_channels, kernel_size=(3, 3), strides=1, padding="same"
        )

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = tf.nn.relu(x)
        x = unpool(x)
        x = self.conv(x)
        x = tf.nn.relu(x)

        return x


class DenseNet(tf.keras.Model, ABC):
    def __init__(
        self, num_init_features, growth_rate, block_layers, compression_rate, drop_rate
    ):
        super(DenseNet, self).__init__()
        self.num_channels = num_init_features
        self.conv = tf.keras.layers.Conv2D(
            filters=num_init_features, kernel_size=(7, 7), strides=2, padding="same"
        )

        # pooling down
        self.dense_block_1 = DenseBlock(
            num_layers=block_layers[0], growth_rate=growth_rate, drop_rate=drop_rate
        )
        self.num_channels += growth_rate * block_layers[0]
        self.num_channels = compression_rate * self.num_channels
        self.transition_1 = TransitionLayerup(out_channels=int(self.num_channels))
        self.dense_block_2 = DenseBlock(
            num_layers=block_layers[1], growth_rate=growth_rate, drop_rate=drop_rate
        )
        self.num_channels += growth_rate * block_layers[1]
        self.num_channels = compression_rate * self.num_channels
        self.transition_2 = TransitionLayerup(out_channels=int(self.num_channels))
        self.dense_block_3 = DenseBlock(
            num_layers=block_layers[2], growth_rate=growth_rate, drop_rate=drop_rate
        )
        self.num_channels += growth_rate * block_layers[2]
        self.num_channels = compression_rate * self.num_channels
        self.transition_3 = TransitionLayerup(out_channels=int(self.num_channels))
        self.dense_block_4 = DenseBlock(
            num_layers=block_layers[3], growth_rate=growth_rate, drop_rate=drop_rate
        )
        self.transition_4 = TransitionLayerup(out_channels=int(self.num_channels))
        self.dense_block_5 = DenseBlock(
            num_layers=block_layers[1], growth_rate=growth_rate, drop_rate=drop_rate
        )
        self.num_channels += growth_rate * block_layers[1]
        self.num_channels = compression_rate * self.num_channels
        self.transition_5 = TransitionLayerup(out_channels=int(self.num_channels))
        self.dense_block_6 = DenseBlock(
            num_layers=block_layers[2], growth_rate=growth_rate, drop_rate=drop_rate
        )
        self.num_channels += growth_rate * block_layers[2]
        self.num_channels = compression_rate * self.num_channels
        self.transition_6 = TransitionLayerup(out_channels=int(self.num_channels))

        # pooling up
        self.dense_block_7 = DenseBlock(
            num_layers=block_layers[0], growth_rate=growth_rate, drop_rate=drop_rate
        )
        self.num_channels += growth_rate * block_layers[0]
        self.num_channels = compression_rate * self.num_channels
        self.transition_7 = TransitionLayerup(out_channels=int(self.num_channels))
        self.dense_block_8 = DenseBlock(
            num_layers=block_layers[1], growth_rate=growth_rate, drop_rate=drop_rate
        )
        self.num_channels += growth_rate * block_layers[1]
        self.num_channels = compression_rate * self.num_channels
        self.transition_8 = TransitionLayerup(out_channels=int(self.num_channels))
        self.dense_block_9 = DenseBlock(
            num_layers=block_layers[2], growth_rate=growth_rate, drop_rate=drop_rate
        )
        self.num_channels += growth_rate * block_layers[2]
        self.num_channels = compression_rate * self.num_channels
        self.transition_9 = TransitionLayerup(out_channels=int(self.num_channels))
        self.dense_block_10 = DenseBlock(
            num_layers=block_layers[3], growth_rate=growth_rate, drop_rate=drop_rate
        )
        self.num_channels += growth_rate * block_layers[0]
        self.num_channels = compression_rate * self.num_channels
        self.transition_10 = TransitionLayerup(out_channels=int(self.num_channels))
        self.dense_block_11 = DenseBlock(
            num_layers=block_layers[1], growth_rate=growth_rate, drop_rate=drop_rate
        )
        self.num_channels += growth_rate * block_layers[1]
        self.num_channels = compression_rate * self.num_channels
        self.transition_11 = TransitionLayerup(out_channels=int(self.num_channels))
        self.dense_block_12 = DenseBlock(
            num_layers=block_layers[2], growth_rate=growth_rate, drop_rate=drop_rate
        )
        self.num_channels += growth_rate * block_layers[2]
        self.num_channels = compression_rate * self.num_channels
        self.transition_12 = TransitionLayerup(out_channels=int(self.num_channels))
        self.dense_block_13 = DenseBlock(
            num_layers=block_layers[2], growth_rate=growth_rate, drop_rate=drop_rate
        )

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="valid")

        x = self.dense_block_1(x, training=training)
        x = self.transition_1(x, training=training)
        x = self.dense_block_2(x, training=training)
        x = self.transition_2(x, training=training)
        x = self.dense_block_3(x, training=training)
        x = self.transition_3(x, training=training)
        x = self.dense_block_4(x, training=training)
        x = self.transition_4(x, training=training)
        x = self.dense_block_5(x, training=training)
        x = self.transition_5(x, training=training)
        x = self.dense_block_6(x, training=training)
        x = self.transition_6(x, training=training)
        x = self.dense_block_7(x, training=training)
        x = self.transition_7(x, training=training)
        x = self.dense_block_8(x, training=training)
        x = self.transition_8(x, training=training)
        x = self.dense_block_9(x, training=training)
        x = self.transition_9(x, training=training)
        x = self.dense_block_10(x, training=training)
        x = self.transition_10(x, training=training)
        x = self.dense_block_11(x, training=training)
        x = self.transition_11(x, training=training)
        x = self.dense_block_12(x, training=training)
        x = self.transition_12(x, training=training)
        x = self.dense_block_13(x, training=training)

        x = self.conv(x)
        x = unpool(x)
        x = tf.nn.relu(x)

        return x


def densenet_121():
    return DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_layers=[6, 12, 24, 16],
        compression_rate=0.5,
        drop_rate=0.5,
    )


def densenet_169():
    return DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_layers=[6, 12, 32, 32],
        compression_rate=0.5,
        drop_rate=0.5,
    )


def densenet_201():
    return DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_layers=[6, 12, 48, 32],
        compression_rate=0.5,
        drop_rate=0.5,
    )


def densenet_264():
    return DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_layers=[6, 12, 64, 48],
        compression_rate=0.5,
        drop_rate=0.5,
    )
