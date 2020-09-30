
# Global average pooling replaced with Conv2D
# Unpooling replaced with ConvTranspose
import tensorflow as tf

# import, data augmentation code from https://www.tensorflow.org/tutorials/generative/pix2pix
import os
import time
from matplotlib import pyplot as plt
from IPython import display

_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'

path_to_zip = tf.keras.utils.get_file('facades.tar.gz',
                                      origin=_URL,
                                      extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
N = 256  # size of training images 64*64
K = 32  # number of filters at convolutional layer
DROP_OUT = 0.3


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


inp, re = load(PATH + 'train/100.jpg')


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


# normalizing the images to [-1, 1]

def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                     IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


train_dataset = tf.data.Dataset.list_files(PATH + 'train/*.jpg')
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(PATH + 'test/*.jpg')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

OUTPUT_CHANNELS = 3

def downsample(filters=3, size=3, input_shape=[], apply_batchnorm=True):

  result = tf.keras.Sequential()
  result.add(tf.keras.layers.BatchNormalization())
  result.add(tf.keras.layers.ReLU())
  initializer = tf.random_normal_initializer(0., 0.02)
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())
  result.add(tf.keras.layers.ReLU())
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  shortcut=tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False)
  inputs = tf.keras.layers.Input(input_shape)
  x = inputs
  x = result(x)
  y = inputs
  y = shortcut(y)

  return tf.keras.Model(inputs=inputs, outputs=x+y)






# https://github.com/tensorflow/addons/issues/632
def unpooling(input_shape=[]):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    inputs = tf.keras.layers.Input(input_shape)

    with tf.name_scope('inputs') as scope:
        sh = inputs.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(inputs, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat([out, tf.zeros_like(out)], i)
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        outputs = tf.reshape(out, out_size, name=scope)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

def transitionDown(filters=K, size=1, input_shape=[]):

    initializer = tf.random_normal_initializer(0., 0.02)

    down = tf.keras.Sequential()
    down.add(tf.keras.layers.BatchNormalization())
    down.add(tf.keras.layers.ReLU())
    down.add(
        tf.keras.layers.Conv2D(filters, size, strides=1,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    down.add(tf.keras.layers.Dropout(DROP_OUT))
    # down.add(tf.keras.layers.GlobalAveragePooling2D())
    down.add(tf.keras.layers.Conv2D(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    return down

def transitionUp(filters=K, size=3, input_shape=[]):

    initializer = tf.random_normal_initializer(0., 0.02)

    up = tf.keras.Sequential()
    up.add(
        tf.keras.layers.Conv2D(filters, size, strides=1,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))
    up.add(tf.keras.layers.ReLU())
    up.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))
    up.add(
        tf.keras.layers.Conv2D(filters, size, strides=1,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))
    up.add(tf.keras.layers.ReLU())

    return up

def DenseBlockLayer(filters=K, size=3, input_shape=[]):
    initializer = tf.random_normal_initializer(0., 0.02)

    layer = tf.keras.Sequential()
    layer.add(tf.keras.layers.BatchNormalization())
    layer.add(tf.keras.layers.ReLU())
    layer.add(
        tf.keras.layers.Conv2D(filters, size, strides=1,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    layer.add(tf.keras.layers.Dropout(DROP_OUT))

    return layer

def DenseBlock(filters=K, size=2, input_shape=[]):

    inputs = tf.keras.layers.Input(input_shape)
    outputs = inputs
    inputflow = []

    if len(inputs.shape)==3 and inputs.shape[2]==3:
        inputflow.append(tf.keras.layers.Conv2D(K, 1, strides=1, padding='same', use_bias=False)(inputs))
    else:
        inputflow.append(inputs)

    outputs = tf.keras.layers.Concatenate()([DenseBlockLayer()(outputs), inputflow[0]])
    inputflow.append(outputs)

    outputs = tf.keras.layers.Concatenate()([DenseBlockLayer()(outputs), inputflow[0], inputflow[1]])
    inputflow.append(outputs)

    outputs = tf.keras.layers.Concatenate()([DenseBlockLayer()(outputs), inputflow[0], inputflow[1], inputflow[2]])

    return tf.keras.Model(inputs=inputs, outputs=outputs)

def DenseNet(filters=K, size=2, input_shape=[256,256,3]):

    inputs = tf.keras.layers.Input(input_shape)

    initializer = tf.random_normal_initializer(0., 0.02)

    x = inputs

    # build the main network body

    initial = tf.keras.Sequential()
    initial.add(tf.keras.layers.BatchNormalization())
    initial.add(tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                               kernel_initializer=initializer, use_bias=False))
    initial.add(tf.keras.layers.MaxPooling2D())

    x = initial(x)

    n = int(N/2)
    down_stack = []
    downTransition_stack = []
    up_stack = []
    upTransition_stack = []
    while n != 1:

        down_stack.append(DenseBlock(input_shape=[n, n, K]))
        downTransition_stack.append(transitionDown())

        if n == 1:
            up_stack.append(DenseBlock(input_shape=[n, n, K]))

        else:
            up_stack.append(DenseBlock(input_shape=[n, n, 256+K]))


        upTransition_stack.append(transitionUp())

        n = int(n / 2)



    up_stack.reverse()
    upTransition_stack.reverse()

    # Downsampling through the model
    skips = []
    print("down shapes")
    print(x.shape)
    for down, downTrans in zip(down_stack, downTransition_stack):
        x = down(x)
        skips.append(x)
        x = downTrans(x)
        print(x.shape)

    # x = DenseBlock()(x)

    skips = list(reversed(skips))
    #
    # print("skip shapes")
    # for skip in skips:
    #     print(skipsample(skip).shape)

    print("up shapes")
    # Upsampling and establishing the skip connections
    for up, upTrans, skip in zip(up_stack, upTransition_stack, skips):

        x = upTrans(x)
        print(x.shape)
        x = tf.keras.layers.Concatenate()([x, skip])
        x = up(x)
        print(x.shape)

    last = tf.keras.Sequential()
    last.add(tf.keras.layers.Conv2D(3, 3, strides=1, padding='same',
                                 kernel_initializer=initializer, use_bias=False))
    last.add(tf.keras.layers.Conv2DTranspose(3, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))
    last.add(tf.keras.layers.ReLU())

    outputs = last(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

densenet = DenseNet()

densenet.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['mse'])
history = densenet.fit(train_dataset, epochs=10,
                    validation_data=test_dataset)
