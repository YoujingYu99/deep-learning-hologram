# code from https://www.tensorflow.org/tutorials/generative/pix2pix
# do the image generation with CNN network only
import tensorflow as tf

import os
import time

from matplotlib import pyplot as plt
from IPython import display

import tensorflow as tf

PATH = '../coco/'


BUFFER_SIZE = 400
BATCH_SIZE = 10
IMG_WIDTH = 256
IMG_HEIGHT = 256
N = 256  # size of training images 64*64
K = 32  # number of filters at convolutional layer
k = 12
REGULARIZE_RATIO = 0.1
DROP_OUT = 0.2


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


inp, re = load(PATH + 'coco_train/000000000776 resizedconcat.jpg')

#casting to int for matplotlib to show the image
# plt.figure()
# plt.imshow(inp / 255.0)
# plt.figure()
# plt.imshow(re / 255.0)


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


# plt.figure(figsize=(6, 6))
# for i in range(4):
#     rj_inp, rj_re = random_jitter(inp, re)
#     plt.subplot(2, 2, i + 1)
#     plt.imshow(rj_inp / 255.0)
#     plt.axis('off')
# plt.show()


def load_image_train(image_file):
    input_image, real_image = load(image_file)
    # input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                     IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


train_dataset = tf.data.Dataset.list_files(PATH + 'coco_train/*.jpg')
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(PATH + 'coco_test/*.jpg')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

OUTPUT_CHANNELS = 3

def downsample(filters=K, size=3, input_shape=[], apply_batchnorm=True):

  result = tf.keras.Sequential()
  result.add(tf.keras.layers.BatchNormalization())
  result.add(tf.keras.layers.ReLU())
  initializer = tf.random_normal_initializer(0., 0.02)
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False,
                                kernel_regularizer=tf.keras.regularizers.L2(REGULARIZE_RATIO)))

  result.add(tf.keras.layers.BatchNormalization())
  result.add(tf.keras.layers.ReLU())
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                             kernel_initializer=initializer, use_bias=False,
                                kernel_regularizer=tf.keras.regularizers.L2(REGULARIZE_RATIO)))

  shortcut=tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False,
                                kernel_regularizer=tf.keras.regularizers.L2(REGULARIZE_RATIO))
  inputs = tf.keras.layers.Input(input_shape)
  x = inputs
  x = result(x)
  y = inputs
  y = shortcut(y)

  return tf.keras.Model(inputs=inputs, outputs=x+y)


# down_model = downsample(filters=3, size=3, input_shape=[256,256,1])
# down_result = down_model(tf.expand_dims(inp, 0))
# print (down_result.shape)

def upsample(filters=K, size=2, input_shape=[], apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(tf.keras.layers.BatchNormalization())
  result.add(tf.keras.layers.ReLU())
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False,
                                kernel_regularizer=tf.keras.regularizers.L2(REGULARIZE_RATIO)))

  result.add(tf.keras.layers.BatchNormalization())

  # if apply_dropout:
  #     result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())
  result.add(
      tf.keras.layers.Conv2D(filters, size+1, strides=1, padding='same',
                             kernel_initializer=initializer, use_bias=False,
                                kernel_regularizer=tf.keras.regularizers.L2(REGULARIZE_RATIO)))

  shortcut = tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False,
                                kernel_regularizer=tf.keras.regularizers.L2(REGULARIZE_RATIO))

  inputs = tf.keras.layers.Input(input_shape)
  x = inputs
  x = result(x)
  y = inputs
  y = shortcut(y)
  # shortcut = tf.keras.Sequential()
  # shortcut.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
  #                            kernel_initializer=initializer, use_bias=False))

  return tf.keras.Model(inputs=inputs, outputs=x+y)

# up_model = upsample(filters=3, size=4, input_shape=[256,256,1])
# up_result = up_model(down_result)
# print (up_result.shape)

def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 1])

    # build the main network body
    n = N
    down_stack = []
    up_stack = []
    while n != 1:
        if n == N:
            down_stack.append(downsample(filters=K*k, input_shape=[n, n, 1]))
        else:
            down_stack.append(downsample(filters=K*k, input_shape=[n, n, K*k]))

        if n / 2 == 1:
            up_stack.append(upsample(input_shape=[int(n / 2), int(n / 2), K*k], filters=K*k))
        elif n / 2 == 128:
            up_stack.append(upsample(input_shape=[int(n / 2), int(n / 2), 2*K*k], filters=K))
        else:
            up_stack.append(upsample(input_shape=[int(n / 2), int(n / 2), 2*k*K], filters=K*k))

        n = int(n / 2)


    up_stack.reverse()

    initializer = tf.random_normal_initializer(0., 0.02)
    # last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
    #                                        strides=2,
    #                                        padding='same',
    #                                        kernel_initializer=initializer,
    #                                        activation='tanh') # (bs, 256, 256, 3)
    # up_stack.append(upsample(input_shape=[128,128,384]))
    # up_stack.append(tf.keras.layers.BatchNormalization())
    # up_stack.append(tf.keras.layers.ReLU())
    # up_stack.append(tf.keras.layers.Conv2D(K, 1, strides=1, padding='same',
    #                                       kernel_initializer=initializer, use_bias=False))

    # skip block
    skipsample = tf.keras.Sequential()
    skipsample.add(tf.keras.layers.Conv2D(K * k, 3, strides=1, padding='same',
                                          kernel_initializer=initializer, use_bias=False))
    skipsample.add(tf.keras.layers.BatchNormalization())
    skipsample.add(tf.keras.layers.ReLU())
    skipsample.add(tf.keras.layers.Conv2D(K * k, 3, strides=1, padding='same',
                                          kernel_initializer=initializer, use_bias=False))
    skipsample.add(tf.keras.layers.BatchNormalization())
    skipsample.add(tf.keras.layers.ReLU())

    # last skip block
    lastskipsample = tf.keras.Sequential()
    lastskipsample.add(tf.keras.layers.Conv2D(K, 3, strides=1, padding='same',
                                              kernel_initializer=initializer, use_bias=False))
    lastskipsample.add(tf.keras.layers.BatchNormalization())
    lastskipsample.add(tf.keras.layers.ReLU())
    lastskipsample.add(tf.keras.layers.Conv2D(K, 3, strides=1, padding='same',
                                              kernel_initializer=initializer, use_bias=False))
    lastskipsample.add(tf.keras.layers.BatchNormalization())
    lastskipsample.add(tf.keras.layers.ReLU())

    x = inputs

    # Downsampling through the model
    skips = []
    print("down shapes")
    for down in down_stack:
        skips.append(x)
        print(x.shape)
        x = down(x)

    skips = list(reversed(skips))
    #
    # print("skip shapes")
    # for skip in skips:
    #     print(skipsample(skip).shape)

    print("up shapes")
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack[0:len(up_stack) - 1], skips[0:len(skips) - 1]):
        x = up(x)
        print(x.shape)
        # print(skipsample(skip).shape)
        if x.shape.as_list()[1] == 256:
            pass
        else:
            x = tf.keras.layers.Concatenate()([x, skipsample(skip)])
        # if skip.shape==(256,256,3):
        #     x = tf.keras.layers.Concatenate()([x, lastskipsample(skip)])
        # else:
        #     x = tf.keras.layers.Concatenate()([x, skipsample(skip)])

    x = up_stack[len(up_stack) - 1](x)
    # x = tf.keras.layers.Concatenate()([x, lastskipsample(skips[len(skips)-1])])

    # R blocks
    R = tf.keras.Sequential()
    R.add(tf.keras.layers.BatchNormalization())
    R.add(tf.keras.layers.ReLU())
    R.add(tf.keras.layers.Conv2D(K * k, 3, strides=1, padding='same',
                                 kernel_initializer=initializer, use_bias=False))
    R.add(tf.keras.layers.BatchNormalization())
    R.add(tf.keras.layers.ReLU())
    R.add(tf.keras.layers.Conv2D(K, 3, strides=1, padding='same',
                                 kernel_initializer=initializer, use_bias=False))

    y = R(x)

    outputs = x + y

    last = tf.keras.layers.Conv2D(3, 3, strides=1, padding='same',
                                  kernel_initializer=initializer, use_bias=False)

    outputs = last(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

generator = Generator()

generator.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['mse'])
history = generator.fit(train_dataset, epochs=10,
                    validation_data=test_dataset)

def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.gray()
        # getting the pixel values between [0, 1] to plot it.
        img = tf.squeeze(display_list[i]) * 0.5 + 0.5
        plt.imshow(img)

        plt.axis('off')
    plt.show()
for example_input, example_target in test_dataset.take(1):
    generate_images(generator, example_input, example_target)