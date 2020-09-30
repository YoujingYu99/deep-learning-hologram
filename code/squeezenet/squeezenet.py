# code from https://colab.research.google.com/github/GoogleCloudPlatform/training-data-analyst/blob/master/courses/fast-and-lean-data-science/07_Keras_Flowers_TPU_squeezenet.ipynb#scrollTo=XLJNVGwHUDy1
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
REGULARIZE_RATIO = 0.1
DROP_OUT = 0.2
bnmomemtum = 0.9
EPOCHS = 100


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


# casting to int for matplotlib to show the image
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


def firedown(x, squeeze, expand):
    y = tf.keras.layers.Conv2D(filters=squeeze, kernel_size=1, activation='relu', padding='same')(x)
    y = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y)
    y1 = tf.keras.layers.Conv2D(filters=expand // 2, kernel_size=1, activation='relu', padding='same')(y)
    y1 = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y1)
    y3 = tf.keras.layers.Conv2D(filters=expand // 2, kernel_size=3, activation='relu', padding='same')(y)
    y3 = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y3)
    return tf.keras.layers.concatenate([y1, y3])


def fire_moduledown(squeeze, expand):
    return lambda x: firedown(x, squeeze, expand)

def upsample(filters=32, size=2, input_shape=[], apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    # if apply_dropout:
    #     result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())
    result.add(
        tf.keras.layers.Conv2D(filters, size + 1, strides=1, padding='same',
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

    return tf.keras.Model(inputs=inputs, outputs=x + y)


def Generator():
    N = 256
    n = N

    inputs = tf.keras.layers.Input(shape=[256, 256, 1])

    # size down
    y = tf.keras.layers.Conv2D(kernel_size=3, filters=32, padding='same', use_bias=True, activation='relu')(inputs)
    y = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y)
    y = fire_moduledown(24, 48)(y)
    y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
    y = fire_moduledown(48, 96)(y)
    y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
    y = fire_moduledown(64, 128)(y)
    y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
    y = fire_moduledown(96, 192)(y)
    y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
    y = fire_moduledown(64, 128)(y)
    y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
    y = fire_moduledown(48, 96)(y)
    y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
    y = fire_moduledown(24, 32)(y)
    y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)


    up_stack = []
    # size up
    while n != 1:
        if n / 2 == 1:
            up_stack.append(upsample(input_shape=[int(n), int(n), K]))
            break
        else:
            up_stack.append(upsample(input_shape=[int(n), int(n), K]))

        n = int(n / 2)

    up_stack.reverse()

    up_stack = up_stack[:-1]
    for up in up_stack[0:len(up_stack)-1]:
        y = up(y)
        print(y.shape)


    y = up_stack[len(up_stack) - 1](y)

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2D(1, 3, strides=1, padding='same',
                                  kernel_initializer=initializer, use_bias=False,
                                  kernel_regularizer=tf.keras.regularizers.L2(REGULARIZE_RATIO))

    outputs = last(y)

    return tf.keras.Model(inputs=inputs, outputs=outputs)



generator = Generator()
generator.compile(optimizer='adam',
                   loss=tf.keras.losses.MeanSquaredError(),
                   metrics=['mse'])
# history = generator.fit(train_dataset, epochs=10,
#                         validation_data=test_dataset)

generator.summary()

# def generate_images(model, test_input, tar):
#     prediction = model(test_input, training=True)
#     plt.figure(figsize=(15, 15))
#
#     display_list = [test_input[0], tar[0], prediction[0]]
#     title = ['Input Image', 'Ground Truth', 'Predicted Image']
#
#     for i in range(3):
#         plt.subplot(1, 3, i + 1)
#         plt.title(title[i])
#         plt.gray()
#         # getting the pixel values between [0, 1] to plot it.
#         img = tf.squeeze(display_list[i]) * 0.5 + 0.5
#         plt.imshow(img)
#
#         plt.axis('off')
#     plt.show()
#
# def fit(train_ds, epochs, test_ds):
#
#     print("fitting")
#     for epoch in range(epochs):
#         print("fitting")
#         start = time.time()
#
#         display.clear_output(wait=True)
#
#         for example_input, example_target in test_ds.take(1):
#             generate_images(generator, example_input, example_target)
#         print("Epoch: ", epoch)
#         #history = densenet.fit(train_dataset, epochs=10,
#                     #validation_data=test_dataset)
#
#         # Train
#         for n, (input_image, target) in train_ds.enumerate():
#             print('.', end='')
#             if (n + 1) % 10 == 0:
#                 print()
#             #train_step(input_image, target, epoch)
#         print()
#
#         # saving (checkpoint) the model every 20 epochs
#         # if (epoch + 1) % 2 == 0:
#         #     checkpoint.save(file_prefix=checkpoint_prefix)
#
#         print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
#                                                            time.time() - start))
#     # checkpoint.save(file_prefix=checkpoint_prefix)
#
#
# fit(train_dataset, EPOCHS, test_dataset)