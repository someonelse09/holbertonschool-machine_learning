#!/usr/bin/env python3
"""This module includes the function that builds the
inception network as described in Going Deeper with Convolutions (2014):"""

from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Args:
        You can assume the input data will have shape (224, 224, 3)
        All convolutions inside and outside the inception block
         should use a rectified linear activation (ReLU)
        Using inception_block = __import__
         ('0-inception_block').inception_block is allowed
    Returns:
        the keras model
    """
    input_layer = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        activation='relu'
    )(input_layer)

    pool1 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=2,
        padding='same'
    )(conv1)

    conv2_reduce = K.layers.Conv2D(
        filters=64,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        activation='relu'
    )(pool1)

    conv2 = K.layers.Conv2D(
        filters=192,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu'
    )(conv2_reduce)

    pool2 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=2,
        padding='same'
    )(conv2)

    inception_3a = inception_block(pool2, [64, 96, 128, 16, 32, 32])
    inception_3b = inception_block(inception_3a, [128, 128, 192, 32, 96, 64])

    pool3 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=2,
        padding='same'
    )(inception_3b)

    inception_4a = inception_block(pool3, [192, 96, 208, 16, 48, 64])
    inception_4b = inception_block(inception_4a, [160, 112, 224, 24, 64, 64])
    inception_4c = inception_block(inception_4b, [128, 128, 256, 24, 64, 64])
    inception_4d = inception_block(inception_4c, [112, 144, 288, 32, 64, 64])
    inception_4e = inception_block(inception_4d, [256, 160, 320, 32,128, 128])

    pool4 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=2,
        padding='same'
    )(inception_4e)

    inception_5a = inception_block(pool4, [256, 160, 320, 32, 128, 128])
    inception_5b = inception_block(inception_5a, [384, 192, 384, 48, 128, 128])
    pool5 = K.layers.AvgPool2D(
        pool_size=(7, 7),
        strides=(1, 1),
        padding='valid'
    )(inception_5b)

    dropout = K.layers.Dropout(rate=0.4)(pool5)
    dense = K.layers.Dense(units=1000, activation='softmax')(dropout)
    model = K.models.Model(inputs=input_layer, outputs=dense)

    return model
