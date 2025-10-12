#!/usr/bin/env python3
"""This module includes the function that
builds the ResNet-50 architecture as described in
Deep Residual Learning for Image Recognition (2015)"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block

def resnet50():
    """
    Args:You can assume the input data will have shape (224, 224, 3)
        All convolutions inside and outside the blocks should be followed by batch normalization along the channels axis and a rectified linear activation (ReLU), respectively.
        All weights should use he normal initialization
        The seed for the he_normal initializer should be set to zero
        You may use:
        identity_block = __import__('2-identity_block').identity_block
        projection_block = __import__('3-projection_block').projection_block
    Returns:
        the keras model
    """
    initializer = K.initializers.HeNormal(seed=0)

    input_layer = K.Input(shape=(224, 224, 3))

    X = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializer
    )(input_layer)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(X)

    X = projection_block(X, [64, 64, 256], s=1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])

    X = projection_block(X, [128, 128,512], s=2)
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])

    X = projection_block(X, [256, 256, 1024], s=2)
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])

    X = projection_block(X, [512, 512, 2048], s=2)
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])

    X = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1)
    )(X)

    X = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=initializer
    )(X)

    model = K.models.Model(inputs=input_layer, outputs=X)

    return model
