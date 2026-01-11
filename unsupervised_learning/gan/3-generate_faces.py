#!/usr/bin/env python3
"""
This module includes convolutional generator and discriminator
for face generation using GANs.

The generator uses transposed convolutions (upsampling) to create
16x16 grayscale images from 16-dimensional latent vectors.
The discriminator uses standard convolutions to classify images.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np


def convolutional_GenDiscr():
    """
    Build convolutional generator and discriminator for face generation.

    The generator transforms a 16-dimensional latent vector into
    a 16x16 grayscale image through upsampling and convolution.
    The discriminator classifies 16x16 images as real or fake.

    Returns:
        Tuple of (generator, discriminator) Keras models
    """

    def generator():
        """
        Build the generator model.

        Architecture:
        - Input: 16-dimensional latent vector
        - Dense layer to 2048 units
        - Reshape to (2, 2, 512)
        - 3 blocks of: UpSampling2D -> Conv2D -> BatchNorm -> Activation
        - Output: (16, 16, 1) image

        Returns:
            Keras Model for the generator
        """
        inputs = keras.Input(shape=(16,))

        # Dense layer and reshape
        x = keras.layers.Dense(2048, activation='tanh')(inputs)
        x = keras.layers.Reshape((2, 2, 512))(x)

        # First upsampling block: 2x2 -> 4x4
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(
            64, kernel_size=9, padding='same'
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('tanh')(x)

        # Second upsampling block: 4x4 -> 8x8
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(
            16, kernel_size=9, padding='same'
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('tanh')(x)

        # Third upsampling block: 8x8 -> 16x16
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(
            1, kernel_size=9, padding='same'
        )(x)
        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Activation('tanh')(x)

        return keras.Model(inputs, outputs, name="generator")

    def get_discriminator():
        """
        Build the discriminator model.

        Architecture:
        - Input: (16, 16, 1) grayscale image
        - 4 blocks of: Conv2D -> MaxPooling2D -> Activation
        - Flatten and Dense layer to output single value
        - Output: Scalar value (real/fake classification)

        Returns:
            Keras Model for the discriminator
        """
        inputs = keras.Input(shape=(16, 16, 1))

        # First conv block: 16x16 -> 8x8
        x = keras.layers.Conv2D(
            32, kernel_size=3, padding='same'
        )(inputs)
        x = keras.layers.MaxPooling2D(pool_size=2)(x)
        x = keras.layers.Activation('tanh')(x)

        # Second conv block: 8x8 -> 4x4
        x = keras.layers.Conv2D(
            64, kernel_size=3, padding='same'
        )(x)
        x = keras.layers.MaxPooling2D(pool_size=2)(x)
        x = keras.layers.Activation('tanh')(x)

        # Third conv block: 4x4 -> 2x2
        x = keras.layers.Conv2D(
            128, kernel_size=3, padding='same'
        )(x)
        x = keras.layers.MaxPooling2D(pool_size=2)(x)
        x = keras.layers.Activation('tanh')(x)

        # Fourth conv block: 2x2 -> 1x1
        x = keras.layers.Conv2D(
            256, kernel_size=3, padding='same'
        )(x)
        x = keras.layers.MaxPooling2D(pool_size=2)(x)
        x = keras.layers.Activation('tanh')(x)

        # Flatten and output
        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(1, activation='tanh')(x)

        return keras.Model(inputs, outputs, name="discriminator")

    return generator(), get_discriminator()
