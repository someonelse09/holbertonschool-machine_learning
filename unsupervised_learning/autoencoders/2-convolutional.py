#!/usr/bin/env python3
"""This module includes the function
 that creates a convolutional autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Args:
        input_dims is a tuple of integers containing
         the dimensions of the model input
        filters is a list containing the number of filters
         for each convolutional layer in the encoder, respectively
        the filters should be reversed for the decoder
        latent_dims is a tuple of integers containing
         the dimensions of the latent space representation
        Each convolution in the encoder should use a kernel
         size of (3, 3) with same padding and relu activation,
          followed by max pooling of size (2, 2)
        Each convolution in the decoder, except for the last two,
         should use a filter size of (3, 3) with same padding and
          relu activation, followed by upsampling of size (2, 2)
        The second to last convolution should instead use valid padding
        The last convolution should have the same number of filters
         as the number of channels in input_dims with
          sigmoid activation and no upsampling
    Returns:
        encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
        The autoencoder model should be compiled using
         adam optimization and binary cross-entropy loss
    """
    inputs = keras.Input(shape=input_dims)

    encoded = inputs
    for f in filters:
        # "kernel size of (3, 3) with same padding and relu activation"
        encoded = keras.layers.Conv2D(filters=f,
                                      kernel_size=(3, 3),
                                      padding='same',
                                      activation='relu')(encoded)
        # "followed by max pooling of size (2, 2)"
        encoded = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                            padding='same')(encoded)

    encoder = keras.Model(inputs, encoded)

    # --- Build Decoder ---
    decoder_inputs = keras.Input(shape=latent_dims)

    decoded = decoder_inputs
    reversed_filters = filters[::-1]
    for i in range(len(reversed_filters) - 1):
        # "filter size of (3, 3) with same padding and relu activation"
        decoded = keras.layers.Conv2D(filters=reversed_filters[i],
                                      kernel_size=(3, 3),
                                      padding='same',
                                      activation='relu')(decoded)
        # "followed by upsampling of size (2, 2)"
        decoded = keras.layers.UpSampling2D(size=(2, 2))(decoded)

    decoded = keras.layers.Conv2D(filters=reversed_filters[-1],
                                  kernel_size=(3, 3),
                                  padding='valid',
                                  activation='relu')(decoded)
    decoded = keras.layers.UpSampling2D(size=(2, 2))(decoded)

    # --- Last Convolution (Output Layer) ---
    decoded = keras.layers.Conv2D(filters=input_dims[-1],
                                  kernel_size=(3, 3),
                                  padding='same',
                                  activation='sigmoid')(decoded)

    decoder = keras.Model(decoder_inputs, decoded)

    # --- Build Autoencoder ---
    auto_inputs = inputs
    auto_outputs = decoder(encoder(auto_inputs))
    auto = keras.Model(auto_inputs, auto_outputs)

    # Compile
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
