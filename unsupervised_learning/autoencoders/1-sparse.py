#!/usr/bin/env python3
"""This module contains the function
 that creates a sparse autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Args:
        input_dims is an integer containing the
         dimensions of the model input
        hidden_layers is a list containing the number of nodes
         for each hidden layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
        latent_dims is an integer containing the
         dimensions of the latent space representation
        lambtha is the regularization parameter used for
         L1 regularization on the encoded output
    Returns:
        encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the sparse autoencoder model
    The sparse autoencoder model should be compiled using
     adam optimization and binary cross-entropy loss
    All layers should use a relu activation except for the
     last layer in the decoder, which should use sigmoid
    """
    inputs = keras.Input(shape=(input_dims,))

    encoded = inputs
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)

    ar = keras.regularizers.l1(lambtha)
    # Latent Space (Bottleneck)
    latent = keras.layers.Dense(latent_dims,
                                activation='relu',
                                activity_regularizer=ar)(encoded)

    encoder = keras.Model(inputs, latent)

    # --- Build Decoder ---
    decoder_inputs = keras.Input(shape=(latent_dims,))

    decoded = decoder_inputs
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)

    # Output layer (Sigmoid for reconstruction)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)

    decoder = keras.Model(decoder_inputs, outputs)

    # --- Build Autoencoder ---
    auto_inputs = inputs
    auto_outputs = decoder(encoder(auto_inputs))
    auto = keras.Model(auto_inputs, auto_outputs)

    # Compile
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
