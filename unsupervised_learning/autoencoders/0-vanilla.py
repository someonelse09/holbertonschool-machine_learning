#!/usr/bin/env python3
"""This module includes the function
that creates an autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Args:
        input_dims is an integer containing the
         dimensions of the model input
        hidden_layers is a list containing the number of
         nodes for each hidden layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
        latent_dims is an integer containing the
         dimensions of the latent space representation
    Returns:
        encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    The autoencoder model should be compiled using
     adam optimization and binary cross-entropy loss
    All layers should use a relu activation except for the
     last layer in the decoder, which should use sigmoid
    """
    # Input layer
    inputs = keras.Input(shape=(input_dims,))

    # Hidden layers for Encoder
    encoded = inputs
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)

    # Latent space (Bottleneck)
    # "All layers should use a relu activation..."
    latent = keras.layers.Dense(latent_dims, activation='relu')(encoded)

    # Instantiate Encoder Model
    encoder = keras.Model(inputs, latent)

    # --- Build Decoder ---
    # Input to decoder is the latent vector
    decoder_inputs = keras.Input(shape=(latent_dims,))

    # Hidden layers for Decoder (Reversed)
    decoded = decoder_inputs
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)

    # Output layer (Reconstruction)
    # "...except for the last layer in the decoder, which should use sigmoid"
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)

    # Instantiate Decoder Model
    decoder = keras.Model(decoder_inputs, outputs)

    # --- Build Autoencoder ---
    # Connect Encoder and Decoder
    auto_inputs = inputs
    auto_outputs = decoder(encoder(auto_inputs))
    auto = keras.Model(auto_inputs, auto_outputs)

    # Compile
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto

