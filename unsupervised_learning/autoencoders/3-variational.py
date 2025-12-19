#!/usr/bin/env python3
"""This module contains the function that creates a variational autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    Args:
        input_dims is an integer containing the
        dimensions of the model input
        hidden_layers is a list containing the number of nodes
        for each hidden layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
        latent_dims is an integer containing the
        dimensions of the latent space representation
    Returns: encoder, decoder, auto
        encoder is the encoder model, which should output the
        latent representation, the mean, and the log variance, respectively
        decoder is the decoder model
        auto is the full autoencoder model
    The autoencoder model should be compiled using
    adam optimization and binary cross-entropy loss
    All layers should use a relu activation except for the
    mean and log variance layers in the encoder,
    which should use None, and the last layer in the
    decoder, which should use sigmoid
    """

    # --- Build Encoder ---
    inputs = keras.Input(shape=(input_dims,))

    encoded = inputs
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)

    # Latent Space: Mean and Log Variance (No activation)
    z_mean = keras.layers.Dense(latent_dims, activation=None)(encoded)
    z_log_sigma = keras.layers.Dense(latent_dims, activation=None)(encoded)

    # Sampling Function (Reparameterization Trick)
    def sampling(args):
        z_m, z_ls = args
        batch = keras.backend.shape(z_m)[0]
        dim = keras.backend.int_shape(z_m)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_m + keras.backend.exp(z_ls / 2) * epsilon

    # Sample z from latent distribution
    z = keras.layers.Lambda(sampling,
                            output_shape=(latent_dims,))([z_mean, z_log_sigma])

    # Encoder outputs: z (sampled), mean, and log_variance
    encoder = keras.Model(inputs, [z, z_mean, z_log_sigma], name='encoder')

    # --- Build Decoder ---
    decoder_inputs = keras.Input(shape=(latent_dims,))

    decoded = decoder_inputs
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)

    # Output layer with sigmoid activation
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)

    decoder = keras.Model(decoder_inputs, outputs, name='decoder')

    # --- Build Full Autoencoder ---
    # Get encoder outputs
    z_sample, z_mean_out, z_log_sigma_out = encoder(inputs)

    # Pass sampled z through decoder
    reconstructed = decoder(z_sample)

    # Full autoencoder model
    auto = keras.Model(inputs, reconstructed, name='autoencoder')

    # --- Custom VAE Loss ---
    # Reconstruction loss (binary crossentropy)
    reconstruct_loss = keras.losses.binary_crossentropy(inputs, reconstructed)
    reconstruct_loss *= input_dims

    # KL Divergence loss
    kl_loss = -0.5 * keras.backend.sum(
        1 + z_log_sigma_out - keras.backend.square(z_mean_out) -
        keras.backend.exp(z_log_sigma_out),
        axis=-1
    )

    # Total VAE loss
    vae_loss = keras.backend.mean(reconstruct_loss + kl_loss)

    # Add loss to the model
    auto.add_loss(vae_loss)

    # Compile with adam optimizer
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
