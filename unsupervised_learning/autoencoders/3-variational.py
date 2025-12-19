#!/usr/bin/env python3
"""This module contains the function
that creates a variational autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
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
    inputs = keras.Input(shape=(input_dims,))

    encoded = inputs
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)

    # Latent Space: Mean and Log Variance (Activation IS None)
    z_mean = keras.layers.Dense(latent_dims, activation=None)(encoded)
    z_log_sigma = keras.layers.Dense(latent_dims, activation=None)(encoded)

    # Sampling Function (Reparameterization Trick)
    def sampling(args):
        z_m, z_ls = args
        # Get batch size and dimension dynamically from the input tensor
        batch = keras.backend.shape(z_m)[0]
        dim = keras.backend.int_shape(z_m)[1]

        # epsilon ~ N(0, 1)
        epsilon = keras.backend.random_normal(shape=(batch, dim))

        # z = mu + sigma * epsilon (where sigma = exp(log_sigma / 2))
        return z_m + keras.backend.exp(z_ls / 2) * epsilon

    # Lambda Layer
    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))([z_mean, z_log_sigma])

    # Encoder outputs: z, mean, and log_variance
    encoder = keras.Model(inputs, [z, z_mean, z_log_sigma])

    # --- Build Decoder ---
    decoder_inputs = keras.Input(shape=(latent_dims,))

    decoded = decoder_inputs
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)

    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)

    decoder = keras.Model(decoder_inputs, outputs)

    # --- Build Autoencoder ---
    auto_inputs = inputs
    # Get the 3 outputs from encoder
    z_enc, z_mean_enc, z_log_sigma_enc = encoder(auto_inputs)
    # Pass only 'z' to the decoder
    auto_outputs = decoder(z_enc)

    auto = keras.Model(auto_inputs, auto_outputs)

    # --- KL Divergence Loss ---
    def kl_loss_function(inputs):
        z_m, z_ls = inputs
        # KL loss formula
        kl_loss = -0.5 * keras.backend.sum(
            1 + z_ls - keras.backend.square(z_m) - keras.backend.exp(z_ls),
            axis=-1
        )
        return kl_loss

    # Calculate KL using the encoder outputs
    kl = kl_loss_function([z_mean_enc, z_log_sigma_enc])

    # Add KL loss to the model (averaged over batch)
    auto.add_loss(keras.backend.mean(kl))

    # Compile
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
