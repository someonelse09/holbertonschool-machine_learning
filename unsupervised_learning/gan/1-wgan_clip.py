#!/usr/bin/env python3
"""
This module includes the WGAN_clip class.

Implements a Wasserstein GAN with weight clipping to enforce
the Lipschitz constraint required for Wasserstein distance.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_clip(keras.Model):
    """
    Wasserstein GAN with weight clipping implementation.

    This GAN uses the Wasserstein distance as the loss metric,
    which provides more stable training compared to the original
    GAN formulation. Weight clipping is applied to enforce the
    Lipschitz constraint on the discriminator (critic).
    """

    def __init__(
        self,
        generator,
        discriminator,
        latent_generator,
        real_examples,
        batch_size=200,
        disc_iter=2,
        learning_rate=0.005
    ):
        """
        Initialize the WGAN components and optimizers.

        Args:
            generator: The generator neural network model
            discriminator: The discriminator (critic) network model
            latent_generator: Function to generate latent vectors
            real_examples: Tensor of real training examples
            batch_size: Number of samples per batch (default: 200)
            disc_iter: Number of discriminator updates per step
                      (default: 2)
            learning_rate: Learning rate for Adam optimizer
                          (default: 0.005)
        """
        super().__init__()

        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = 0.5
        self.beta_2 = 0.9

        # Generator loss: maximize discriminator output on fake samples
        # Loss = -mean(D(G(z)))
        self.generator.loss = lambda x: -tf.math.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss
        )

        # Discriminator loss: Wasserstein distance approximation
        # x = D(real), y = D(fake) (matching Simple_GAN arg pattern)
        # Want to maximize: mean(D(real)) - mean(D(fake)) = mean(x) - mean(y)
        # So minimize: -(mean(x) - mean(y)) = mean(y) - mean(x)
        self.discriminator.loss = (
            lambda x, y: tf.math.reduce_mean(y) -
            tf.math.reduce_mean(x)
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss
        )

    def get_fake_sample(self, size=None, training=False):
        """
        Generate fake samples using the generator.

        Args:
            size: Number of samples to generate. If None, uses
                 batch_size (default: None)
            training: Whether the generator is in training mode
                     (default: False)

        Returns:
            Tensor of generated samples
        """
        if not size:
            size = self.batch_size
        return self.generator(
            self.latent_generator(size),
            training=training
        )

    def get_real_sample(self, size=None):
        """
        Randomly sample real examples from the dataset.

        Args:
            size: Number of real samples to return. If None, uses
                 batch_size (default: None)

        Returns:
            Tensor of real samples
        """
        if not size:
            size = self.batch_size

        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """
        Perform one training step of the WGAN.

        The discriminator (critic) is trained multiple times per step
        with weight clipping, followed by a single generator update.

        Args:
            useless_argument: Unused argument required by Keras API

        Returns:
            Dictionary containing discriminator and generator losses
        """
        # Train discriminator multiple times
        for _ in range(self.disc_iter):
            # Get a real sample
            real_sample = self.get_real_sample()
            # Get a fake sample
            fake_sample = self.get_fake_sample(training=True)

            with tf.GradientTape() as tape:
                # Get discriminator outputs
                real_output = self.discriminator(
                    real_sample,
                    training=True
                )
                fake_output = self.discriminator(
                    fake_sample,
                    training=True
                )

                # Compute the discriminator loss
                # Note: x = real, y = fake (matching Simple_GAN pattern)
                discr_loss = self.discriminator.loss(
                    real_output,
                    fake_output
                )

            # Apply gradient descent to the discriminator
            gradients = tape.gradient(
                discr_loss,
                self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(
                    gradients,
                    self.discriminator.trainable_variables
                )
            )

            # Clip the weights between -1 and 1 (Lipschitz constraint)
            for var in self.discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -1.0, 1.0))

        # Train generator once
        with tf.GradientTape() as tape:
            # Get a fake sample
            fake_sample = self.get_fake_sample(training=True)

            # Get discriminator output
            fake_output = self.discriminator(
                fake_sample,
                training=True
            )

            # Compute the generator loss
            gen_loss = self.generator.loss(fake_output)

        # Apply gradient descent to the generator
        gradients = tape.gradient(
            gen_loss,
            self.generator.trainable_variables
        )
        self.generator.optimizer.apply_gradients(
            zip(
                gradients,
                self.generator.trainable_variables
            )
        )

        return {
            "discr_loss": discr_loss,
            "gen_loss": gen_loss
        }
