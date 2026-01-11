#!/usr/bin/env python3
"""
This module includes the WGAN_GP class.

Implements a Wasserstein GAN with gradient penalty to enforce
the Lipschitz constraint. This is an improvement over weight
clipping that provides more stable training.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_GP(keras.Model):
    """
    Wasserstein GAN with Gradient Penalty implementation.

    Uses gradient penalty instead of weight clipping to enforce
    the Lipschitz constraint, resulting in more stable training
    and better convergence compared to WGAN-clip.
    """

    def __init__(
        self,
        generator,
        discriminator,
        latent_generator,
        real_examples,
        batch_size=200,
        disc_iter=2,
        learning_rate=0.005,
        lambda_gp=10
    ):
        """
        Initialize the WGAN-GP components and optimizers.

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
            lambda_gp: Weight for gradient penalty term (default: 10)
        """
        super().__init__()

        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = 0.3
        self.beta_2 = 0.9

        # Gradient penalty weight
        self.lambda_gp = lambda_gp

        # Setup for computing gradient penalty
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(
            1, self.len_dims, delta=1, dtype='int32'
        )
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

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

        # Discriminator loss: Wasserstein distance
        # x = D(real), y = D(fake)
        # Maximize: mean(D(real)) - mean(D(fake))
        # Minimize: mean(D(fake)) - mean(D(real))
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

    def get_interpolated_sample(self, real_sample, fake_sample):
        """
        Generate interpolated samples between real and fake samples.

        Creates samples along straight lines between real and fake
        samples, used for computing the gradient penalty.

        Args:
            real_sample: Batch of real samples
            fake_sample: Batch of fake samples

        Returns:
            Tensor of interpolated samples
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """
        Compute the gradient penalty for WGAN-GP.

        Enforces the Lipschitz constraint by penalizing the gradient
        norm of the discriminator at interpolated points.

        Args:
            interpolated_sample: Batch of interpolated samples

        Returns:
            Gradient penalty value
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, useless_argument):
        """
        Perform one training step of the WGAN-GP.

        The discriminator is trained multiple times with gradient
        penalty, followed by a single generator update.

        Args:
            useless_argument: Unused argument required by Keras API

        Returns:
            Dictionary containing discriminator loss, generator loss,
            and gradient penalty
        """
        # Train discriminator multiple times
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                # Get a real sample
                real_sample = self.get_real_sample()
                # Get a fake sample
                fake_sample = self.get_fake_sample(training=True)
                # Get interpolated sample
                interpolated_sample = self.get_interpolated_sample(
                    real_sample, fake_sample
                )

                # Get discriminator outputs
                real_output = self.discriminator(
                    real_sample,
                    training=True
                )
                fake_output = self.discriminator(
                    fake_sample,
                    training=True
                )

                # Compute the standard discriminator loss
                discr_loss = self.discriminator.loss(
                    real_output,
                    fake_output
                )

                # Compute gradient penalty
                gp = self.gradient_penalty(interpolated_sample)

                # Compute total loss with gradient penalty
                new_discr_loss = discr_loss + self.lambda_gp * gp

            # Apply gradient descent to discriminator
            gradients = tape.gradient(
                new_discr_loss,
                self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(
                    gradients,
                    self.discriminator.trainable_variables
                )
            )

        # Train generator once
        with tf.GradientTape() as tape:
            # Get a fake sample
            fake_sample = self.get_fake_sample(training=True)

            # Get discriminator output
            fake_output = self.discriminator(
                fake_sample,
                training=True
            )

            # Compute generator loss
            gen_loss = self.generator.loss(fake_output)

        # Apply gradient descent to generator
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
            "gen_loss": gen_loss,
            "gp": gp
        }
