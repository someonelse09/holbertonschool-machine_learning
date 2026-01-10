#!/usr/bin/env python3
"""
This module includes the Simple_GAN class.

The goal of the exercise is to implement a custom GAN model by
overriding the train_step method of keras.Model.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class Simple_GAN(keras.Model):
    """Implementation of a simple Generative Adversarial Network (GAN)."""

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
        """Initialize the GAN components and optimizers."""
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

        # Generator loss and optimizer
        self.generator.loss = (
            lambda x: tf.keras.losses.MeanSquaredError()(
                x, tf.ones(x.shape)
            )
        )
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss
        )

        # Discriminator loss and optimizer
        self.discriminator.loss = (
            lambda x, y: tf.keras.losses.MeanSquaredError()(
                x, tf.ones(x.shape)
            )
            + tf.keras.losses.MeanSquaredError()(
                y, -1 * tf.ones(y.shape)
            )
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
            size (int): Number of samples to generate.
            training (bool): Whether the generator is in training mode.

        Returns:
            Tensor of generated samples.
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
            size (int): Number of real samples to return.

        Returns:
            Tensor of real samples.
        """
        if not size:
            size = self.batch_size

        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """
        Perform one training step of the GAN.

        The discriminator is trained multiple times per step,
        followed by a single generator update.
        """
        # Train discriminator multiple times
        for _ in range(self.disc_iter):
            real_sample = self.get_real_sample()
            fake_sample = self.get_fake_sample(training=True)

            with tf.GradientTape() as tape:
                real_output = self.discriminator(
                    real_sample,
                    training=True
                )
                fake_output = self.discriminator(
                    fake_sample,
                    training=True
                )

                discr_loss = self.discriminator.loss(
                    real_output,
                    fake_output
                )

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

        # Train generator once
        with tf.GradientTape() as tape:
            fake_sample = self.get_fake_sample(training=True)
            fake_output = self.discriminator(
                fake_sample,
                training=True
            )
            gen_loss = self.generator.loss(fake_output)

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
