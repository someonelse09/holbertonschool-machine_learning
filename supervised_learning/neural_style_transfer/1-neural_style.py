#!/usr/bin/env python3
"""This module includes the class NST
that performs tasks for neural style transfer"""

import numpy as np
import tensorflow as tf


class NST:
    """
    Public class attributes:
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                        'block4_conv1', 'block5_conv1']
        content_layer = 'block5_conv2'
    """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Args:
            style_image - the image used as a style reference,
             stored as a numpy.ndarray
            content_image - the image used as a content reference,
             stored as a numpy.ndarray
            alpha - the weight for content cost
            beta - the weight for style cost
            if style_image is not a np.ndarray with the shape (h, w, 3),
             raise a TypeError with the message
              style_image must be a numpy.ndarray with shape (h, w, 3)
            if content_image is not a np.ndarray with the shape (h, w, 3),
             raise a TypeError with the message
              content_image must be a numpy.ndarray with shape (h, w, 3)
            if alpha is not a non-negative number, raise a TypeError
             with the message alpha must be a non-negative number
            if beta is not a non-negative number, raise a TypeError
             with the message beta must be a non-negative number
            Sets the instance attributes:
            style_image - the preprocessed style image
            content_image - the preprocessed content image
            alpha - the weight for content cost
            beta - the weight for style cost
            model - the Keras model used to calculate cost
        """
        if (not isinstance(style_image, np.ndarray) or
            style_image.ndim != 3) or (style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.model = self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Args:
            image - a numpy.ndarray of shape (h, w, 3)
             containing the image to be scaled
            if image is not a np.ndarray with the shape (h, w, 3),
             raise a TypeError with the message
              image must be a numpy.ndarray with shape (h, w, 3)
            The scaled image should be a tf.tensor with the shape
             (1, h_new, w_new, 3) where max(h_new, w_new) == 512
              and min(h_new, w_new) is scaled proportionately
            The image should be resized using bicubic interpolation
            After resizing, the image’s pixel values should be
             rescaled from the range [0, 255] to [0, 1].
        Returns:
            the scaled image
        """
        if (not isinstance(image, np.ndarray) or
                image.ndim != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        h, w, _ = image.shape
        # Calculate new dimensions
        if h > w:
            h_new = 512
            w_new = int(w * 512 / h)
        else:
            w_new = 512
            h_new = int(h * 512 / w)

        # Convert to tensor and add batch dimension
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, 0)

        # Resize using bicubic interpolation
        scaled = tf.image.resize(image_tensor,
                                 size=[h_new, w_new],
                                 method=tf.image.ResizeMethod.BICUBIC)

        # Rescale pixel values from [0, 255] to [0, 1]
        scaled = scaled / 255.0

        # Clip values to ensure they're in [0, 1] range
        scaled = tf.clip_by_value(scaled, 0.0, 1.0)

        return scaled

    def load_model(self):
        """
        Creates the model used to calculate cost
        Returns:
            the Keras model
        """
        # Load VGG19 with pretrained ImageNet weights, excluding top layers
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )
        # Freeze the VGG19 model
        vgg.trainable = False

        # Replace MaxPooling layers with AveragePooling layers
        for layer in vgg.layers:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                layer.__class__ = tf.keras.layers.AveragePooling2D

        # Get the outputs for style layers and content layer explicitly
        style_outputs = [vgg.get_layer(name).output
                         for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output

        # Combine outputs: style layers first, then content layer
        model_outputs = style_outputs + [content_output]

        # Create Model
        model = tf.keras.Model(inputs=vgg.input, outputs=model_outputs)

        # Crucial: Return the model so it can be assigned in __init__
        return model
