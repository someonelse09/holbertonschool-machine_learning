#!/usr/bin/env python3
"""This module includes the class NST
that performs tasks for neural style transfer"""

import tensorflow as tf
import numpy as np
from tensorflow.python.ops.numpy_ops.np_dtypes import float32


class NST:
    """
    Public class attributes:
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        content_layer = 'block5_conv2'
    """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
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
             raise a TypeError with the message style_image
              must be a numpy.ndarray with shape (h, w, 3)
            if content_image is not a np.ndarray with the shape (h, w, 3),
             raise a TypeError with the message content_image
              must be a numpy.ndarray with shape (h, w, 3)
            if alpha is not a non-negative number,
             raise a TypeError with the message
              alpha must be a non-negative number
            if beta is not a non-negative number,
             raise a TypeError with the message
              beta must be a non-negative number
            Sets the instance attributes:
            style_image - the preprocessed style image
            content_image - the preprocessed content image
            alpha - the weight for content cost
            beta - the weight for style cost
        """
        if not isinstance(style_image, np.ndarray) or len(style_image) != 3 or style_image[2] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or len(content_image) != 3 or content_image[2] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, float) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, float) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Args:
            image - a numpy.ndarray of shape (h, w, 3)
             containing the image to be scaled
            if image is not a np.ndarray with the shape (h, w, 3),
             raise a TypeError with the message image must
              be a numpy.ndarray with shape (h, w, 3)
            The scaled image should be a tf.tensor with the shape
             (1, h_new, w_new, 3) where max(h_new, w_new) == 512 and
              min(h_new, w_new) is scaled proportionately
            The image should be resized using bicubic interpolation
            After resizing, the image’s pixel values should be
             rescaled from the range [0, 255] to [0, 1].
        Returns:
            the scaled image
        """
        if not isinstance(image, np.ndarray) or len(image) != 3 or image[2] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")
        h, w, c = image.shape

        # We have to calculate new dimensions to make the largest side 512
        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))
        # Adding batch dimension and convert to tensor
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, axis=0)

        scaled_image = tf.image.resize(image_tensor,
                                       size=[h_new, w_new],
                                       method=tf.image.ResizeMethod.BICUBIC)
        # Rescaling pixels values from [0, 255] to [0, 1]
        scaled_image = scaled_image / 255.0
        scaled_image = tf.clip_by_value(scaled_image, 0.0, 1.0)

        return scaled_image
