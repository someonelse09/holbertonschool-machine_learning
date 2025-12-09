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
        self.load_model()
        self.generate_features()

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
        VGG19_model = tf.keras.applications.VGG19(include_top=False,
                                                  weights='imagenet')
        VGG19_model.save("VGG19_base_model")
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}

        vgg = tf.keras.models.load_model("VGG19_base_model",
                                         custom_objects=custom_objects)

        style_outputs = []
        content_output = None

        for layer in vgg.layers:
            if layer.name in self.style_layers:
                style_outputs.append(layer.output)
            if layer.name in self.content_layer:
                content_output = layer.output

            layer.trainable = False

        outputs = style_outputs + [content_output]

        model = tf.keras.models.Model(vgg.input, outputs)
        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """
        Args:
            input_layer - an instance of tf.Tensor or tf.Variable
             of shape (1, h, w, c) containing the layer output
              whose gram matrix should be calculated
            if input_layer is not an instance of tf.Tensor
             or tf.Variable of rank 4, raise a TypeError with
              the message input_layer must be a tensor of rank 4
        Returns:
            a tf.Tensor of shape (1, c, c) containing the gram matrix of input_layer
        """
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable)) or
            len(input_layer.shape) != 4):
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = input_layer.shape
        # Reshape to (batch, h*w, c)
        # We perform the product of the channels across the spatial features
        features = tf.reshape(input_layer, (1, -1, c))

        # Perform matrix multiplication: features^T * features
        # Dimensions: (1, c, h*w) * (1, h*w, c) -> (1, c, c)
        gram = tf.matmul(features, features, transpose_a=True)

        # Normalize by the number of spatial locations (h * w)
        gram = tf.cast(gram, tf.float32) / tf.cast(h * w, tf.float32)

        return gram

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost
        Sets the public instance attributes:
            gram_style_features - a list of gram matrices
             calculated from the style layer outputs of the style image
            content_feature - the content layer output of the content image
        """
        style_input = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255
        )
        content_input = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255
        )
        style_outputs = self.model(style_input)
        # We take all except last one (content layer)
        style_layer_outputs = style_outputs[:-1]

        self.gram_style_features = [
            self.gram_matrix(style_layer) for style_layer in style_layer_outputs
        ]

        content_outputs = self.model(content_input)
        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """Calculates the style cost for a single layer
        Args:
            style_output - tf.Tensor of shape (1, h, w, c)
             containing the layer style output of the generated image
            gram_target - tf.Tensor of shape (1, c, c) the gram matrix
             of the target style output for that layer
            if style_output is not an instance of tf.Tensor
             or tf.Variable of rank 4, raise a TypeError with
              the message style_output must be a tensor of rank 4
            if gram_target is not an instance of tf.Tensor or
             tf.Variable with shape (1, c, c), raise a TypeError with
              the message gram_target must be a tensor of shape [1, {c}, {c}]
               where {c} is the number of channels in style_output
        Returns:
            the layer’s style cost
        """
        if (not isinstance(style_output, (tf.Tensor, tf.Variable))
           or len(style_output.shape) != 4):
            raise TypeError("style_output must be a tensor of rank 4")
        # Get number of channels from style_output
        c = int(style_output.shape[-1])
        if (not isinstance(gram_target, (tf.Tensor, tf.Variable))
           or gram_target.shape != (1, c, c)):
            raise TypeError(f"gram_target must be a tensor of shape [1, {c}, {c}]")

        # Calculate gram matrix of the generated image's style output
        gram_style = self.gram_matrix(style_output)

        layer_cost = tf.reduce_mean(tf.square(gram_style - gram_target))

        return layer_cost

    def style_cost(self, style_outputs):
        """Calculates the style cost for generated image
        Args:
            style_outputs - a list of tf.Tensor style outputs for the generated image
            if style_outputs is not a list with the same length as self.style_layers,
             raise a TypeError with the message style_outputs must be a list with a
              length of {l} where {l} is the length of self.style_layers
            each layer should be weighted evenly with all weights summing to 1
        Returns:
            the style cost
        """
        l = len(self.style_layers)
        if (not isinstance(style_outputs, list)
           or len(style_outputs) != len(self.style_layers)):
            raise TypeError(f"style_outputs must be a list with a length of {l}")

        # Calculate weight for each layer (evenly distributed, sum to 1)
        w = 1.0 / len(self.style_layers)

        total_style_cost = 0.0

        for style_output, gram_target in zip(style_outputs, self.gram_style_features):
            layer_cost = self.layer_style_cost(style_output, gram_target)
            total_style_cost += w * layer_cost

        return total_style_cost

    def content_cost(self, content_output):
        """Calculates the content cost for the generated image
        Args:
            content_output - a tf.Tensor containing the
             content output for the generated image
            if content_output is not an instance of tf.Tensor or tf.Variable with
             the same shape as self.content_feature, raise a TypeError with the message
              content_output must be a tensor of shape {s}
               where {s} is the shape of self.content_feature
        Returns:
            the content cost
        """
        s = self.content_feature.shape
        if (not isinstance(content_output, (tf.Tensor, tf.Variable))
           or content_output.shape != s):
            raise TypeError(f"content_output must be a tensor of shape {s}")
        cost = tf.reduce_mean(tf.square(content_output - self.content_feature))

        return cost
