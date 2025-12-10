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

    def total_cost(self, generated_image):
        """Calculates the total cost for the generated image
        Args:
            generated_image - a tf.Tensor of shape (1, nh, nw, 3)
             containing the generated image
            if generated_image is not an instance of tf.Tensor or tf.Variable with
             the same shape as self.content_image, raise a TypeError with the message
              generated_image must be a tensor of shape {s}
               where {s} is the shape of self.content_image
        Returns:
            (J, J_content, J_style)
            J is the total cost
            J_content is the content cost
            J_style is the style cost
        """
        s = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable))
           or generated_image.shape != s):
            raise TypeError(f"generated_image must be a tensor of shape {s}")
        preprocessed = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255
        )
        outputs = self.model(preprocessed)
        # Split outputs into
        # style (all except last one) and content (last one)
        style_outputs = outputs[:-1]
        content_outputs = outputs[-1]

        # Computing individual costs
        j_content = self.content_cost(content_outputs)
        j_style = self.style_cost(style_outputs)

        # Calculate total cost: J = alpha * J_content + beta * J_style
        j_total = self.alpha * j_content + self.beta * j_style

        return j_total, j_content, j_style

    def compute_grads(self, generated_image):
        """Calculates the gradients for the
         tf.Tensor generated image of shape (1, nh, nw, 3)
        Args:
            if generated_image is not an instance of tf.Tensor or tf.Variable
             with the same shape as self.content_image, raise a TypeError with the message
              generated_image must be a tensor of shape {s}
               where {s} is the shape of self.content_image
        Returns:
            gradients, J_total, J_content, J_style
            gradients is a tf.Tensor containing the gradients for the generated image
            J_total is the total cost for the generated image
            J_content is the content cost for the generated image
            J_style is the style cost for the generated image
        """
        s = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable))
           or generated_image.shape != s):
            raise TypeError(f"generated_image must be a tensor of shape {s}")
        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            # Calculating total cost and individual costs
            j_total, j_content, j_style = self.total_cost(generated_image)
        # Compute gradients of total cost with respect to generated_image
        gradients = tape.gradient(j_total, generated_image)

        return gradients, j_total, j_content, j_style

    def generate_image(self, iterations=1000,
                       step=None, lr=0.01, beta1=0.9, beta2=0.99):
        """
        Args:
            iterations - the number of iterations to perform gradient descent over
            step - if not None, the step at which you should print
             information about the training, including the final iteration:
            print Cost at iteration {i}: {J_total},
             content {J_content}, style {J_style}
            i is the iteration
            J_total is the total cost
            J_content is the content cost
            J_style is the style cost
            lr - the learning rate for gradient descent
            beta1 - the beta1 parameter for gradient descent
            beta2 - the beta2 parameter for gradient descent
            if iterations is not an integer, raise a TypeError
             with the message iterations must be an integer
            if iterations is not positive, raise a ValueError
             with the message iterations must be positive
            if step is not None and not an integer, raise a TypeError
             with the message step must be an integer
            if step is not None and not positive or
             less than iterations , raise a ValueError with the message
              step must be positive and less than iterations
            if lr is not a float or an integer, raise a TypeError
             with the message lr must be a number
            if lr is not positive, raise a ValueError with the message
             lr must be positive
            if beta1 is not a float, raise a TypeError with the message
             beta1 must be a float
            if beta1 is not in the range [0, 1], raise a ValueError
             with the message beta1 must be in the range [0, 1]
            if beta2 is not a float, raise a TypeError
             with the message beta2 must be a float
            if beta2 is not in the range [0, 1], raise a ValueError
             with the message beta2 must be in the range [0, 1]
            gradient descent should be performed using Adam optimization
            the generated image should be initialized as the content image
            keep track of the best cost and the image associated with that cost
        Returns:
            generated_image, cost
            generated_image is the best generated image
            cost is the best cost
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError("step must be positive and less than iterations")
        if not isinstance(lr, (float, int)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("beta1 must be in the range [0, 1]")
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("beta2 must be in the range [0, 1]")
        # Initialize generated image as content image
        generated_image = tf.Variable(self.content_image)

        # Initialize Adam optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=beta1,
            beta_2=beta2
        )
        # Track best cost and best image
        best_cost = float('inf')
        best_image = None

        for i in range(iterations + 1):
            # Compute gradients and costs
            grads, j_total, j_content, j_style = self.compute_grads(generated_image)
            # Update generated image using Adam optimizer
            optimizer.apply_gradients([(grads, generated_image)])
            # Clip pixel values to [0, 1] range
            generated_image.assign(tf.clip_by_value(generated_image, 0.0, 1.0))
            # Tracking best cost and image
            if j_total < best_cost:
                best_cost = j_total
                best_image = generated_image.numpy()
            # Printing information at specified steps
            if step is not None:
                if i % step == 0 or i == iterations:
                    print(f"Cost at iteration {i}: {j_total.numpy()}, "
                          f"content {j_content.numpy()}, style {j_style.numpy()}")
        return best_image, best_cost
