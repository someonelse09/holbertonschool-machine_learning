#!/usr/bin/env python3
"""This module includes the class that uses
the Yolo v3 algorithm to perform object detection"""

import numpy as np
from tensorflow import keras as K


class Yolo:
    """
        Public instance attributes:
            model: the Darknet Keras model
            class_names: a list of the class names for the model
            class_t: the box score threshold for the initial filtering step
            nms_t: the IOU threshold for non-max suppression
            anchors: the anchor boxes
        """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Args:
            model_path is the path to where a Darknet Keras model is stored
            classes_path is the path to where the list of
             class names used for the Darknet model,
              listed in order of index, can be found
            class_t is a float representing the box score
             threshold for the initial filtering step
            nms_t is a float representing the IOU threshold for non-max suppression
            anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
             containing all of the anchor boxes:
            outputs is the number of outputs (predictions) made by the Darknet model
            anchor_boxes is the number of anchor boxes used for each prediction
            2 => [anchor_box_width, anchor_box_height]
        """
        self.model = K.models.load_model(model_path, compile=False)
        with open(classes_path, 'r') as file:
            self.class_names = [line.strip() for line in file.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Args:
            outputs is a list of numpy.ndarrays containing the
             predictions from the Darknet model for a single image:
            Each output will have the shape (grid_height,
             grid_width, anchor_boxes, 4 + 1 + classes)
            grid_height & grid_width => the height and
             width of the grid used for the output
            anchor_boxes => the number of anchor boxes used
            4 => (t_x, t_y, t_w, t_h)
            1 => box_confidence
            classes => class probabilities for all classes
            image_size is a numpy.ndarray containing the image's
             original size [image_height, image_width]
        Returns a tuple of (boxes, box_confidences, box_class_probs):
            boxes: a list of numpy.ndarrays of shape (grid_height,
             grid_width, anchor_boxes, 4) containing the processed
              boundary boxes for each output, respectively:
            4 => (x1, y1, x2, y2)
            (x1, y1, x2, y2) should represent the
             boundary box relative to original image
            box_confidences: a list of numpy.ndarrays of shape (grid_height,
             grid_width, anchor_boxes, 1) containing
              the box confidences for each output, respectively
            box_class_probs: a list of numpy.ndarrays of shape (grid_height,
             grid_width, anchor_boxes, classes) containing the
              box's class probabilities for each output, respectively
        """
        image_h, image_w = image_size
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            # Extract predictions
            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]
            object_confidence = output[..., 4:5]
            class_probs = output[..., 5:]

            # Create grid
            cx = np.tile(np.arange(grid_w).reshape(1, grid_w, 1),
                         (grid_h, 1, anchor_boxes))
            cy = np.tile(np.arange(grid_h).reshape(grid_h, 1, 1),
                         (1, grid_w, anchor_boxes))

            # Get anchors for this layer
            anchors = self.anchors[i]

            # Calculate box center (normalized to 0-1)
            bx = (1 / (1 + np.exp(-tx)) + cx) / grid_w
            by = (1 / (1 + np.exp(-ty)) + cy) / grid_h

            # Calculate box dimensions (normalized to 0-1)
            bw = (anchors[:, 0] * np.exp(tw)) / self.model.input.shape[1]
            bh = (anchors[:, 1] * np.exp(th)) / self.model.input.shape[2]

            # Convert to corner coordinates in image space
            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            # Stack into box format
            box = np.stack((x1, y1, x2, y2), axis=-1)
            boxes.append(box)

            # Apply sigmoid to confidences and class probabilities
            box_confidences.append(1 / (1 + np.exp(-object_confidence)))
            box_class_probs.append(1 / (1 + np.exp(-class_probs)))

        return boxes, box_confidences, box_class_probs
