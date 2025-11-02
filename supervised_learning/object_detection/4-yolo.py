#!/usr/bin/env python3
"""This module includes the class that uses
the Yolo v3 algorithm to perform object detection"""

import numpy as np
from tensorflow import keras as K
import os
import cv2


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
            nms_t is a float representing the IOU
             threshold for non-max suppression
            anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
             containing all of the anchor boxes:
            outputs is the number of outputs
             (predictions) made by the Darknet model
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
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        processed_boxes = []
        box_confidence_list = []
        box_class_probs_list = []

        for i, output in enumerate(outputs):
            grid_h, grid_w, num_anchors, _ = output.shape

            # 1. Splitting raw predictions
            t_xy = output[..., :2]
            t_wh = output[..., 2:4]
            box_confidences = output[..., 4:5]
            box_class_probs = output[..., 5:]

            # 2. Applying sigmoid activation
            t_xy = 1.0 / (1.0 + np.exp(-t_xy))
            box_confidences = 1.0 / (1.0 + np.exp(-box_confidences))
            box_class_probs = 1.0 / (1.0 + np.exp(-box_class_probs))

            box_confidence_list.append(box_confidences)
            box_class_probs_list.append(box_class_probs)

            # 3. Create grid for cell offsets
            col = np.arange(grid_w).reshape(1, grid_w, 1)
            col = np.tile(col, [grid_h, 1, num_anchors])
            row = np.arange(grid_h).reshape(grid_h, 1, 1)
            row = np.tile(row, [1, grid_w, num_anchors])

            # 4. Calculate box center coordinates
            # bx = sigmoid(tx) + cx
            # by = sigmoid(ty) + cy
            box_xy = np.zeros_like(t_xy)
            box_xy[..., 0] = t_xy[..., 0] + col
            box_xy[..., 1] = t_xy[..., 1] + row

            # Normalize to grid
            box_xy[..., 0] /= grid_w
            box_xy[..., 1] /= grid_h

            # 5. Calculate box dimensions
            # bw = pw * e^(tw)
            # bh = ph * e^(th)
            anchors_layer = self.anchors[i]
            box_wh = anchors_layer * np.exp(t_wh)

            # Normalize by input dimensions
            box_wh[..., 0] /= input_w
            box_wh[..., 1] /= input_h

            # 6. Convert from center coords to corner coords
            # x1 = (bx - bw/2) * image_w
            # y1 = (by - bh/2) * image_h
            # x2 = (bx + bw/2) * image_w
            # y2 = (by + bh/2) * image_h
            box_xy[..., 0] *= image_w
            box_xy[..., 1] *= image_h
            box_wh[..., 0] *= image_w
            box_wh[..., 1] *= image_h

            x1 = box_xy[..., 0] - box_wh[..., 0] / 2
            y1 = box_xy[..., 1] - box_wh[..., 1] / 2
            x2 = box_xy[..., 0] + box_wh[..., 0] / 2
            y2 = box_xy[..., 1] + box_wh[..., 1] / 2

            processed_box = np.stack([x1, y1, x2, y2], axis=-1)
            processed_boxes.append(processed_box)

        return processed_boxes, box_confidence_list, box_class_probs_list

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Args:
            boxes: a list of numpy.ndarrays of shape (grid_height,
             grid_width, anchor_boxes, 4) containing the
              processed boundary boxes for each output, respectively
            box_confidences: a list of numpy.ndarrays of shape (grid_height,
             grid_width, anchor_boxes, 1) containing the
              processed box confidences for each output, respectively
            box_class_probs: a list of numpy.ndarrays of shape (grid_height,
             grid_width, anchor_boxes, classes) containing the
              processed box class probabilities for each output, respectively
        Returns a tuple of (filtered_boxes, box_classes, box_scores):
            filtered_boxes: a numpy.ndarray of shape (?, 4)
             containing all of the filtered bounding boxes:
            box_classes: a numpy.ndarray of shape (?,) containing the
             class number that each box in
              filtered_boxes predicts, respectively
            box_scores: a numpy.ndarray of shape (?) containing the
             box scores for each box in filtered_boxes, respectively
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []
        for i in range(len(boxes)):
            # Calculating box scores: confidence * class_probability
            # box_confidences[i]: (grid_height, grid_width, anchor_boxes, 1)
            # box_class_probs[i]: (grid_height,
            # grid_width, anchor_boxes, classes)
            scores = box_confidences[i] * box_class_probs[i]

            # We need to get the class with maximum score for each box
            # box_class: (grid_height, grid_width, anchor_boxes)
            box_class = np.argmax(scores, axis=-1)

            # Then get the maximum score for each box
            # box_score: (grid_height, grid_width, anchor_boxes)
            box_score = np.max(scores, axis=-1)

            # Creating filtering mask based threshold
            filtering_mask = box_score >= self.class_t

            # We have to apply this mask to
            # filter boxes, classes and scores
            filtered_boxes.append(boxes[i][filtering_mask])
            box_classes.append(box_class[filtering_mask])
            box_scores.append(box_score[filtering_mask])
        # Concatenating results from all scales
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Args:
            filtered_boxes: a numpy.ndarray of shape (?, 4)
             containing all of the filtered bounding boxes:
            box_classes: a numpy.ndarray of shape (?,) containing the
             class number for the class that
              filtered_boxes predicts, respectively
            box_scores: a numpy.ndarray of shape (?) containing
             the box scores for each box in filtered_boxes, respectively
        Returns a tuple of (box_predictions,
        predicted_box_classes, predicted_box_scores):
            box_predictions: a numpy.ndarray of shape (?, 4) containing
             all of the predicted bounding boxes ordered by class and box score
            predicted_box_classes: a numpy.ndarray of shape
             (?,) containing the class number for box_predictions
              ordered by class and box score, respectively
            predicted_box_scores: a numpy.ndarray of shape (?)
             containing the box scores for box_predictions
              ordered by class and box score, respectively
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []
        unique_classes = np.unique(box_classes)

        # Processing each class separately
        for cls in unique_classes:
            # First we get indices of boxes belonging to this class
            class_mask = box_classes == cls
            class_boxes = filtered_boxes[class_mask]
            class_scores = box_scores[class_mask]

            # Sorting boxes by score in descending order
            sorted_indices = np.argsort(class_scores)[::-1]
            class_boxes_sorted = class_boxes[sorted_indices]
            class_scores_sorted = class_scores[sorted_indices]

            # Now we need to apply non-maximum suppression
            keep_indices = []

            while len(class_boxes_sorted) > 0:
                # Keep the first box (highest score)
                keep_indices.append(0)

                if len(class_boxes_sorted) == 1:
                    break

                # Calculating IoU of the best box with all other boxes
                ious = self._iou(class_boxes_sorted[0], class_boxes_sorted[1:])

                # Keep only boxes with IoU less than threshold
                keep_mask = ious < self.nms_t
                class_boxes_sorted = class_boxes_sorted[1:][keep_mask]
                class_scores_sorted = class_scores_sorted[1:][keep_mask]

            # Reconstruct the kept boxes from the original sorted arrays
            kept_boxes = class_boxes[sorted_indices][:len(keep_indices)]
            kept_scores = class_scores[sorted_indices][:len(keep_indices)]

            box_predictions.append(kept_boxes)
            predicted_box_classes.append(np.full(len(keep_indices), cls))
            predicted_box_scores.append(kept_scores)

        # Concatenate all results
        box_predictions = np.concatenate(box_predictions, axis=0)
        predicted_box_classes = np.concatenate(predicted_box_classes, axis=0)
        predicted_box_scores = np.concatenate(predicted_box_scores, axis=0)

        return box_predictions, predicted_box_classes, predicted_box_scores

    def _iou(self, box, boxes):
        """
                Calculate Intersection over Union
                between one box and multiple boxes

                Args:
                    box: numpy.ndarray of shape (4,)
                    representing one box [x1, y1, x2, y2]
                    boxes: numpy.ndarray of shape (?, 4)
                    representing multiple boxes

                Returns:
                    numpy.ndarray of shape (?,) containing IoU values
                """
        # Calculating intersection coordinates
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        # We need to Calculate Intersection Area
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # Calculating union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - intersection

        iou = intersection / union

        return iou

    @staticmethod
    def load_images(folder_path):
        """
        Args:
            folder_path: a string representing the path
             to the folder holding all the images to load
        Returns a tuple of (images, image_paths):
            images: a list of images as numpy.ndarrays
            image_paths: a list of paths to the individual images in images
        """
        images = []
        image_paths = []

        # Get all files in the folder
        files = os.listdir(folder_path)

        for filename in files:
            filepath = os.path.join(folder_path, filename)

            # Checking if it is a file (not a directory)
            if os.path.isfile(filepath):
                # Trying to load the image
                image = cv2.imread(filepath)

                # We can add only if image was successfully loaded
                if image is not None:
                    images.append(image)
                    image_paths.append(filepath)

        return images, image_paths
