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
             class number that each box in filtered_boxes predicts, respectively
            box_scores: a numpy.ndarray of shape (?) containing the
             box scores for each box in filtered_boxes, respectively
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []
        for i in range(len(boxes)):
            # Calculating box scores: confidence * class_probability
            # box_confidences[i]: (grid_height, grid_width, anchor_boxes, 1)
            # box_class_probs[i]: (grid_height, grid_width, anchor_boxes, classes)
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
             class number for the class that filtered_boxes predicts, respectively
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
            cls_boxes = filtered_boxes[class_mask]  # Renamed for clarity, like groupmate
            cls_scores = box_scores[class_mask]  # Renamed for clarity, like groupmate

            # Sorting boxes by score in descending order
            # The order array now holds the indices of cls_boxes sorted by score
            order = np.argsort(cls_scores)[::-1]

            while len(order) > 0:
                # Get the index of the box with the highest score from the 'order' list
                i = order[0]

                # Keep the first box (highest score) and its score/class
                box_predictions.append(cls_boxes[i])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_scores[i])

                if len(order) == 1:
                    break

                # --- IoU Calculation (like your groupmate's internal logic) ---
                # Coordinates of the best box
                best_box = cls_boxes[i]

                # Coordinates of all other candidate boxes (using remaining indices in 'order')
                candidate_boxes = cls_boxes[order[1:]]

                # Calculate intersection coordinates
                x1 = np.maximum(best_box[0], candidate_boxes[:, 0])
                y1 = np.maximum(best_box[1], candidate_boxes[:, 1])
                x2 = np.minimum(best_box[2], candidate_boxes[:, 2])
                y2 = np.minimum(best_box[3], candidate_boxes[:, 3])

                # Calculate intersection area
                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                inter_area = inter_w * inter_h

                # Calculate union area
                best_box_area = (best_box[2] - best_box[0]) * (best_box[3] - best_box[1])
                candidate_areas = ((candidate_boxes[:, 2] - candidate_boxes[:, 0]) *
                                   (candidate_boxes[:, 3] - candidate_boxes[:, 1]))
                union = best_box_area + candidate_areas - inter_area

                iou = inter_area / union
                # --- End IoU Calculation ---

                # Keep indices where IoU is less than nms_t
                keep = np.where(iou <= self.nms_t)[0]

                # Update 'order' to only contain the indices that were kept (plus 1 because
                # we sliced from index 1 for the IoU calculation)
                order = order[keep + 1]

        # Convert lists to final numpy arrays
        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores
