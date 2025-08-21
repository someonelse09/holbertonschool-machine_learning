#!/usr/bin/env python3
""" This module includes class
implementation of a decision tree """

import numpy as np


class Node:
    """ class to interpret Nodes """
    def __init__(self, feature=None, threshold=None, left_child=None,
        right_child=None, is_root=False, depth=0):
        """ Constructor of Node class """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """ method to find maximum depth """
        if self.left_child is None and self.right_child is None:
            return self.depth
        max_depth = self.depth
        if self.left_child is not None:
            max_depth = max(max_depth, self.left_child.max_depth_below())
        if self.right_child is not None:
            max_depth = max(max_depth, self.right_child.max_depth_below())
        return max_depth

    def count_nodes_below(self, only_leaves=False):
        """method to count all nodes(or leaves)"""
        if only_leaves:
            count = 0
        else:
            count = 1
        if self.left_child is not None:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child is not None:
            count += self.right_child.count_nodes_below(only_leaves)
        return count

    def left_child_add_prefix(self, text):
        """this method adds prefixes to the left child nodes"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """this method adds prefixes to the right child nodes"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return (new_text)

    def __str__(self):
        """this method implements the str structure for the nodes"""
        if self.is_root:
            result = f"root [feature={self.feature}, threshold={self.threshold}]\n"
        else:
            result = f"-> node [feature={self.feature}, threshold={self.threshold}]\n"
        if self.left_child is not None:
            left_part = str(self.left_child)
            result += self.left_child_add_prefix(left_part)
        if self.right_child is not None:
            right_part = str(self.right_child)
            result += self.right_child_add_prefix(right_part)
        return result.rstrip()

    def get_leaves_below(self):
        """this method retrieves all the leaves"""
        leaves_below = []
        if self.left_child is not None:
            leaves_below.extend(self.left_child.get_leaves_below())
        if self.right_child is not None:
            leaves_below.extend(self.right_child.get_leaves_below())
        return leaves_below

    def update_bounds_below(self):
        """method to update bounds"""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.upper = self.upper.copy()
                child.lower = self.lower.copy()

                if child == self.left_child:
                    # Left: feature <= threshold (so upper bound becomes threshold)
                    child.lower[self.feature] = self.threshold
                else:
                    # Right: feature > threshold (so lower bound becomes threshold)
                    child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()

    def _collect_features(self, features):
        """ method to collect features """
        if self.feature is not None:
            features.add(self.feature)
        if self.left_child:
            self.left_child._collect_features(features)
        if self.right_child:
            self.right_child._collect_features(features)

    def update_indicator(self):
        """this method serves as indicator updater"""
        def is_large_enough(x):
            """this method checks whether  x is large enough or not"""
            # <- fill the gap : this function returns a 1D numpy array of size
            # `n_individuals` so that the `i`-th element of the later is `True`
            # if the `i`-th individual has all its features > the lower bounds
            conditions = [np.greater(x[:, key], self.lower[key]) for key in list(self.lower.keys())]
            # np.all with axis=0 ensures ALL conditions are True for each individual
            return np.all(np.array(conditions), axis=0)
        def is_small_enough(x):
            """this method checks whether x is small enough or not"""

            #  <- fill the gap : this function returns a 1D numpy array of size
            # `n_individuals` so that the `i`-th element of the later is `True`
            # if the `i`-th individual has all its features <= the lower bounds
            conditions = [np.less_equal(x[:, key], self.upper[key]) for key in list(self.upper.keys())]
            # np.all with axis=0 ensures ALL conditions are True for each individual
            return np.all(np.array(conditions), axis=0)

        self.indicator = lambda x: np.all(np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """prediction function for the Nodes"""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """ class to interpret Leaves of the tree """
    def __init__(self, value, depth=None):
        """Constructor of leaf class"""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """ method to return maximum depth """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """method to return 1 if a leaf is reached"""
        return 1

    def __str__(self):
        """method to return str structure of the leaves"""
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """method to return leaves"""
        return [self]

    def update_bounds_below(self):
        """boundary update method of Leaf class"""
        pass

    def pred(self, x):
        """method to predict for the Leaves"""
        return self.value

class Decision_Tree():
    """ class to interpret Decision Trees """
    def __init__(self, max_depth=10, min_pop=1,
        seed=0, split_criterion="random", root=None):
        """ Constructor of Decision Tree class """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self) :
        """ method to return depth of the tree """
        return self.root.max_depth_below()
    def count_nodes(self, only_leaves=False):
        """method to apply count_nodes_below
        method of Node class to root """
        return self.root.count_nodes_below(only_leaves)

    def __str__(self):
        """method to return str structure of the decision tree"""
        return self.root.__str__()

    def depth(self) :
        """ method to return depth of the tree """
        return self.root.max_depth_below()
    def count_nodes(self, only_leaves=False):
        """method to apply count_nodes_below
        method of Node class to root """
        return self.root.count_nodes_below(only_leaves)

    def __str__(self):
        """method to return str structure of the decision tree"""
        return self.root.__str__()

    def get_leaves(self):
        """method to return leaves of the decision tree"""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """boundary update method of Decision Tree class"""
        self.root.update_bounds_below()

    def pred(self, x):
        """method to generalise pred to whole decision tree"""
        return self.root.pred(x)
    def update_predict(self):
        """method for updating predictions"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        def predict_function(A):
            indicators = np.array([leaf.indicator(A) for leaf in leaves])
            values = np.array([leaf.value for leaf in leaves])
            predictions = np.zeros(A.shape[0], dtype=int)
            for i in range(A.shape[0]):
                leaf_idx = np.where(indicators[:, i])[0][0]
                predictions[i] = values[leaf_idx]
            return predictions
        self.predict = predict_function

