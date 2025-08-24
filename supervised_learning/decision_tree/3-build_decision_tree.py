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
        thres = "threshold={self.threshold}]"
        feat = "[feature={self.feature}"
        if self.is_root:
            result = (
                f"root " + feat + "," + thres + "\n"
            )
        else:
            result = (
                f"-> node" + feat + "," + thres + "\n"
            )
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

    def depth(self):
        """ method to return depth of the tree """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """method to apply count_nodes_below
        method of Node class to root """
        return self.root.count_nodes_below(only_leaves)

    def __str__(self):
        """method to return str structure of the decision tree"""
        return self.root.__str__()

    def depth(self):
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
