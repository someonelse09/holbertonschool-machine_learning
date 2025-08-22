#!/usr/bin/env python3
"""Isolation Random Tree algorithm."""
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf
import numpy as np


class Isolation_Random_Tree():
    """Isolation Random Trees class, to train the model labelless data."""

    def __init__(self, max_depth=10, seed=0, root=None):
        """Initialize isolation random tree object."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """Print the whole tree in ASCII."""
        return self.root.__str__() + "\n"

    def depth(self):
        """Return the maximum depth of tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Return the count of leaves."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """Bounds of the whole tree."""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Return the leaves of the decision tree."""
        return self.root.get_leaves_below()

    def update_predict(self):
        """Vectorize the predict function for isolation trees."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        self.predict = lambda A: np.array([leaf.value for leaf in leaves])[
            np.argmax([leaf.indicator(A) for leaf in leaves], axis=0)
        ]

    def np_extrema(self, arr):
        """Return the extremas of the array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Split the population based on random criterion."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            column_values = self.explanatory[:, feature][node.sub_population]
            feature_min, feature_max = self.np_extrema(column_values)
            diff = feature_max-feature_min
        x = self.rng.uniform()
        threshold = (1-x)*feature_min + x*feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Create a leaf for the given subpopulation."""
        leaf_child = Leaf(value=node.depth + 1, depth=node.depth + 1)
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create an internal node for the given sub_population."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Fit a single node."""
        node.feature, node.threshold = self.random_split_criterion(node)

        # Split population (SAME as Decision_Tree)
        left_population = node.sub_population \
            & (self.explanatory[:, node.feature] > node.threshold)
        right_population = node.sub_population \
            & (self.explanatory[:, node.feature] <= node.threshold)

        # Is left node a leaf? - DIFFERENT from Decision_Tree
        # In Decision_Tree: we check for class purity, min_pop, and max_depth
        # In Isolation_Tree: we ONLY check for max_depth and min_pop
        is_left_leaf = (node.depth + 1 >= self.max_depth) \
            or (np.sum(left_population) <= self.min_pop)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf? - DIFFERENT from Decision_Tree
        # Same logic as left - only depth and population size matter
        is_right_leaf = (node.depth + 1 >= self.max_depth) \
            or (np.sum(right_population) <= self.min_pop)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """Fit the model."""
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype=bool)

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : { self.depth()       }
    - Number of nodes           : { self.count_nodes() }
    - Number of leaves          : { self.count_nodes(only_leaves=True) }""")
