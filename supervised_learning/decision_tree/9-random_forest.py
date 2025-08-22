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

    def fit_node(self, node):
        """method to training nodes"""
        # Extract sub-population mask (individuals reaching this node)
        sub_pop = node.sub_population
        y = self.target[sub_pop]

        # Check stopping conditions (leaf criteria)
        if (
                y.size < self.min_pop
                or node.depth == self.max_depth
                or np.all(y == y[0])  # all same class
        ):
            node.is_leaf = True
            node.value = np.bincount(y).argmax()  # most frequent class
            return

        # Otherwise: split this node
        node.feature, node.threshold = self.split_criterion(node)

        # Split individuals using vectorized masks
        left_population = sub_pop & (self.explanatory[:, node.feature] > node.threshold)
        right_population = sub_pop & (self.explanatory[:, node.feature] <= node.threshold)

        # Left child
        if (
                np.sum(left_population) < self.min_pop
                or node.depth + 1 == self.max_depth
                or np.all(self.target[left_population] == self.target[left_population][0])
        ):
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Right child
        if (
                np.sum(right_population) < self.min_pop
                or node.depth + 1 == self.max_depth
                or np.all(self.target[right_population] == self.target[right_population][0])
        ):
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """retrieving the leaf child in a decision tree"""
        y = self.target[sub_population]
        # majority class
        value = np.bincount(y).argmax() if y.size > 0 else 0
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """obtaining the child of the node"""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """model's performance calculator"""
        return np.mean(self.predict(test_explanatory) == test_target)

    def fit(self, explanatory, target, verbose=0):
        """method for making our dataset trainable"""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion # < --- to be defined later
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)  # < --- to be defined later

        self.update_predict()  # < --- defined in the previous task
        if verbose == 1:
            print(f"""  Training finished.
        - Depth                     : {self.depth()}
        - Number of nodes           : {self.count_nodes()}
        - Number of leaves          : {self.count_nodes(only_leaves=True)}
        - Accuracy on training data : {self.accuracy(self.explanatory, self.target)}""")  # < --- to be defined later

    def np_extrema(self, arr):
        """return extrema"""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """method to apply random split criterion"""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def possible_thresholds(self, node, feature):
        """This method considers possible thresholds"""
        values = np.unique((self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """criterion for Gini impurity for one feature"""
# Compute a numpy array of booleans Left_F of shape (n,t,c) where
#    -> n is the number of individuals in the sub_population corresponding to node
#    -> t is the number of possible thresholds
#    -> c is the number of classes represented in node
# such that Left_F[ i , j , k] is true iff
#    -> the i-th individual in node is of class k
#    -> the value of the chosen feature on the i-th individual
#                              is greater than the t-th possible threshold
# then by squaring and summing along 2 of the axes of Left_F[ i , j , k],
#                     you can get the Gini impurities of the putative left childs
#                    as a 1D numpy array of size t
#
# Then do the same with the right child
# Then compute the average sum of these Gini impurities
#
# Then  return the threshold and the Gini average  for which the Gini average is the smallest
        # Extract values & labels for individuals in this node
        X_node = self.explanatory[node.sub_population, feature]
        y_node = self.target[node.sub_population]
        classes = np.unique(y_node)

        # Possible thresholds for this feature
        thresholds = self.possible_thresholds(node, feature)
        if thresholds.size == 0:
            return np.nan, np.inf  # no split possible

        # Expand dims for broadcasting
        X_exp = X_node[:, None]  # shape (n,1)
        thr_exp = thresholds[None, :]  # shape (1,t)

        # Masks for left/right splits
        left_mask = X_exp <= thr_exp  # shape (n,t)
        right_mask = ~left_mask  # shape (n,t)

        # One-hot encode class membership
        Y_onehot = np.array([y_node == c for c in classes])  # shape (c,n)

        # ---- LEFT CHILD ----
        left_counts = Y_onehot[:, :, None] & left_mask[None, :, :]  # (c,n,t)
        left_counts = left_counts.sum(axis=1)  # (c,t)
        left_totals = left_counts.sum(axis=0)  # (t,)
        left_probs = left_counts / np.maximum(left_totals, 1)  # (c,t)
        gini_left = 1.0 - np.sum(left_probs ** 2, axis=0)  # (t,)

        # ---- RIGHT CHILD ----
        right_counts = Y_onehot[:, :, None] & right_mask[None, :, :]  # (c,n,t)
        right_counts = right_counts.sum(axis=1)  # (c,t)
        right_totals = right_counts.sum(axis=0)  # (t,)
        right_probs = right_counts / np.maximum(right_totals, 1)  # (c,t)
        gini_right = 1.0 - np.sum(right_probs ** 2, axis=0)  # (t,)

        # Weighted average Gini
        n = len(y_node)
        gini_avg = (left_totals * gini_left + right_totals * gini_right) / n  # (t,)

        # Best threshold = one with min gini
        j = np.argmin(gini_avg)
        return thresholds[j], gini_avg[j]

    def Gini_split_criterion(self, node):
        """criterion for Gini split operation"""
        X = np.array([self.Gini_split_criterion_one_feature(node, i) for i in range(self.explanatory.shape[1])])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]


class Random_Forest() :
    """This class is collection of
    decision trees to represent Random Forest"""
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0) :
        """Constructor of Random Forest class"""
        self.numpy_predicts  = []
        self.target          = None
        self.numpy_preds     = None
        self.n_trees         = n_trees
        self.max_depth       = max_depth
        self.min_pop         = min_pop
        self.seed            = seed

    def predict(self, explanatory):          #  <--    to be filled
        """Predict class labels for samples in explanatory data.
        For each sample, collects predictions from all trees in the forest
        and returns the most frequent prediction (mode)."""
        # Initialize an empty list to store predictions from individual trees
        all_predictions = []
        # Iterates through each tree's predict function
        # (stored in self.numpy_preds) and
        # gets predictions for the input explanatory data.
        for tree_predict_func in self.numpy_preds:
            tree_predictions = tree_predict_func(explanatory)
            all_predictions.append(tree_predictions)
        all_predictions = np.array(all_predictions)
        n_samples = explanatory.shape[0]
        final_predictions = np.zeros(n_samples, dtype=int) # Shape: (n_trees, n_samples)
        for i in range(n_samples):
            sample_predictions = all_predictions[:, i]
            values, counts = np.unique(sample_predictions, return_counts=True)
            mode_index = np.argmax(counts)
            final_predictions[i] = values[mode_index]
        return final_predictions

        # Calculate the mode (most frequent) prediction for each example

    def fit(self,explanatory,target,n_trees=100,verbose=0) :
        """method for making our tree trainable"""
        self.target      = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths           = []
        nodes            = []
        leaves           = []
        accuracies =[]
        for i in range(n_trees) :
            T = Decision_Tree(max_depth=self.max_depth, min_pop=self.min_pop,seed=self.seed+i)
            T.fit(explanatory,target)
            self.numpy_preds.append(T.predict)
            depths.append(    T.depth()                         )
            nodes.append(     T.count_nodes()                   )
            leaves.append(    T.count_nodes(only_leaves=True)   )
            accuracies.append(T.accuracy(T.explanatory,T.target))
        if verbose==1 :
            print(f"""  Training finished.
    - Mean depth                     : { np.array(depths).mean()      }
    - Mean number of nodes           : { np.array(nodes).mean()       }
    - Mean number of leaves          : { np.array(leaves).mean()      }
    - Mean accuracy on training data : { np.array(accuracies).mean()  }
    - Accuracy of the forest on td   : {self.accuracy(self.explanatory,self.target)}""")

    def accuracy(self, test_explanatory , test_target) :
        """method to find how accurate our prediction is"""
        return np.sum(np.equal(self.predict(test_explanatory), test_target))/test_target.size
