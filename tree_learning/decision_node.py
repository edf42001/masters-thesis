from typing import Tuple
import numpy as np


class DecisionNode(object):
    """
    A decision tree is made of decision nodes. Nodes can either be split nodes or leaf nodes
    Nodes will have functions to help create the tree and to process incomming data
    """

    def __init__(self, depth=0):
        self.is_leaf = False  # Is this node a leaf node
        self.leaf_value = 0  # The value of this leaf (true, false, no change, +1, -1, etc)

        # How far down this is in the tree. Starts at the top at 0
        self.depth = depth

        self.split_idx = 0  # What condition index does this node split on?

        # Children decision nodes
        self.left_node: DecisionNode = None
        self.right_node: DecisionNode = None

        # Total instances classified into left and right, along with individual counts of [T, F]
        self.left_total = 0
        self.left_counts = [0, 0]
        self.right_total = 0
        self.right_counts = [0, 0]

    def recursively_split(self, data):
        """
        Build the decision tree by splitting the data until the entropy is 0
        i.e., every effect is explained
        """

        # If there is nothing left to split, this node is a leaf node
        left_trues = np.count_nonzero(data[:, -1])
        if left_trues == 0 or left_trues == len(data):
            self.is_leaf = True
            self.leaf_value = data[0, -1]  # They are all the same, so pick any value to represent
            return

        # Otherwise, mark this as not a leaf node, because when we call more than once on a tree it remembers
        self.is_leaf = False

        # Step 1: Find the best starting split
        self.split_idx = self.find_best_split(data)

        # Split the data there
        data_left, data_right = self.split_data(data, self.split_idx)

        # Recursively keep splitting
        depth = self.depth + 1
        self.left_node = DecisionNode(depth=depth)
        self.right_node = DecisionNode(depth=depth)

        self.left_node.recursively_split(data_left)
        self.right_node.recursively_split(data_right)

    def split_data(self, data, condition_idx) -> Tuple[np.ndarray, np.ndarray]:
        """
        Takes in data and splits it into two based on the condition idx
        The first N-1 cols of data are the inputs (x values, conditions) and the last column is the classification
        """

        # left is true, right is false
        data_left = data[data[:, condition_idx] == 1]
        data_right = data[data[:, condition_idx] == 0]

        # Calc statistics
        self.left_total = len(data_left)
        self.right_total = len(data_right)

        left_trues = np.count_nonzero(data_left[:, -1])
        right_trues = np.count_nonzero(data_right[:, -1])

        self.left_counts = [left_trues, self.left_total - left_trues]
        self.right_counts = [right_trues, self.right_total - right_trues]

        return data_left, data_right

    def entropy(self, c1: int, c2: int) -> float:
        """Calculates entropy of a binary split with c1 of 1 class and c2 of the other"""
        # Entropy = sum(-p*log2(p)) = -p1 * log2(p1) - p2 * log2(p2) =
        # -p1*log2(p1) - (1-p1)*log2(1-p1)

        # If entireley one side, return 0. This also applies to if both are 0
        if c1 == 0 or c2 == 0:
            return 0

        p = c1 / (c1 + c2)

        return -p * np.log2(p) - (1-p) * np.log2(1-p)

    def information_gain_of_split(self, data, condition_idx) -> float:
        """Given data and a split, finds the information gain"""
        start_entropy = 1  # TODO: Calc this from data

        # Make the split
        self.split_data(data, condition_idx)

        # Calculate left and right entropy
        left_entropy = self.entropy(self.left_counts[0], self.left_counts[1])
        right_entropy = self.entropy(self.right_counts[0], self.right_counts[1])

        # Weight the entropy by the relative number of elements
        total = self.left_total + self.right_total
        end_entropy = left_entropy * self.left_total / total + right_entropy * self.right_total / total

        # Entropy will have gone down, so we want to maximize this value
        return start_entropy - end_entropy

    def find_best_split(self, data) -> int:
        """Returns which split gives the highest information gain"""

        best_idx = -1
        best_gain = 0

        for condition_idx in range(data.shape[1] - 1):  # Remember that last column is the output
            gain = self.information_gain_of_split(data, condition_idx)
            if gain > best_gain:
                best_gain = gain
                best_idx = condition_idx

        return best_idx

    def print(self):
        if self.is_leaf:
            print("Leaf: +{}".format(self.leaf_value))
        else:
            print("Split: {}".format(self.split_idx))

        if self.left_node:
            self.left_node.print()

        if self.right_node:
            self.right_node.print()

    def predict(self, condition):
        """Given a condition, predict the effect"""

        # If leaf, return the value
        if self.is_leaf:
            return self.leaf_value

        # If true, go down the left branch, otherwise right
        if condition[self.split_idx]:
            return self.left_node.predict(condition)
        else:
            return self.right_node.predict(condition)

    def to_string(self) -> str:
        ret = ""

        if self.is_leaf:
            ret += "({})".format(self.leaf_value)
        else:
            ret += str(self.split_idx)

        if self.left_node:
            ret += self.left_node.to_string()

        if self.right_node:
            ret += self.right_node.to_string()

        return ret

    def __repr__(self):
        return self.to_string()
