from typing import Tuple
import numpy as np


class DecisionNode(object):
    """
    A decision tree is made of decision nodes. Nodes can either be split nodes or leaf nodes
    Nodes will have functions to help create the tree and to process incomming data
    """

    def __init__(self):
        self.is_leaf = False  # Is this node a leaf node
        self.split_idx = 0  # What condition index does this node split on?

        # Children decision nodes
        self.left_node: DecisionNode = None
        self.right_node: DecisionNode = None

        # Total instances classified into left and right, along with individual counts of [T, F]
        self.left_total = 0
        self.left_counts = [0, 0]
        self.right_total = 0
        self.right_counts = [0, 0]

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
