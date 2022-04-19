import numpy as np


class Node(object):
    """
    A node in a bayes network. Has parents, children, and a conditional probability table
    for each assignment of parents values
    """

    def __init__(self, idx, arity=2, name=""):
        self.idx = idx
        self.parent_idxs = []  # Indices of parents
        self.parent_arities = []  # State variable arities of parents

        # How many values does this node take?
        self.arity = arity

        # Node name, for debugging
        self.name = name

        # Create a conditional probability table. Arities are "how many values does a variable take"
        # i.e., if two parents each have a single boolean variable (arity 2), then we need a two by two table
        # None until network is baked
        self.cpt = None

    def arities(self):
        """Returns the arities for this node"""
        # All our nodes are boolean variables with arity two. It is a list, because in general could have
        # more than one state variable (Wait, can they?)
        return [self.arity]

    def add_parent(self, node):
        self.parent_idxs.append(node.idx)
        self.parent_arities.extend(node.arities())

    def bake(self):
        # Initialize the empty conditional probability table based on parent arities

        # Lets say node has one neighbor, and obviously its state depends on itself
        # it has 2 parents with a boolean variable, thus 2x2. But the output is 2, because the next state is a boolean
        # So we have to store counts for both true and false
        table_arities = self.parent_arities.copy()
        table_arities.extend(self.arities())

        self.cpt = np.zeros(table_arities)

    def update_count(self, table_indices):
        """Updates the count for a particular set of parent state values + observed new state value"""
        # Use tuple here to make this work as you expect (cpt[1, 1, 1]) as opposed to whatever cpt[[1, 1, 1]] does)
        self.cpt[tuple(table_indices)] += 1

