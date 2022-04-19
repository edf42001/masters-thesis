import numpy as np
from helpers.utils import flip_connection

from bayes_networks.node import Node


class BayesNetworkCondEffect(object):
    def __init__(self, adj_matrix, arities=None, names=None):
        """
        A bayes network is initialized from an adjacency matrix
        For now, assume all variables are booleans

        TODO: This one is designed for specifically conditions and effects. There must be some way to merge the
        two into one unified bayesian network framework
        """

        # Store adj matrix and number of variables
        self.adj_matrix = adj_matrix
        self.n = adj_matrix.shape[0]

        # Store our nodes in a list for access
        self.nodes = []

        self.create_nodes_and_edge(arities=arities, names=names)
        self.bake()

    def create_nodes_and_edge(self, arities=None, names=None):
        """Initializes nodes with their parents"""

        # Create the node objects
        for i in range(self.n):
            # Default arity and name, use passed values if given
            arity = 2
            name = str(i)
            if arities is not None:
                arity = arities[i]
            if names is not None:
                name = names[i]

            self.nodes.append(Node(i, arity=arity, name=name))

        # Go through adjacency matrix and add edges (in this case all edges are NOT symmetric)
        # THe order is [i, j] = [parent, child]
        for i in range(self.n):
            for j in range(self.n):
                if self.adj_matrix[i, j]:
                    parent = self.nodes[i]
                    child = self.nodes[j]

                    child.add_parent(parent)

    def bake(self):
        """
        Once we are done adding nodes, we know how big each one will be,
        so we can set the size of the cpt and start filling in data
        """
        for i in range(self.n):
            self.nodes[i].bake()

    def get_parents(self, idx: int):
        # Returns the neighbors (parent?) indices of this computer using the adjacency matrix
        # Nodes are parents of themselves
        return [j for j in range(self.n) if self.adj_matrix[j, idx]]

    def get_node(self, idx) -> Node:
        return self.nodes[idx]

    def update_node_counts(self, data):
        """This function does the bulk of the data processing"""

        # For each node, update its counts for parent values + transition
        for i, node in enumerate(self.nodes):
            parents = self.get_parents(i)

            # If no parents, nothing to do here (this is an input node (or just not affected by anything))
            if len(parents) == 0:
                continue

            # TODO: do this effeciently with numpy slicing
            for row in data:
                # First extract prev state, then extract the parent node values
                # The difference from the other one is that the other one had a tuple
                # representing the state, so this says row[parents] and the other was row[0][parents]
                parent_node_values = row[parents]

                # Extract the next state, then the resulting node value
                result_node_value = row[i]

                cpt_index = np.hstack((parent_node_values, result_node_value))

                # Add one to an observation of this particular parent state variables + new state
                node.update_count(cpt_index)

    def view_graph(self):
        """Uses graphviz to visualize this network"""
        pass


if __name__ == "__main__":
    pass
