import numpy as np
from helpers.utils import flip_connection

from bayes_networks.node import Node


class BayesNetwork(object):
    def __init__(self, adj_matrix):
        """
        A bayes network is initialized from an adjacency matrix
        For now, assume all variables are booleans
        """

        # Store adj matrix and number of variables
        self.adj_matrix = adj_matrix
        self.n = adj_matrix.shape[0]

        # Store our nodes in a list for access
        self.nodes = []

        self.create_nodes_and_edge()
        self.bake()

    def create_nodes_and_edge(self):
        """Initializes nodes with their parents"""

        # Create the node objects
        for i in range(self.n):
            self.nodes.append(Node(i))

        # Go through adjacency matrix and add edges (in this case all edges are symmetric)
        for i in range(self.n):
            for j in range(self.n):
                if self.adj_matrix[i, j]:
                    node1 = self.nodes[i]
                    node2 = self.nodes[j]

                    node1.add_parent(node2)

    def bake(self):
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

            # TODO: do this effeciently with numpy slicing
            for row in data:
                # First extract prev state, then extract the parent node values
                parent_node_values = row[0][parents]

                # Extract the next state, then the resulting node value
                result_node_value = row[2][i]

                cpt_index = np.hstack((parent_node_values, result_node_value))

                # Add one to an observation of this particular parent state variables + new state
                node.update_count(cpt_index)


if __name__ == "__main__":
    n = 3
    adj_matrix = np.eye(n)
    flip_connection(adj_matrix, 0, 1)

    network = BayesNetwork(adj_matrix)

    print("CPT for each node")
    print(network.get_node(0).cpt)
    print(network.get_node(1).cpt)
    print(network.get_node(2).cpt)

