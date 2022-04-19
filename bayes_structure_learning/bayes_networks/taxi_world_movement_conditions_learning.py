"""
This file is to test if we can learn the structure for which conditions matter
to a taxi moving in taxi world

"""

import numpy as np

from bayes_networks.bayes_network import BayesNetwork
from bayes_networks.bayes_network_conditions_effects import BayesNetworkCondEffect
from markov_chain_monte_carlo import calculated_p_data_given_graph


# Read the data from the csv file
# Format: action (URDL), conditions (wall up, l, d, r, on dest, not on dest, p in taxi), dx, dy, movement dir result
# (URDLN) indicated with 0-4
# dytpe is int (for now?) because we need to represent these in discrete cpt tables
data = np.loadtxt("../../envs/taxi_world/movement_data.csv", delimiter=",", dtype=int)
print("Raw data length", len(data))
print()

# Ignore the on destination conditions for now
# Take only the 4 wall conditions, along with the dx
data = data[:, [0, 1, 2, 3, 4, 8]]

# Data where the action was up
data = data[data[:, 0] == 0]
data = data[:, 1:]  # We can remove the action since it is always 0


# Take a subset just for testing purposes
data = data[:1000, :]

print("Action = up data length", len(data))
print()

# What each column corresponds to
headers = ["touch_n", "touch_e", "touch_s", "touch_w", "dx"]

# Adjacency matrix for which effects influence which variable. But, is an adjacency matrix the best way of
# representing this, or is it better ot split into input/output spaces.
adj_matrix = np.zeros((5, 5))

# The robot's up action is influence by walls to the left and right, also up
adj_matrix[0, 4] = 1
adj_matrix[1, 4] = 1
adj_matrix[2, 4] = 1

print("Adjacency Matrix:")
print(adj_matrix)

# Need to add arities because the x value can be -1, 0, or 1 (do we have to convert this to 0-2)?
arities = [2, 2, 2, 2, 3]
network = BayesNetworkCondEffect(adj_matrix, arities=arities, names=headers)
network.update_node_counts(data)

for node in network.nodes:
    print(node.cpt)


# # Data where the action was up and there was no wall to the left or right or up
# data = data[np.all(data[:, [1, 2, 4]] == 0, axis=1)]
# # data = data[np.all(data[:, [3]] == 1, axis=1)]
# print(data, len(data))
#
# hist = np.histogram(data[:, -1], density=True, bins=3)
# print(hist)

# DO an experiment with all possibilities to see if we can find the most likely one
n = 5
inputs = 4
outputs = 1

adj_matrices = []
networks = []
likelihoods = []

for i in range(2 ** inputs):
    # This generates binary numbers from 0 through 16
    # This represents all combinations of ways that the 4 conditions could effect
    # the output, dx
    connections = np.unravel_index(i, [2] * inputs)

    # Create the adjacency matrix indicating this
    adj_matrix = np.zeros((n, n), dtype=int)
    adj_matrix[:-1, -1] = connections

    adj_matrices.append(adj_matrix)

for adj_matrix in adj_matrices:
    network = BayesNetworkCondEffect(adj_matrix, arities, headers)
    network.update_node_counts(data)
    networks.append(network)

for network in networks:
    for node in network.nodes:
        if node.name == "dx":
            print("{} -> {}".format([network.nodes[p].name for p in node.parent_idxs], node.name))

for adj_matrix in adj_matrices:
    l = calculated_p_data_given_graph(adj_matrix, data)
    likelihoods.append(l)

for network, l in zip(networks, likelihoods):
    for node in network.nodes:
        if node.name == "dx":
            print("{} -> {}: {:.4f}".format([network.nodes[p].name for p in node.parent_idxs], node.name, l))