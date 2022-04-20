"""
This file is to test what happens when a output node has no parents
Currently my math returns a likelihood of 100%
"""

import numpy as np

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
data = data[:500, :]

print("Action = up data length", len(data))
print()

# What each column corresponds to
headers = ["touch_n", "touch_e", "touch_s", "touch_w", "dx"]

# Adjacency matrix for which effects influence which variable. But, is an adjacency matrix the best way of
# representing this, or is it better ot split into input/output spaces.
adj_matrix = np.zeros((5, 5))

print("Adjacency Matrix:")
print(adj_matrix)

# Need to add arities because the x value can be -1, 0, or 1 (do we have to convert this to 0-2)?
# Currently am relying on the fact that -1 wraps around to the last value
arities = [2, 2, 2, 2, 3]
ins_or_outs = [True, True, True, True, False]  # False represents an output node
network = BayesNetworkCondEffect(adj_matrix, arities=arities, names=headers, ins_or_outs=ins_or_outs)
network.update_node_counts(data)

for node in network.nodes:
    print(node.cpt)

l = calculated_p_data_given_graph(adj_matrix, data)
print(l)


