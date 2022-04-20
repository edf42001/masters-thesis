"""
This file is to test if we can learn the structure for which conditions matter
to a taxi moving in taxi world

"""

import numpy as np
import matplotlib.pyplot as plt

from bayes_networks.bayes_network_conditions_effects import BayesNetworkCondEffect
from markov_chain_monte_carlo import calculated_p_data_given_graph, boolean_array_to_number, should_accept_graph_transition


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
data_to_use = 600
data = data[:data_to_use, :]

print("Action = up data length", len(data))
print()

# Adjacency matrix for which effects influence which variable. But, is an adjacency matrix the best way of
# representing this, or is it better ot split into input/output spaces.
adj_matrix = np.zeros((5, 5))

# The robot's up action is influence by walls to the left and right, also up
adj_matrix[0, 4] = 1
adj_matrix[1, 4] = 1
adj_matrix[2, 4] = 1

print("Adjacency Matrix:")
print(adj_matrix)

# What each column corresponds to
names = ["touch_n", "touch_e", "touch_s", "touch_w", "dx"]
# Need to add arities because the x value can be -1, 0, or 1 (do we have to convert this to 0-2)?
arities = [2, 2, 2, 2, 3]
ins_or_outs = [True, True, True, True, False]  # False represents an output node

network = BayesNetworkCondEffect(adj_matrix, arities=arities, names=names, ins_or_outs=ins_or_outs)
network.update_node_counts(data)

# Do an experiment with all possibilities to see if we can find the most likely one
# This experiment works: it says ['touch_e', 'touch_w'] -> dx: -227.2375 which is the lowest
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
    network = BayesNetworkCondEffect(adj_matrix, arities=arities, names=names, ins_or_outs=ins_or_outs)
    network.update_node_counts(data)
    networks.append(network)


for adj_matrix in adj_matrices:
    l = calculated_p_data_given_graph(adj_matrix, data)
    likelihoods.append(l)

for network, l in zip(networks, likelihoods):
    for node in network.nodes:
        if node.name == "dx":
            print("{} -> {}: {:.4f}".format([network.nodes[p].name for p in node.parent_idxs], node.name, l))


# Now, do an experiment where we run markov_chain_monte_carlo on this to see what it thinks is the most
# likely graph, as we give it more and more data. (This code is duplicated from markov_chain_monte_carlo.py)
# TODO: the graph tranistion probabilites are set to 1/3 in that file but that's fine because they are the same,
# although technically they should be 1/n

# Actually, because we have a small number of graphs, just calculate directly
likelihoods = np.empty((data_to_use, len(adj_matrices)))
for i in range(data_to_use):
    for j, adj_matrix in enumerate(adj_matrices):
        l = calculated_p_data_given_graph(adj_matrix, data[:i])
        likelihoods[i, j] = l

    likelihoods[i, :] -= np.mean(likelihoods[i, :])

# Generate names to use as a legend
parents = []
for network in networks:
    # The last node is the dx node
    parent_names = "+".join([network.nodes[p].name[-1] for p in network.nodes[-1].parent_idxs])
    parents.append(parent_names)

plt.plot(likelihoods)
plt.legend(parents, loc="upper left")
plt.show()


# Graph transitions here are just turning off or on one of the boolean conditions
# that affect the output variables (in the future, might also be connections between output variables)
def propose_graph_transition(adj_matrix: np.ndarray) -> np.ndarray:
    n_input = adj_matrix.shape[0] - 1  # 1 is the output
    new_adj_matrix = adj_matrix.copy()
    random = np.random.randint(0, n_input)

    # Flip the bit in the last column
    new_adj_matrix[random, -1] = 1 - new_adj_matrix[random, -1]
    return new_adj_matrix


