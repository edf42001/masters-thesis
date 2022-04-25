"""
This file is to test if we can learn the structure for which conditions matter
to a taxi moving in taxi world

"""

import numpy as np
import matplotlib.pyplot as plt

from bayes_networks.bayes_network_conditions_effects import BayesNetworkCondEffect
from markov_chain_monte_carlo import calculated_p_data_given_graph


NAMES_DICT = {0: "action", 1: "touch_n", 2: "touch_e", 3: "touch_s", 4: "touch_w", 5: "on_pickup",
              6: "on_dest", 7: "in_taxi", 8: "dx", 9: "dy"}
ACTION_DICT = {0: "N", 1: "E", 2: "S", 3: "W"}


def learn_structure(action: int, variable: int, data: np.ndarray):
    """
    Given an action (UDLR) and a variable (XY), learns which conditions
    actually affect the outcome probability
    """

    # Extract the data with the specified action (action is first column)
    data = data[data[:, 0] == action]

    # Print how many times this action was taken
    print("Action = {} data length".format(ACTION_DICT[action]), len(data))
    print()

    # Take a subset of the full data for testing purposes
    len_data_to_use = 150
    data = data[:len_data_to_use, :]

    # Extract only the data we care about so that the first 4 columns are the touch conditions and the
    # last is the dx or dy change
    data = data[:, (1, 2, 3, 4, variable)]

    # What each column corresponds to (conditions are always the same, only the variable dx or dy changes)
    names = ["touch_n", "touch_e", "touch_s", "touch_w"] + [NAMES_DICT[variable]]
    # Need to add arities because the x value can be -1, 0, or 1 (do we have to convert this to 0-2)?
    arities = [2, 2, 2, 2, 3]
    ins_or_outs = [True, True, True, True, False]  # False represents an output node

    # Do an experiment with all possibilities to see if we can find the most likely one
    # This experiment works: it says ['touch_e', 'touch_w'] -> dx: -227.2375 which is the lowest
    n = 5
    inputs = 4

    adj_matrices = []
    networks = []

    for i in range(2 ** inputs):
        # This generates binary numbers from 0 through 16
        # This represents all combinations of ways that the 4 conditions could effect
        # the output, dx
        connections = np.unravel_index(i, [2] * inputs)

        # Create the adjacency matrix indicating this
        adj_matrix = np.zeros((n, n), dtype=int)
        adj_matrix[:-1, -1] = connections

        adj_matrices.append(adj_matrix)

    # Just so we have names TODO: Could just get this from name dict
    for adj_matrix in adj_matrices:
        network = BayesNetworkCondEffect(adj_matrix, arities=arities, names=names, ins_or_outs=ins_or_outs)
        network.update_node_counts(data)
        networks.append(network)

    # Now, do an experiment where we run markov_chain_monte_carlo on this to see what it thinks is the most
    # Actually, because we have a small number of graphs, just calculate directly
    likelihoods = np.empty((len_data_to_use, len(adj_matrices)))
    for i in range(len_data_to_use):
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

    plt.figure()
    plt.plot(likelihoods)
    plt.legend(parents, loc="upper left")

    # Generate title and save figure
    title = "Action {} vs Variable {}".format(ACTION_DICT[action], NAMES_DICT[variable])
    plt.title(title)
    plt.savefig("figures/" + title)

    # Don't show so figure is saved faster
    # plt.show()


if __name__ == "__main__":
    # Read the data from the csv file
    # Format: action (URDL), conditions (wall up, l, d, r, on dest, not on dest, p in taxi), dx, dy, movement dir result
    # (URDLN) indicated with 0-4
    # dytpe is int (for now?) because we need to represent these in discrete cpt tables
    data = np.loadtxt("../../envs/taxi_world/movement_data.csv", delimiter=",", dtype=int)
    print("Raw data length", len(data))
    print()

    for action in [0, 1, 2, 3]:
        for variable in [8, 9]:
            learn_structure(action, variable, data)

    learn_structure(0, 8, data)
