"""
This file is to test if the method we use can learn that x and y depend on each other
in the sense that both can not change at the same time
"""

import numpy as np
import matplotlib.pyplot as plt

from bayes_networks.bayes_network_conditions_effects import BayesNetworkCondEffect
from markov_chain_monte_carlo import calculated_p_data_given_graph


NAMES_DICT = {0: "action", 1: "touch_n", 2: "touch_e", 3: "touch_s", 4: "touch_w", 5: "on_pickup",
              6: "on_dest", 7: "in_taxi", 8: "dx", 9: "dy"}
ACTION_DICT = {0: "N", 1: "E", 2: "S", 3: "W"}


def learn_structure(action: int, data: np.ndarray):
    """
    Given an action (UDLR), see if we can learn that x and y are connected to each other
    """

    # Extract the data with the specified action (action is first column)
    data = data[data[:, 0] == action]

    # Print how many times this action was taken
    print("Action = {} data length".format(ACTION_DICT[action]), len(data))
    print()

    # Take a subset of the full data for testing purposes
    len_data_to_use = 25
    data = data[500:len_data_to_use+500, :]

    # Extract only the data we care about so that the first 4 columns are the touch conditions and the
    # last is the dx and dy change
    data = data[:, (1, 2, 3, 4, 8, 9)]

    # What each column corresponds to
    names = ["touch_n", "touch_e", "touch_s", "touch_w"] + [NAMES_DICT[8], NAMES_DICT[9]]
    # Need to add arities because the x value can be -1, 0, or 1 (do we have to convert this to 0-2)?
    arities = [2, 2, 2, 2, 3, 3]
    ins_or_outs = [True, True, True, True, False, False]  # False represents an output node

    n = 6
    inputs = 4

    adj_matrices = []
    networks = []

    for i in range(2 ** inputs):
        for j in range(2 ** inputs):
            for k in range(2):
                # This generates binary numbers from 0 through 16
                # This represents all combinations of ways that the 4 conditions could effect the output
                # Then another for loop for the ys, then another for whether or not dx and dy are connected
                connections_x = np.unravel_index(i, [2] * inputs)
                connections_y = np.unravel_index(j, [2] * inputs)

                # Create the adjacency matrix indicating this
                adj_matrix = np.zeros((n, n), dtype=int)
                adj_matrix[0:4, -2] = connections_x
                adj_matrix[0:4, -1] = connections_y
                adj_matrix[-2, -1] = k

                adj_matrices.append(adj_matrix)

    # Test the one matrix that is breaking my code
    adj_matrices = adj_matrices[0:]

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

    # Extract the top 10 best guesses to make looking at the plot easier
    last_likelihoods = likelihoods[-1]
    n = 10
    top_10_indices = np.argpartition(last_likelihoods, -n)[-n:]
    likelihoods = likelihoods[:, top_10_indices]

    # Generate names to use as a legend (only for the top ones)
    parents = []
    for i in top_10_indices:
        network = networks[i]
        # The last node is the dx node. Use -1 to get rid of the touch_ part and also combine x and y values
        parent_names = "+".join([network.nodes[p].name[-1] for p in network.nodes[-2].parent_idxs]) + "-" + \
                       "+".join([network.nodes[p].name[-1] for p in network.nodes[-1].parent_idxs])
        parents.append(parent_names)

    plt.figure()
    plt.plot(likelihoods)
    plt.legend(parents, loc="upper left")

    # Generate title and save figure
    # title = "Action {} vs Variable {}".format(ACTION_DICT[action], NAMES_DICT[variable])
    title = "Action N vs dx and dy"
    plt.title(title)
    plt.savefig("figures/" + title)

    # Don't show so figure is saved faster
    plt.show()


if __name__ == "__main__":
    # Read the data from the csv file
    # Format: action (URDL), conditions (wall up, l, d, r, on dest, not on dest, p in taxi), dx, dy, movement dir result
    # (URDLN) indicated with 0-4
    # dytpe is int (for now?) because we need to represent these in discrete cpt tables
    data = np.loadtxt("../../envs/taxi_world/movement_data.csv", delimiter=",", dtype=int)
    print("Raw data length", len(data))
    print()

    action = 0  # north
    learn_structure(0, data)
