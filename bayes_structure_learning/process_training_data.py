"""In this file we load training data to do experiments on learning bayesian network structure and parameters"""
import numpy as np

from helpers.utils import flip_connection
from bayes_networks.bayes_network import BayesNetwork


def process_sas_tuples(states_actions, noop_action=3):
    pairs = []

    for i in range(len(states_actions)-1):
        row = states_actions[i, :]
        action = row[3]

        if action == noop_action:
            sas = (row[0:3], action, states_actions[i+1, 0:3])
            pairs.append(sas)

    return pairs


if __name__ == "__main__":
    states_actions = np.load("data/training_data.npy").astype("int")

    # We want to know the dynamics of the system when an action was not preformed.
    # Create a list of state, action, next state tuples
    sas_pairs = process_sas_tuples(states_actions)

    # Create and train the bayes network
    n = 3
    adj_matrix = np.eye(n)
    flip_connection(adj_matrix, 0, 1)

    network = BayesNetwork(adj_matrix)

    print("Data to test update with")
    print("Length " + str(len(sas_pairs)))
    network.update_node_counts(sas_pairs)

    print("Node cpt tables")
    print(network.get_node(0).cpt)
    print(network.get_node(1).cpt)
    print(network.get_node(2).cpt)

