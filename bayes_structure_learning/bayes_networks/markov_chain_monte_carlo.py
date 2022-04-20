import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import math

from helpers.utils import flip_connection, random_connection
from process_training_data import process_sas_tuples
from bayes_networks.bayes_network import BayesNetwork
from bayes_networks.bayes_network_conditions_effects import BayesNetworkCondEffect

import cProfile
import pstats

"""
Markov Chain Monte Carlo is the Metropolis-Hasting algorithm for generating a posterior distribution
from the stationary distribution of a markov chian. If the states in our markov chain represent graphs,
and edges are transforming graph operation (i.e., add or remove and edge), and each edge has a transition probability,
then the stationary distribution is the limit as the random walk spends what percentage of its time in which graphs,
based on some available data. This will exactly match the probability of a graph given data P(G | D) if you follow
some specific rules for the transitions.
"""


def boolean_array_to_number(arr):
    return sum([2 ** i * v for i, v in enumerate(arr)])


def propose_graph_transition(adj_matrix: np.ndarray) -> np.ndarray:
    """Takes an incoming graph and modifies it, returns the new suggested graph structure"""
    n = adj_matrix.shape[0]
    i, j = random_connection(n)
    new_matrix = adj_matrix.copy()
    flip_connection(new_matrix, i, j)

    return new_matrix


def bayesian_scoring_metric(Xs: int, rs, Pas, a_ijk: np.ndarray, N_ijk: np.ndarray) -> float:
    # Log likelihood of total equation (I took the log otherwise the numbers go out of range)
    total = 0
    for i in range(Xs):
        # Iterate over unique combinations of assignments to parent's variables
        for j in range(Pas[i]):
            # Extract these terms for the equation
            a_ij = sum(a_ijk[i, j, :])
            N_ij = sum(N_ijk[i, j, :])

            total += math.lgamma(a_ij) - math.lgamma(a_ij + N_ij)

            # Iterate over next values this variable can take
            for k in range(rs[i]):
                total += math.lgamma(a_ijk[i, j, k] + N_ijk[i, j, k]) - math.lgamma(a_ijk[i, j, k])

    return total


def extract_gamma_equation_params_from_cpts(cpt_tables) -> float:
    # See https://www.dbmi.pitt.edu/sites/default/files/Kayaalp_1.pdf
    Xs = len(cpt_tables)  # Number of variables in the structure
    rs = np.empty((Xs, ), dtype=int)  # Number of distinct states each variable takes
    Pas = np.empty((Xs, ), dtype=int)  # Number of distinct combinations of the parent variables for each variable

    for i, table in enumerate(cpt_tables):
        shape = table.shape
        rs[i] = shape[-1]  # In my formulation, the output variable is the last index of the CPT
        Pas[i] = np.prod(shape[:-1])  # Each parent variable gets an index, so the product is the # of combinations

    # Store priors and counts for each variable
    a_ijk = np.ones((Xs, max(Pas), max(rs)))
    N_ijk = np.zeros((Xs, max(Pas), max(rs)))

    for i, table in enumerate(cpt_tables):
        for j in range(Pas[i]):
            parent_shape = table.shape[:-1]

            # Algorithm to extract index values from a number (works for more than binary)
            # Like this: https://www.instructables.com/How-to-Convert-Numbers-to-Binary/
            indices = []
            remainder = j
            for arity in parent_shape:
                indices.append(remainder % arity)
                remainder = int(remainder / arity)

            indices.reverse()

            for k in range(rs[i]):
                index = tuple(indices + [k])
                N_ijk[i, j, k] = table[index]

    return bayesian_scoring_metric(Xs, rs, Pas, a_ijk, N_ijk)


def calculated_p_data_given_graph(adj_matrix: np.ndarray, data) -> float:
    # Create the structure and the parameters for this bayesian network
    # We will use this in our calculations

    # TODO: We need these here: or, can we pass like a bayseian network as a template?
    # or better yet, this become a method in the bayesian network? Don't know why it isn't like that

    headers = ["touch_n", "touch_e", "touch_s", "touch_w", "dx"]
    arities = [2, 2, 2, 2, 3]
    ins_or_outs = [True, True, True, True, False]  # False represents an output node

    network = BayesNetworkCondEffect(adj_matrix, ins_or_outs=ins_or_outs, arities=arities, names=headers)
    network.update_node_counts(data)

    n = adj_matrix.shape[0]  # Number of "variables" (nodes) in this network

    # Bundle all the conditional probability tables together
    node_cpts = []
    for i in range(n):
        node_cpts.append(network.get_node(i).cpt)

    # Do the bayes scoring metric calculation. TODO: rename this function
    likelihood = extract_gamma_equation_params_from_cpts(node_cpts)

    return likelihood


def should_accept_graph_transition(adj_matrix: np.ndarray, new_adj_matrix: np.ndarray, data) -> bool:
    # The acceptance probability of going from graph Gk to Gi is equal to:
    # (P(D|Gi) * P(Gi) * Q(Gk|Gi)) / (P(D|Gk) * P(Gk) * Q(Gi|Gk)),
    # where P(D|G) is the likelihood of data given a graph (dirichlet?),
    # Q(Gk|Gi) is the transition probability (This depends on how transitions are defined. For example, in
    # acyclic graphs, going one way may have less options, due to the possibility of a cycle being created when an
    # edge is added.
    # P(G) is probably the prior on graphs
    # Yes, for some reason it goes k->i, not i->k

    # Assume uniform priors
    p_gk = 1
    p_gi = 1

    # We have no issues with cycles, so the transform probability is the same for each: 1/3, since in our case
    # for every graph with 3 nodes there are 3 edges that can be swapped (3 spots in bottom left triangle)
    q_gk_gi = 1.0 / 3.0
    q_gi_gk = 1.0 / 3.0

    # These ps are calculated in log likelihood. Thus, instead of dividing to find the ratio, we will subtract
    # and then do exp() to convert back to probability domain
    p_d_gk = calculated_p_data_given_graph(adj_matrix, data)
    p_d_gi = calculated_p_data_given_graph(new_adj_matrix, data)

    p_d_gik_ratio = math.exp(p_d_gi - p_d_gk)  # Equivalent to p_d_gi / p_d_gk if they were probabilities

    # Calculate acceptance probability, and return true or false accordingly
    # Original equation: (p_d_gi * p_gi * q_gk_gi) / (p_d_gk * p_gk * q_gi_gk)
    # Modified for log likelihood p_d_gi/k:
    acceptance_prob = p_d_gik_ratio * (p_gi * q_gk_gi) / (p_gk * q_gi_gk)
    return np.random.uniform() < acceptance_prob


if __name__ == "__main__":
    # Load our testing data with 0-1 connected, 2 by itself
    # We want to know the dynamics of the system when an action was not preformed.
    # Create a list of state, action, next state tuples
    states_actions = np.load("../data/training_data.npy").astype("int")
    data = process_sas_tuples(states_actions)[:300]

    n = 3  # Number of nodes
    adj_matrix = np.eye(n)  # Starting guess for adj

    # I want to experiment with what happens if it takes more than one step to find the correct matrix
    adj_matrix = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])

    # Iterations to sample for
    n_iterations = 300

    # Can use dict for high dimensions, hash table for low (wait, did I mean the opposite?)
    graph_counts = dict()
    # TODO <- is this equation correct for counting unique graphs? I think it may need to be 2^(n*(n-1)/2)
    graph_counts_history = np.zeros((n_iterations, 2 ** 3))

    # Profiling
    # profiler = cProfile.Profile()
    # profiler.enable()

    # Randomly sample 100 graphs in a random walk
    for i in range(n_iterations):
        # Propose new graph
        new_adj_matrix = propose_graph_transition(adj_matrix)

        # print("Proposed adjaceny matrix:")
        # print(new_adj_matrix)
        # print()
        # Whether or not to accept new graph
        if should_accept_graph_transition(adj_matrix, new_adj_matrix, data):
            # print("Accepted. New matrix:")
            # print(new_adj_matrix)
            # print()
            adj_matrix = new_adj_matrix

        # Extract the unique connections, which are the lower (or upper) triangular part without the diagonal,
        # and convert the 0/1 connections to a number with binary
        tri_indices = np.tril_indices(3, k=-1)
        unique_graph_values = adj_matrix[tri_indices]
        unique_graph_id = boolean_array_to_number(unique_graph_values)

        # Track counts of graphs
        if unique_graph_id not in graph_counts:
            graph_counts[unique_graph_id] = 1
        else:
            graph_counts[unique_graph_id] += 1

        # Record probabilities of each graph over time
        total = i + 1  # Total graphs so far: sum([v for v in graph_counts.values()])
        for k, v in graph_counts.items():
            graph_counts_history[i, int(k)] = v / (i+1)

    # Graph counts will now match P(G|D)
    # Plot these probabilities over time
    print("Graph counts:")
    print(graph_counts)

    # Profiling
    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.dump_stats('mcmc_stats.prof')

    plt.plot(graph_counts_history)
    plt.show()

