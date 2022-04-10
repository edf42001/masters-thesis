import numpy as np
import math

from helpers.utils import flip_connection
from process_training_data import process_sas_tuples
from bayes_networks.markov_chain_monte_carlo import calculated_p_data_given_graph
from bayes_networks.bayes_network import BayesNetwork


def test_gamma_function_math():
    a_ijk = np.ones((3, 2, 2))
    N_ijk = np.zeros((3, 2, 2))

    N_ijk[0, 0, 0] = 334
    N_ijk[0, 0, 1] = 0
    N_ijk[0, 1, 0] = 66
    N_ijk[0, 1, 1] = 600

    N_ijk[1, 0, 0] = 481
    N_ijk[1, 0, 1] = 0
    N_ijk[1, 1, 0] = 47
    N_ijk[1, 1, 1] = 472

    N_ijk[2, 0, 0] = 608
    N_ijk[2, 0, 1] = 0
    N_ijk[2, 1, 0] = 31
    N_ijk[2, 1, 1] = 361

    # Log likelihood of total equation
    total = 0
    for i in range(3):
        for j in range(2):
            # Extract these terms for the equation
            a_ij = sum(a_ijk[i, j, :])
            N_ij = sum(N_ijk[i, j, :])

            print(i, j, a_ij, N_ij)

            total += math.lgamma(a_ij) - math.lgamma(a_ij + N_ij)

            for k in range(2):
                total += math.lgamma(a_ijk[i, j, k] + N_ijk[i, j, k]) - math.lgamma(a_ijk[i, j, k])

    print(total)


def test_gamma_function_math_2():
    Xs = 3
    rs = [2, 2, 2]
    Pas = [4, 4, 2]

    a_ijk = np.ones((Xs, max(Pas), max(rs)))
    N_ijk = np.zeros((Xs, max(Pas), max(rs)))

    N_ijk[0, 0, 0] = 249
    N_ijk[0, 0, 1] = 0
    N_ijk[0, 1, 0] = 85
    N_ijk[0, 1, 1] = 0
    N_ijk[0, 2, 0] = 50
    N_ijk[0, 2, 1] = 182
    N_ijk[0, 3, 0] = 16
    N_ijk[0, 3, 1] = 418

    N_ijk[1, 0, 0] = 247
    N_ijk[1, 0, 1] = 0
    N_ijk[1, 1, 0] = 17
    N_ijk[1, 1, 1] = 68
    N_ijk[1, 2, 0] = 232
    N_ijk[1, 2, 1] = 0
    N_ijk[1, 3, 0] = 30
    N_ijk[1, 3, 1] = 404

    N_ijk[2, 0, 0] = 608
    N_ijk[2, 0, 1] = 0
    N_ijk[2, 1, 0] = 31
    N_ijk[2, 1, 1] = 361

    # Log likelihood of total equation
    total = 0
    for i in range(Xs):
        # Iterate over unique combinations of assignments to parent's variables
        for j in range(Pas[i]):
            # Extract these terms for the equation
            a_ij = sum(a_ijk[i, j, :])
            N_ij = sum(N_ijk[i, j, :])

            print(i, j, a_ij, N_ij)

            total += math.lgamma(a_ij) - math.lgamma(a_ij + N_ij)
            print(total)

            # Iterate over next values this variable can take
            for k in range(rs[i]):
                total += math.lgamma(a_ijk[i, j, k] + N_ijk[i, j, k]) - math.lgamma(a_ijk[i, j, k])
                print(total)

    print(total)
    print()


if __name__ == "__main__":    # Load data
    states_actions = np.load("../data/training_data.npy").astype("int")

    # We want to know the dynamics of the system when an action was not preformed.
    # Create a list of state, action, next state tuples
    sas_pairs = process_sas_tuples(states_actions)
    new_sas_pairs = sas_pairs[:300]

    # Create and train the bayes network
    n = 3

    # THis is the network the data was generated with
    real_adj_matrix = np.eye(n)
    flip_connection(real_adj_matrix, 0, 1)

    # Default, no connections network
    default_adj_matrix = np.eye(n)

    prob1 = calculated_p_data_given_graph(real_adj_matrix, new_sas_pairs)
    prob2 = calculated_p_data_given_graph(default_adj_matrix, new_sas_pairs)

    # Do a test of our calculations with the gamma function
    # test_gamma_function_math()
    # test_gamma_function_math_2()
    print("Real score metric: {:.2f}".format(prob1))
    print("Default score metric: {:.2f}".format(prob2))
