import numpy as np
from typing import Tuple


def flip_connection(adj_matrix, i, j):
    # flips a connection from i to j, adding or removing an edge
    # Assumes symmetry in the graph structure (all edges bidirectional)
    adj_matrix[i, j] = 1 - adj_matrix[i, j]
    adj_matrix[j, i] = 1 - adj_matrix[j, i]


def random_connection(n: int) -> Tuple[int, int]:
    """
    Picks a random pair of indices indicating a connection on a graph
    We specifically do not care about self connections in this function
    """
    i = np.random.randint(0, n)
    j = i

    while j == i:
        j = np.random.randint(0, n)

    return i, j