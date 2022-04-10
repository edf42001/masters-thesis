import numpy as np
import graphviz
import time
# Experimenting with visualizing graphs based on adjacency matrices


def flip_connection(adj_matrix, i, j):
    # flips a connection from i to j, adding or removing an edge
    # Assumes symmetry in the graph structure (all edges bidirectional)
    adj_matrix[i, j] = 1 - adj_matrix[i, j]
    adj_matrix[j, i] = 1 - adj_matrix[j, i]


def make_graph_plot(adj_matrix: np.core.multiarray, graph: graphviz.Graph):
    # Plots a graph from an adjacency matrix

    # Create n nodes
    n = adj_matrix.shape[0]
    for i in range(n):
        graph.node(str(i), str(i))  # id is the string number, display text is the number. Pin the node so they don't move

    # Connect the nodes (Ignore diagonal (don't draw that they're connected to themselves), and don't double count)
    for i in range(n):
        for j in range(i+1, n):
            if adj_matrix[i, j] == 1:
                graph.edge(str(i), str(j))


def random_connection(n):
    # Returns to random indices indicating a connection
    i = np.random.randint(0, n)
    j = i

    while j == i:
        j = np.random.randint(0, n)

    return i, j


if __name__ == "__main__":
    n = 5
    adj_matrix = np.eye(n)

    # Plot and show the graph structure
    for _ in range(20):
        # Flip some random connections
        i, j = random_connection(n)
        flip_connection(adj_matrix, i, j)

        graph = graphviz.Graph()  # The graph object (Digraph is for when you want directional arrows)
        make_graph_plot(adj_matrix, graph)
        graph.view()
        time.sleep(0.2)

        print(i, j)
