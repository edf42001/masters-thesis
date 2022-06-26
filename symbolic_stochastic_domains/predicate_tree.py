
class PredicateTree:
    """
    Stores the current state of the world as objects connected by predicates
    for example, taxi --In-- key, and also taxi --TouchLeft -- wall
    or taxi --TouchDown-- door --- door.open
    This can be continued for each object in the chain
    """
    def __init__(self):
        self.base_object = Node("taxi")

    def print(self):
        self.base_object.print()


class Node:
    def __init__(self, object_name):
        self.object_name = object_name

        self.edges = []  # List of edges

    def add_edge(self, edge):
        self.edges.append(edge)

    def print(self):
        for edge in self.edges:
            print(self.object_name, edge.type, edge.to_node.object_name)
            edge.to_node.print()


class Edge:
    def __init__(self, type, to_node):
        self.type = type
        self.to_node = to_node
