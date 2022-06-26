class PredicateTree:
    """
    Stores the current state of the world as objects connected by predicates
    for example, taxi --In-- key, and also taxi --TouchLeft -- wall
    or taxi --TouchDown-- door --- door.open
    This can be continued for each object in the chain
    """
    def __init__(self):
        self.base_object = Node("taxi")

        # I believe strings are always unique and identifiable to to the order we add nodes to the tree
        # Thus, we use this for hashing and equality. We save it because it is expensive to create strings
        self.str_repr = self.__str__()

    def copy(self):
        """Create a copy of this tree"""
        ret = PredicateTree()
        ret.base_object.copy(self.base_object)
        return ret

    def __str__(self):
        return str(self.base_object)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.str_repr == other.str_repr


class Node:
    """A node in the tree is an object, connected one-directionally to other objects via predicate edges"""
    def __init__(self, object_name):
        self.object_name = object_name

        self.edges = []  # List of edges

    def add_edge(self, edge):
        self.edges.append(edge)

    def copy(self, node):
        # Copy node into ourself
        for edge in node.edges:
            to_node = Node(edge.to_node.object_name)  # Create a copy of the end node
            to_node.copy(edge.to_node)  # Recursively copy that node to match our copy
            self.add_edge(Edge(edge.type, to_node))  # Add that edge

    def str_helper(self):
        ret = ""

        for edge in self.edges:
            ret += f"{self.object_name}-{edge}"
            ret += "," if len(edge.to_node.edges) == 0 else ""  # Indicate the ends of chains of objects
            ret += " "
            ret += edge.to_node.str_helper()

        return ret

    def __str__(self):
        return "[" + self.str_helper()[:-2] + "]"  # There's a trailing space and comma on str_helper we want to remove

    def __repr__(self):
        return self.__str__()


class Edge:
    """An edge consists of a predicate type and an end node"""
    def __init__(self, type, to_node):
        self.type = type
        self.to_node = to_node

    def __str__(self):
        return f"{str(self.type)[14:]}-{self.to_node.object_name}"  # Remove the PredicateType. prefix from the enum

    def __repr__(self):
        return self.__str__()
