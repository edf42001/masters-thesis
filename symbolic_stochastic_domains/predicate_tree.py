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

        # These aren't used in representing state, as the closed world assumption says anything not mentioned
        # is assumed False. These are used for rule contexts, to represent (~TouchLeft(taxi, wall))
        self.negative_edges = []

        self.referenced_objects = set()  # Set of object names for nodes in this tree. TODO: What about nodes at different levels?

    def add_edge(self, edge):
        self.edges.append(edge)
        self.referenced_objects.add(edge.to_node.object_name)

    def add_negative_edge(self, edge):
        self.negative_edges.append(edge)
        self.referenced_objects.add(edge.to_node.object_name)

    def no_negative_edges_match(self, node):
        """Checks that this tree does not have any of the negative edges stored in node"""
        for negative_edge in node.negative_edges:
            # If there are any matches, that is failure, return false
            for edge in self.edges:
                if negative_edge.type == edge.type and negative_edge.to_node.object_name == edge.to_node.object_name:
                    return False

        return True

    def has_edge_with(self, type, name: str):
        """Checks if there is a positive or negative edge of the specified type and to object"""
        for edge in self.edges:
            if edge.type == type and edge.to_node.object_name == name:
                return True

        for edge in self.negative_edges:
            if edge.type == type and edge.to_node.object_name == name:
                return True

        return False

    def contains(self, node):
        """
        Returns true if the tree represented by the node is a subgraph of this node tree
        And, there are no matches for negative edges
        """
        for edge in node.edges:  # For every edge, we need to find a matching edge
            found = False
            for edge2 in self.edges:
                # A matching edge is defined as same predicate type and end object. Furthermore,
                # The node at the end's graph must also be contained
                if (
                    edge.type == edge2.type and
                    edge.to_node.object_name == edge2.to_node.object_name and
                    edge2.to_node.contains(edge.to_node)
                ):
                    found = True  # TODO: Could put a break here?

            # If we couldn't find anything, return failure. Otherwise, keep checking
            if not found:
                return False

        # Next, verify there are no negative edge matches
        if not self.no_negative_edges_match(node):
            return False

        # Otherwise, success if we get down here
        return True

    def copy(self, node):
        # Copy node into ourself
        for edge in node.edges:
            to_node = Node(edge.to_node.object_name)  # Create a copy of the end node
            to_node.copy(edge.to_node)  # Recursively copy that node to match our copy
            self.add_edge(Edge(edge.type, to_node))  # Add that edge

        for edge in node.negative_edges:
            # No need for recursion, as negative edges can't recurse because they mean "there is NOT an object there"
            self.add_negative_edge(Edge(edge.type, Node(edge.to_node.object_name)))

    def str_helper(self):
        ret = ""

        for edge in self.edges:
            ret += f"{self.object_name}-{edge}"
            ret += "," if len(edge.to_node.edges) == 0 else ""  # Indicate the ends of chains of objects
            # Add in negative edges here. There is no recursion, because they represent not interacting with an object
            if len(self.negative_edges) > 0:
                # Extra dot on end for same reason, otherwise it gets chopped
                ret += ", ".join(f"~{self.object_name}-{edge}" for edge in self.negative_edges) + "...."
            ret += " " + edge.to_node.str_helper()

        # We needed to put negative edges in the for loop so they'd appear in the right place, but that doesn't
        # work if there are no edges, so include them here in that case
        if len(self.negative_edges) > 0 and len(self.edges) == 0:
            # Have to add two dots at the end because we chop two off because of trailing commas. May need to rethink
            ret += ", ".join([f"~{self.object_name}-{edge}" for edge in self.negative_edges]) + ".."

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
