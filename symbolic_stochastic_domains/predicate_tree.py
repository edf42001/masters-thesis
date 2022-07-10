class PredicateTree:
    """
    Stores the current state of the world as objects connected by predicates
    for example, taxi --In-- key, and also taxi --TouchLeft -- wall
    or taxi --TouchDown-- door --- door.open
    This can be continued for each object in the chain
    """
    def __init__(self):
        self.nodes = []  # List of nodes
        self.node_lookup = dict()  # Dictionary mapping string ids to nodes

        self.base_object = None

        # I believe strings will always be unique and identifiable and can be used for comparison
        # Could have a "finalize" function
        self.str_repr = None

        self.referenced_objects = set()  # Set of objects referred to (deictically) by this tree.

    def add_node(self, name):
        # Check for duplicates
        assert name not in self.node_lookup, f"already have node {name}"

        new_node = Node(name, 0)
        self.nodes.append(new_node)
        self.node_lookup[name] = new_node

        # If this was the first, store it as the base.
        if len(self.nodes) == 1:
            self.base_object = new_node

    def add_edge(self, from_name, to_name, type, negative=False):
        # Check values are there
        assert from_name in self.node_lookup and to_name in self.node_lookup, f"Did not have {from_name} or {to_name}"

        from_node = self.node_lookup[from_name]
        to_node = self.node_lookup[to_name]

        edge = Edge(type)

        # Create the main pointer to the node, and the helper pointers pointing backwards
        edge.to_node = to_node
        edge.from_node = from_node
        to_node.to_edges.append(edge)

        if negative:
            from_node.negative_edges.append(edge)
        else:
            from_node.edges.append(edge)

        # Now that the object is being referenced, we can add how to the list of references objects
        self.referenced_objects.add(f"{from_name[:-1]}-{str(type)[14:]}-{to_name[:-1]}")

    def add_property(self, node_name, type, value):
        self.node_lookup[node_name].properties[type] = value

    def copy(self):
        """Create a copy of this tree"""
        ret = PredicateTree()

        for node in self.nodes:
            ret.add_node(node.object_name)

            for k, v in node.properties.items():
                ret.add_property(node.object_name, k, v)

        for node in self.nodes:
            for edge in node.edges:
                ret.add_edge(node.object_name, edge.to_node.object_name, edge.type)

            for edge in node.negative_edges:
                ret.add_edge(node.object_name, edge.to_node.object_name, edge.type, negative=True)

        return ret

    def print_tree(self):
        for node in self.nodes:
            ret = ""
            ret += node.object_name + ": "

            for edge in node.edges:
                ret += f"{edge} "
            for edge in node.negative_edges:
                ret += f"{edge} {edge.to_node.object_name}"

            print(ret)

    def __str__(self):
        return str(self.base_object)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        # Convert to strings to compare
        if self.str_repr is None:
            self.str_repr = self.__str__()

        if other.str_repr is None:
            other.str_repr = other.__str__()

        return self.str_repr == other.str_repr


class Node:
    """A node in the tree is an object, connected one-directionally to other objects via predicate edges"""
    def __init__(self, object_name, object_id):
        self.object_name = object_name
        self.object_id = object_id

        self.edges = []  # List of edges going out of this node
        self.to_edges = []  # List of edges going into this node. Used for traveling back up the chain

        # These aren't used in representing state, as the closed world assumption says anything not mentioned
        # is assumed False. These are used for rule contexts, to represent (~TouchLeft(taxi, wall))
        self.negative_edges = []

        # List of object properties, such as OPEN for locks, as {PredicateType: value}
        self.properties = dict()

    def no_negative_edges_match(self, node):
        """Checks that this tree does not have any of the negative edges stored in node"""
        for negative_edge in node.negative_edges:
            # If there are any matches, that is failure, return false
            for edge in self.edges:
                if negative_edge.type == edge.type and negative_edge.to_node.object_name[:-1] == edge.to_node.object_name[:-1]:  # Remove the id from the node, we only care about class type
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
                # The node at the end's graph must also be contained.
                if (
                    edge.type == edge2.type and
                    edge.to_node.object_name[:-1] == edge2.to_node.object_name[:-1] and  # Remove the id from the node, we only care about class type here
                    edge2.to_node.contains(edge.to_node)
                ):
                    found = True  # TODO: Could put a break here?

            # If we couldn't find anything, return failure. Otherwise, keep checking
            if not found:
                return False

        # Next, verify there are no negative edge matches
        if not self.no_negative_edges_match(node):
            return False

        # Finally, make sure it has all positive properties and no negative properties
        for key, value in node.properties.items():
            if value and key not in self.properties:  # If the context requires a positive value but we don't have it
                return False

            if not value and key in self.properties:  # If the context requires a negative value, but we have it
                return False

        # Otherwise, success if we get down here
        return True

    def str_helper(self):
        ret = ""

        for edge in self.edges:
            ret += f"{self.object_name}-{edge}"
            ret += "," if len(edge.to_node.edges) == 0 else ""  # Indicate the ends of chains of objects
            # Add in negative edges here. There is no recursion, because they represent not interacting with an object
            if len(self.negative_edges) > 0:
                # Extra dot on end for same reason, otherwise it gets chopped
                ret += ", ".join(f"~{self.object_name}-{edge}" for edge in self.negative_edges) + ","

            # Add properties in here
            if len(self.properties) > 0:
                ret += ", ".join(f"{str(key)[14:]}-{value}" for key, value in self.properties) + ", "

            ret += " " + edge.to_node.str_helper()

        # We needed to put negative edges in the for loop so they'd appear in the right place, but that doesn't
        # work if there are no edges, so include them here in that case
        if len(self.negative_edges) > 0 and len(self.edges) == 0:
            # Have to add two dots at the end because we chop two off because of trailing commas. May need to rethink
            ret += ", ".join([f"~{self.object_name}-{edge}" for edge in self.negative_edges]) + ".."

        if len(self.properties) > 0 and len(self.edges) == 0:
            ret += ", ".join(f"{self.object_name}-{str(key)[14:]}-{value}" for key, value in self.properties.items()) + ", "

        return ret

    def __str__(self):
        return "[" + self.str_helper()[:-2] + "]"  # There's a trailing space and comma on str_helper we want to remove

    def __repr__(self):
        return self.__str__()


class Edge:
    """An edge consists of a predicate type and an end node"""
    def __init__(self, type):
        self.type = type  # Predicate type of the edge (TOUCH_DOWN, IN, etc)
        self.to_node = None  # Pointer to the node this edge is connected to
        self.from_node = None  # Where this edge came from. Used for traveling backwards up the chain

    def __str__(self):
        return f"{str(self.type)[14:]}-{self.to_node.object_name}"  # Remove the PredicateType. prefix from the enum

    def __repr__(self):
        return self.__str__()
