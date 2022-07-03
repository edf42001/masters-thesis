import graphviz

from symbolic_stochastic_domains.predicate_tree import PredicateTree, Node
from symbolic_stochastic_domains.predicates_and_objects import PredicateType

from symbolic_stochastic_domains.symbolic_utils import context_matches


def plot_predicate_tree(tree: PredicateTree, graph: graphviz.Digraph):
    # Plot the node and any edges connected to them, negative edges shown in red, radial view
    # Step one: Add all the nodes to the plot
    for node in tree.nodes:
        graph.node(node.object_name, node.object_name)

        for prop, value in node.properties.items():
            print(prop, value)
            graph.node(node.object_name + str(prop), str(value))

    # Next, connect all the edges
    for node in tree.nodes:
        for edge in node.edges:
            graph.edge(node.object_name, edge.to_node.object_name, str(edge.type)[14:])
        for edge in node.negative_edges:
            graph.edge(node.object_name, edge.to_node.object_name, str(edge.type)[14:], color="red")
        for prop, value in node.properties.items():
            graph.edge(node.object_name, node.object_name + str(prop), str(prop)[14:], color=("" if value else "red"))


if __name__ == "__main__":
    tree = PredicateTree()
    tree.add_node("taxi0")
    tree.add_node("wall0")
    tree.add_edge("taxi0", "wall0", PredicateType.TOUCH_RIGHT2D)

    context = PredicateTree()
    context.add_node("taxi0")

    print(f"Context matches: {context_matches(context, tree)}")



    # graph = graphviz.Digraph()
    # graph.engine = 'neato'
    # plot_predicate_tree(tree, graph)
    # graph.view()
