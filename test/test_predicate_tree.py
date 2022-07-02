from symbolic_stochastic_domains.predicate_tree import PredicateTree
from symbolic_stochastic_domains.predicates_and_objects import PredicateType


if __name__ == "__main__":
    tree = PredicateTree()
    tree.add_node("taxi")
    tree.add_node("lock")
    tree.add_node("key")

    # tree.add_node("lock")
    tree.add_edge("taxi", "lock", PredicateType.TOUCH_DOWN2D)
    tree.add_edge("taxi", "key", PredicateType.IN)
    print(tree)
    tree.print_tree()
