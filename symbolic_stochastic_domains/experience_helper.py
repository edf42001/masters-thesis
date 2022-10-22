"""
Created on 10/22/22 by Ethan Frank

The experience helper keeps track of which experiences the agent has seen.
Experiences are subsets of the literals graph combined with an action.
I.E., have I ever been touching a lock while holding a key and used the unlock action?
"""

from typing import List
import itertools

from symbolic_stochastic_domains.predicate_tree import PredicateTree
from symbolic_stochastic_domains.symbolic_classes import Example
from symbolic_stochastic_domains.predicates_and_objects import PredicateType


class ExperienceHelper:
    def __init__(self):
        # Experiences containing only 1/2 literals
        self.experiences_1 = {}
        self.experiences_2 = {}

    @staticmethod
    def extract_experiences(tree: PredicateTree, n: int) -> List[str]:
        """
        Extracts all permutations of sets of literals size n from a predicate tree.
        Returns them as a list of strings, for use in hashing in dictionaries
        """

        # First, convert each interaction to a string
        interactions = []
        for edge in tree.base_object.edges:
            interaction = f"{tree.base_object.object_name}-{edge.type.name}-{edge.to_node.object_name}"

            # Include properties. Would probably need to sort if there was more than one
            if len(edge.to_node.properties) > 0:
                interaction += ", " + ", ".join([f"{edge.to_node.object_name}.{key.name}: {value}" for key, value in edge.to_node.properties.items()])

            interactions.append(interaction)

        # Wait, this will never be called, states don't explicitly contain negative edges
        # I'd have to specifically enumerate them
        for edge in tree.base_object.negative_edges:
            interaction = f"~{tree.base_object.object_name}-{edge.type.name}-{edge.to_node.object_name}"
            interactions.append(interaction)

        combinations = itertools.combinations(interactions, n)

        # Sort so the order is always the same and they will always hash the same
        experiences = [", ".join(sorted(combination)) for combination in combinations]

        return experiences

    def update_experience_dict(self, example: Example):
        # Experience dict is a list of how many times we have tried for every object, every way to interacti with that
        # object, for every action, how many times we've tried each

        action = example.action
        literals = example.state

        for experience in ExperienceHelper.extract_experiences(literals, n=1):
            if experience not in self.experiences_1:
                self.experiences_1[experience] = dict()

            if action not in self.experiences_1[experience]:
                self.experiences_1[experience][action] = 1
            else:
                self.experiences_1[experience][action] += 1
