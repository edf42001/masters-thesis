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


class ExperienceHelper:
    def __init__(self):
        # Each dictionary keeps track of experiences using n combinations of literals
        self.experiences = [{}, {}]

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
        # TODO: will this become a problem?
        # Am using string representation. Should I instead use predicate tree, or diectic reference?
        experiences = [", ".join(sorted(combination)) for combination in combinations]

        return experiences

    def update_experience_dict(self, example: Example, n: int):
        # Experience dict is a list of how many times we have tried for every object, every way to interacti with that
        # object, for every action, how many times we've tried each

        action = example.action
        literals = example.state
        experiences = self.experiences[n-1]

        for experience in ExperienceHelper.extract_experiences(literals, n=n):
            if experience not in experiences:
                experiences[experience] = dict()

            if action not in experiences[experience]:
                experiences[experience][action] = 1
            else:
                experiences[experience][action] += 1

    def copy(self):
        helper = ExperienceHelper()
        helper.experiences[0] = self.experiences[0].copy()
        helper.experiences[1] = self.experiences[1].copy()

        return helper
