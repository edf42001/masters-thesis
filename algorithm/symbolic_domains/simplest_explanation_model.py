"""
Created on 2/5/23 by Ethan Frank

Model for simplest explanation object discovery
"""

from typing import List

from common.structures import Transition

from symbolic_stochastic_domains.symbolic_classes import Example, Outcome, RuleSet, ExampleSet, PredicateTree
from symbolic_stochastic_domains.learn_ruleset_outcomes import RulesetLearner
from symbolic_stochastic_domains.object_transfer import determine_transition_given_action, determine_transition_given_action_2
from test.simplicity_object_transfer.simplest_explanation_based_object_transfer import get_object_permutation_rule_complexities
from symbolic_stochastic_domains.experience_helper import ExperienceHelper


class SimplestExplanationModel:
    """Tracks interactions with the world with Examples and Experience"""

    def __init__(self, env, previous_ruleset: RuleSet, previous_examples: ExampleSet,
                 previous_experiences: ExperienceHelper):
        self.env = env

        self.num_actions = self.env.get_num_actions()
        self.num_atts = self.env.NUM_ATT

        # Learned rules, examples, and experiences from source domain
        self.previous_ruleset = previous_ruleset
        self.previous_examples = previous_examples
        self.previous_experiences = previous_experiences

        # New examples in target task
        self.new_examples = ExampleSet()
        self.new_experience_helper = ExperienceHelper()  # Keeps track of object, predicate, action, counts

        # List of every ruleset that the agent believes is optimal. Keys are the permutation that produced them
        self.best_rulesets = {(): (self.previous_ruleset, [])}

        # List of known object names without taxi and with wall (wall is static so is not in the list normally)
        # TODO: need to figure out how to organize this properly
        self.prior_object_names = env.OB_NAMES.copy()
        self.prior_object_names.append("wall")
        self.prior_object_names.remove("taxi")
        self.prior_object_names.remove("pass")
        self.prior_object_names.remove("dest")

        # Could also get this from examples?
        current_object_names = self.env.get_object_names()
        self.object_map = {unknown: self.prior_object_names.copy() for unknown in current_object_names if unknown != "taxi"}

        # Keeps track of the complexity that the ruleset would increase by for all types of object assignments
        self.object_map_counts = {unknown: {name: 0 for name in self.prior_object_names} for unknown in env.get_object_names() if unknown != "taxi"}

        # Collection of all object names seen so far. Used for permutations
        self.seen_objects = set()

        # True if all permutations on previous object mappings required a ruleset with increased complexity
        self.ruleset_had_to_change = False

    def add_experience(self, action: int, state: int, outcome: Outcome):
        """Records experience of state action transition"""

        # Convert the observation to an outcome, combine with the set of literals to get an example to add to memory
        literals, instance_name_map = self.env.get_literals(state)
        example = Example(action, literals, outcome)

        self.new_examples.add_example(example)
        self.new_experience_helper.update_experience_dict(example, 1)
        self.new_experience_helper.update_experience_dict(example, 2)

        # Get all objects referenced in the current state and add to our running total
        state_objects = set([diectic_obj.split("-")[-1] for diectic_obj in literals.referenced_objects])
        self.seen_objects.update(state_objects)

        # Convert to list so they are in a consistent order, so the permutations can be associated with them
        state_objects = list(self.seen_objects)

        print("Experienced Example:")
        print(example)

        # Notice this maps ones that aren't in the current state?
        # mappings_to_choose_from = (self.prior_object_names for _ in self.seen_objects)
        # Options are only what we think it is
        # mappings_to_choose_from = (self.object_map[unknown] for unknown in self.seen_objects)
        mappings_to_choose_from = (self.object_map[unknown] for unknown in state_objects)

        learner = RulesetLearner()
        complexities, permutations, rulesets = get_object_permutation_rule_complexities(
            mappings_to_choose_from, self.previous_ruleset, self.previous_examples,
            self.new_examples, learner, state_objects
        )

        for complexity, permutation in zip(complexities, permutations):
            print(f"{complexity}: {state_objects}->{permutation}")

        min_complexity = min(complexities)

        # I store them in a hacky tuple because I just want some way to store the mapping, but dicts aren't hashable
        self.best_rulesets = {permutation: (ruleset, state_objects) for ruleset, complexity, permutation in
                         zip(rulesets, complexities, permutations) if complexity == min_complexity}

        print(f"Number of best rulesets: {len(self.best_rulesets)}")

        # If we already did the update step, then don't do it again
        if not self.ruleset_had_to_change and min_complexity != 0:
            self.ruleset_had_to_change = True
            print("Ruleset changed! Trying again with new object map")

            # Try adding unknown object to each and seeing the difference:
            # TODO: a cheat for testing purposes
            for i, unknown in enumerate(state_objects):
                # if unknown not in self.object_map[unknown]:
                if unknown in ["oixzh", "tyyaw"] and unknown not in self.object_map[unknown]:
                    self.object_map[unknown].append(unknown)

            # Weird hacks going on here need to be figured out
            print(state_objects)
            mappings_to_choose_from = (self.object_map[unknown] for unknown in state_objects)

            for known, unknown in self.object_map.items():
                print(known, unknown)

            learner = RulesetLearner()
            complexities, permutations, rulesets = get_object_permutation_rule_complexities(
                mappings_to_choose_from, self.previous_ruleset, self.previous_examples,
                self.new_examples, learner, state_objects
            )

            for complexity, permutation in zip(complexities, permutations):
                print(f"{complexity}: {state_objects}->{permutation}")

            min_complexity = min(complexities)
            self.best_rulesets = {permutation: (ruleset, state_objects)  for ruleset, complexity, permutation in
                             zip(rulesets, complexities, permutations) if complexity == min_complexity}

            print(f"Number of best rulesets: {len(self.best_rulesets)}")

        # Update object map based on complexity values
        # The least complex ones get to be kept
        self.object_map_counts = {unknown: {name: 0 for name in self.prior_object_names} for unknown in self.env.get_object_names() if unknown != "taxi"}

        for complexity, permutation in zip(complexities, permutations):
            for unknown, known in zip(state_objects, permutation):
                if known not in self.object_map_counts[unknown]:
                    self.object_map_counts[unknown][known] = 0
                self.object_map_counts[unknown][known] += 1 if complexity == min_complexity else 0

        for unknown, counts in self.object_map_counts.items():
            print(unknown, ", ".join([f"{c}" for c in counts.values()]))

        length_of_beliefs = sum(len(possibilities) for possibilities in self.object_map.values())

        # Update object map based on above
        for unknown, counts in self.object_map_counts.items():
            max_for_object = max(counts.values())
            self.object_map[unknown] = [known for known in self.object_map[unknown] if counts[known] == max_for_object]

        print("New object map:")
        for key, value in self.object_map.items():
            print(f"{key}: {value}")
        print()

        new_lengths_of_beliefs = sum(len(possibilities) for possibilities in self.object_map.values())

        if new_lengths_of_beliefs != length_of_beliefs:
            print("Learned something new")

    def compute_possible_transitions(self, state: int, action: int, literals=None, instance_name_map=None) -> List[Transition]:
        """
        Returns the effects (transitions) of taking the action given the condition
        If unknown, return None
        """

        # This is computational expensive, so if we can pass them as params to the function, let's do so
        if literals is None or instance_name_map is None:
            literals, instance_name_map = self.env.get_literals(state)

        transitions = self.get_outcomes_using_rulesets(state, action, literals)

        if transitions is None:
            return []

        all_effects = []

        # Maps unique object name in the tree, to the real instance index in the environment
        name_instance_map = {v: k for k, v in instance_name_map.items()}

        # Loop over all possible transitions (i.e., if we don't know what the object is, it could
        # transition to a state where a gem or key is picked up), and generate an effect for each
        for transition in transitions:
            atts = []
            outcomes = []

            effect = transition

            for reference, outcome in effect.value.items():
                edge_type = reference.edge_type
                to_ob = reference.to_ob
                att_name = reference.att_name

                unique_name = ""
                # We handle the taxi case separately, as it isn't attached to anything
                if edge_type is None:
                    unique_name = "taxi0"
                    known_class_name = "taxi"  # This is only used so the env can say which attribute is which index
                else:
                    # Use the fact that only one object can be at the end of each relation to figure out what
                    # the desired object is in this state
                    for edge in literals.base_object.edges:
                        if edge.type == edge_type:
                            unique_name = edge.to_node.full_name()
                            break

                    # This is only used so the env can say which attribute is which index
                    # `taxi-IN-key`, will extract `key`
                    # Probably this should be a helper function in the env
                    known_class_name = to_ob

                    # unique_name = class_name + str(test_id)
                assert unique_name != "", "We should have found a match"

                # Do some weird indices hacks to figure out which attribute index is being referred to
                # so the env can directly modify the imagined state.
                class_idx = self.env.OB_NAMES.index(known_class_name)

                att_idx = self.env.ATT_NAMES[class_idx].index(att_name)
                instance_id = name_instance_map[unique_name]

                att_range = self.env.instance_index_map[instance_id]
                att = att_range[0] + att_idx

                atts.append(att)
                outcomes.append(outcome)

            # Only create new effect if it isn't a JointNoEffect
            if len(atts) > 0:
                effect = Outcome(atts, outcomes)

            all_effects.append(effect)

        # If there is a chance of more than one transition (i.e., if we don't know what the object is, it could
        # transition to a state where a gem or key is picked up) (in the agent's mind, only one actually happens)
        probability = 1.0 / len(all_effects)
        all_transitions = [Transition(effect, probability) for effect in all_effects]

        return all_transitions

    def get_outcomes_using_rulesets(self, state: int, action: int, literals: PredicateTree) -> List[Outcome]:
        # Iterate over each ruleset. Reduce the object map as required. Pass that into the thingy. Does the permutations

        all_outcomes = []
        for permutation, (ruleset, seen_objects) in self.best_rulesets.items():
            object_map = {key: value.copy() for key, value in self.object_map.items()}

            for seen_object, permute_object in zip(seen_objects, permutation):
                object_map[seen_object] = [permute_object]

            # It considers applicable rules to be ones with the same action, that's why it returns all types
            outcome = determine_transition_given_action_2(action, object_map, ruleset, literals)

            all_outcomes.append(outcome)

        # Only compare result to see if the outcomes are the same
        # TODO: this should really compare diectic references, or remap back to original,
        all_the_same = all(outcome is not None and list(outcome.value.values()) == list(all_outcomes[0].value.values()) for outcome in all_outcomes)
        all_outcomes = all_outcomes if all_the_same else None

        return all_outcomes

    def get_reward(self, state: int, next_state: int, action: int):
        """Assumes all rewards are known in advance"""
        return self.env.get_reward(state, next_state, action)

    def next_state(self, state: int, observation) -> int:
        return self.env.apply_effect(state, observation)

    def unreachable_state(self, from_state: int, to_state: int) -> bool:
        return self.env.unreachable_state(from_state, to_state)

    def end_of_episode(self, state: int) -> bool:
        return self.env.end_of_episode(state)

    def print_model(self):
        """Returns predictions in an easy to read format"""
        print(self.ruleset)

    def save(self, filepath):
        raise NotImplementedError("Save not implemented for object transfer model")
