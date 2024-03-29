from typing import List
import logging
import pickle
import time

from common.structures import Transition

from symbolic_stochastic_domains.symbolic_classes import Example, Outcome, ExampleSet, RuleSet, Rule, OutcomeSet
from symbolic_stochastic_domains.learn_ruleset_outcomes import RulesetLearner
from symbolic_stochastic_domains.symbolic_utils import context_matches
from symbolic_stochastic_domains.experience_helper import ExperienceHelper


class SymbolicModel:
    """Tracks interactions with the world with Examples and Experience"""

    def __init__(self, env):
        self.env = env

        self.num_actions = self.env.get_num_actions()
        self.num_atts = self.env.NUM_ATT

        # Store memory of interactions with environment
        self.examples = ExampleSet()

        # An experience helper keeps track of object, predicate, action, counts
        self.experience_helper = ExperienceHelper()

        # Current beleived set of rules that describe environment
        # Need to init with a default rule or we get out of bounds errors with the list
        self.ruleset = RuleSet([Rule(action=-1, context=[], outcomes=OutcomeSet())])

    def add_experience(self, action: int, state: int, outcome: Outcome):
        """Records experience of state action transition"""

        # Convert the observation to an outcome, combine with the set of literals to get an example to add to memory
        literals, instance_name_map = self.env.get_literals(state)
        example = Example(action, literals, outcome)
        self.examples.add_example(example)

        # Keep track of combinations of 1 and 2 literals. This is a arbitrary input to the program.
        self.experience_helper.update_experience_dict(example, 1)
        self.experience_helper.update_experience_dict(example, 2)

        # Currently, update the model on every step. I wonder how it would work to update it based
        # on the existing ruleset
        start_time = time.perf_counter()
        learner = RulesetLearner()
        self.ruleset = learner.learn_ruleset(self.examples)
        end_time = time.perf_counter()
        # print(f"Ruleset learning took {end_time - start_time:.3f} (# of examples {len(self.examples.examples)})")

        # print("New model:")
        # self.print_model()
        # print()

    def compute_possible_transitions(self, state: int, action: int, literals=None, instance_name_map=None) -> List[Transition]:
        """
        Returns the effects (transitions) of taking the action given the condition
        If unknown, return None
        """

        # This is computational expensive, so if we can pass them as params to the function, let's do so
        if literals is None or instance_name_map is None:
            literals, instance_name_map = self.env.get_literals(state)

        transitions = []

        # Find a rule that is applicable to the current state and action. There should only be one

        # Check for rules that are applicable to the current state and action
        rule = None
        for test_rule in self.ruleset.rules:
            # TODO I wonder if we could have context matches return the found matches? for diectic references for the rule?
            if test_rule.action == action and context_matches(test_rule.context, literals):
                rule = test_rule
                if len(test_rule.outcomes.outcomes) > 1:
                    print("Rule had too many outcomes")
                    import sys
                    sys.exit(1)

        if rule is None:
            return transitions

        effect = rule.outcomes.outcomes[0]

        atts = []
        outcomes = []

        # Maps unique object name in the tree, to the real instance index in the environment
        name_instance_map = {v: k for k, v in instance_name_map.items()}

        for reference, outcome in effect.value.items():
            # obb_att_str is formatted either `taxi.y` or `taxi-IN-key.state`. In general, `ob1-pred1-ob2-pred2...-obn`
            from_ob = reference.from_ob
            edge_type = reference.edge_type
            class_name = reference.to_ob
            att_name = reference.att_name
            # We handle the taxi case separately, as it isn't attached to anything
            if edge_type is None:
                unique_name = "taxi0"
            else:
                # We have to find the correct match. A dictionary would be good for this
                # Check for the match by looking for all of that object, and finding the one with the matching connection
                test_id = 0
                while True:
                    test_name = class_name + str(test_id)
                    node = literals.node_lookup[test_name]
                    edge = node.to_edges[0]
                    if from_ob == edge.from_node.object_name and edge_type == edge.type and class_name == edge.to_node.object_name:  # Check for a match
                        break
                    test_id += 1  # Try the next object of this type
                # This the object that was being referenced
                unique_name = class_name + str(test_id)

            # Extract the name of the object class, and the identifier number
            class_name, id_str = unique_name[:-1], unique_name[-1]
            class_idx = self.env.OB_NAMES.index(class_name)
            att_idx = self.env.ATT_NAMES[class_idx].index(att_name)
            instance_id = name_instance_map[unique_name]

            att_range = self.env.instance_index_map[instance_id]
            att = att_range[0] + att_idx

            atts.append(att)
            outcomes.append(outcome)

        # Only create new effect if it isn't a JointNoEffect
        if len(atts) > 0:
            effect = Outcome(atts, outcomes)

        transitions.append(Transition(effect, 1.0))

        return transitions

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
        logging.info(f"Saving SymbolicModel to {filepath}")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
