from typing import List
import logging
import pickle
import sys
import time

from effects.effect import JointEffect
from algorithm.transition_model import TransitionModel
from common.structures import Transition

from symbolic_stochastic_domains.symbolic_classes import Example, Outcome, ExampleSet, RuleSet, Rule, OutcomeSet
from symbolic_stochastic_domains.learn_ruleset_outcomes import learn_ruleset_outcomes
from symbolic_stochastic_domains.symbolic_utils import context_matches


class SymbolicModel(TransitionModel):
    """Tracks all conditions and effects for each action/attribute pair"""

    def __init__(self, env):
        self.env = env

        self.num_actions = self.env.get_num_actions()
        self.num_atts = self.env.NUM_ATT

        # Store memory of interactions with environment
        self.examples = ExampleSet()

        # Current beleived set of rules that describe environment
        # Need to init with a default rule or we get out of bounds errors with the list
        self.ruleset = RuleSet([Rule(action=-1, context=[], outcomes=OutcomeSet())])

    def add_experience(self, action: int, state: int, obs: JointEffect):
        """Records experience of state action transition"""

        # Convert the observation to an outcome, combine with the set of literals to get an example to add to memory
        outcome = Outcome(obs)
        literals, instance_name_map = self.env.get_literals(state)
        example = Example(action, literals, outcome)
        self.examples.add_example(example)

        # Currently, update the model on every step. I wonder how it would work to update it based
        # on the existing ruleset
        start_time = time.perf_counter()
        self.ruleset = learn_ruleset_outcomes(self.examples)
        end_time = time.perf_counter()
        print(f"Ruleset learning took {end_time - start_time:.3f} (# of examples {len(self.examples.examples)})")

        print("New model:")
        self.print_model()
        print()

    def compute_possible_transitions(self, state: int, action: int) -> List[Transition]:
        """
        Returns the effects (transitions) of taking the action given the condition
        If unknown, return None
        """

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

        effect = rule.outcomes.outcomes[0].outcome

        atts = []
        outcomes = []

        # Maps unique object name in the tree, to the real instance index in the environment
        name_instance_map = {v: k for k, v in instance_name_map.items()}

        for ob_att_str, outcome in effect.value.items():
            # obb_att_str is formatted either `taxi.y` or `taxi-IN-key.state`. In general, `ob1-pred1-ob2-pred2...-obn`
            identifier_str, att_name_str = ob_att_str.split(".")

            # We handle the taxi case seperatley, as it isn't attached to anything
            if identifier_str == "taxi":
                unique_name = "taxi0"
            else:
                # We have to find the correct match. A dictionary would be good for this
                # Check for the match by lookng for all of that object, and finding the one with the matching connection
                class_name = identifier_str.split("-")[-1]  # `taxi-IN-key`, will extract `key`
                test_id = 0
                while True:
                    test_name = class_name + str(test_id)
                    node = literals.node_lookup[test_name]
                    edge = node.to_edges[0]
                    test_identifier_str = f"{edge.from_node.object_name[:-1]}-{str(edge)[:-1]}"  # Will be`taxi-IN-key`, maybe `taxi-ON-key`
                    if test_identifier_str == identifier_str:  # Check for a match
                        break
                    test_id += 1  # Try the next object of this type
                # This the object that was being referenced
                unique_name = class_name + str(test_id)

            # Extract the name of the object class, and the identifier number
            class_name, id_str = unique_name[:-1], unique_name[-1]
            class_idx = self.env.OB_NAMES.index(class_name)
            att_idx = self.env.ATT_NAMES[class_idx].index(att_name_str)
            instance_id = name_instance_map[unique_name]

            att_range = self.env.instance_index_map[instance_id]
            att = att_range[0] + att_idx

            atts.append(att)
            outcomes.append(outcome)

        # Only create new effect if it isn't a JointNoEffect
        if len(atts) > 0:
            effect = JointEffect(atts, outcomes)

        transitions.append(Transition(effect, 1.0))

        return transitions

    def get_reward(self, state: int, next_state: int, action: int):
        """Assumes all rewards are known in advance"""
        return self.env.get_reward(state, next_state, action)

    def next_state(self, state: int, observation) -> int:
        return self.env.apply_effect(state, observation)

    def print_action_predictions(self, state: int):
        condition = self.env.get_condition(state)
        pass

    def print_parent_predictions(self, state: int, action: int):
        condition = self.env.get_condition(state)
        pass

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
