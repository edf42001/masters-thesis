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
        literals, groundings, properties = self.env.get_literals(state)
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

        literals, groundings, properties = self.env.get_literals(state)

        transitions = []

        # Check for rules that are applicable to the current state and action
        for rule in self.ruleset.rules:
            if rule.action == action and context_matches(rule.context, literals):
                if len(rule.outcomes.outcomes) > 1:
                    print("Rule had too many outcomes")
                    sys.exit(1)

                # Assume discrete
                effect = rule.outcomes.outcomes[0].outcome
                # convert back to ints # TODO: This doesn't really just work, you need the grounding to know what
                # attribute goes to which object. Also, what if there's a key to the top and bottom?
                # Then you need deictic references. Or, we could assume that doesn't happen for now and see what happens
                # AS a hack, yeah lets try that first
                atts = []
                outcomes = []
                # Need mapping from object id and variable name to att value
                # Lets do this manually for now, because I am confused
                for att_str, outcome in effect.value.items():
                    # Convert string to class id and which attribute of that class it is
                    class_str, att_idx_str = att_str.split(".")
                    class_idx = self.env.OB_NAMES.index(class_str)
                    att_idx_str = self.env.ATT_NAMES[class_idx].index(att_idx_str)

                    if class_str == "taxi":
                        att = 0 if att_idx_str == "x" else 1
                    else:
                        # print("I do not know what to do with this class ")
                        # print(rule)
                        # print(class_str, att_idx_str)
                        # print([literal for literal in literals if literal.value])
                        # print(groundings)

                        # First, find how the key is referred to.
                        references = [lit for lit in rule.context if lit.object2 == class_str]
                        if len(references) == 0 or len(references) == 2:
                            print(f"Whoops, how can we refer to the object if the references are {references}")
                            sys.exit(1)

                        # Get the referring literal, and
                        # Use the dict to convert that reference to the object it refers to
                        reference = references[0]
                        grounding = groundings[reference]

                        # Convert the grounding to the indecies of it's state variables
                        att_range = self.env.instance_index_map[grounding]
                        # The attribute to change is the start of this objects attribute in the global list,
                        # plus which attribute it is in that object specifically
                        # Note this only works if the objects in the grounding from get_literals are in the same
                        # order as the objects attributes order is considered
                        att = att_range[0] + int(att_idx_str)

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
