from typing import List, Union
import logging

from effects.effect import JointEffect, EffectType, JointNoEffect
from algorithm.transition_model import TransitionModel
from algorithm.doormax.doormax_rule import DoormaxRule
from environment.environment import Environment
from common.structures import Transition

from algorithm.doormax.utils import find_matching_prediction, boolean_arr_to_string,\
    condition_matches, commute_condition_strings, check_conditions_overlap, incompatible_effects


class DoormaxRuleset(TransitionModel):
    """Tracks all conditions and effects for each action/attribute pair"""

    def __init__(self, env: Environment):
        self.env = env

        self.num_inputs = self.env.get_condition_size()
        self.num_actions = self.env.get_num_actions()
        self.num_atts = self.env.NUM_ATT

        # Maximum number of different predictions allowed for an effect type
        self.k = 1

        # Collection of failure conditions for actions (what conditions cause no change for given action)
        self.failure_conditions = dict()

        # List of effect predictions for each attribute and action
        self.predictions = dict()

        self.init_data_structures()

    def init_data_structures(self):
        for action in range(self.num_actions):
            # Initialize all failure conditions for that action to empty
            self.failure_conditions[action] = []

            # Setup empty prediction list
            self.predictions[action] = dict()

            # For all attributes and effects, set predictions for the combination to null
            for attribute in range(self.num_atts):

                # Setup empty prediction list
                self.predictions[action][attribute] = dict()

                for e_type in EffectType:
                    # List of predictions starts empty
                    self.predictions[action][attribute][e_type] = []

    # TODO
    def add_experience(self, action: int, state: int, obs: List[Union[List[int], JointEffect]]):
        """Records experience of state action transition"""
        condition = self.env.get_condition(state)
        condition_str = boolean_arr_to_string(condition)

        logging.info("Adding experience")
        logging.info(f"action: {action}")
        logging.info(f"condition: {condition_str}")
        logging.info(f"observation: {obs}")

        # obs_list is a dictionary of lists of effects, or empty for all vars have no change
        if not obs:
            logging.info("Is a failure condition")
            # If the states are the same, this is a failure condition for the action (nothing changed)

            # TODO: remove all matches conditions (to prevent duplicates? why?)
            # Print thrice, to check for change
            # print(self.failure_conditions[action])
            self.failure_conditions[action] = [c for c in self.failure_conditions[action] if not condition_matches(condition_str, c)]
            # print(self.failure_conditions[action])
            self.failure_conditions[action].append(condition_str)
            logging.info(f"Failure conditions {self.failure_conditions[action]}")
        else:
            logging.info("Not a failure condition")
            # Look through all effects to all attributes
            for attribute, effect_list in obs.items():
                for effect in effect_list:
                    logging.info(f"Attribute/effect: {attribute}, {effect}")

                    # Look through predictions for current action, attribute, and e type
                    # to find a matching effect
                    current_predictions = self.predictions[action][attribute][effect.type]

                    # If ever set to None, this means this is the wrong effect type for this pair
                    if current_predictions is None:
                        logging.info("This condition is known to be wrong")
                        continue

                    logging.info("Current predictions: {}".format(current_predictions))

                    matched_prediction = find_matching_prediction(current_predictions, effect)

                    # Check if we already have an effect that matches for this action and attribute
                    if matched_prediction is not None:
                        # We already have a prediction for what will happen to this attribute when
                        # this action is taken. Lets update it to make the condition more accurate
                        # print("Found matching prediction")

                        matched_prediction.model = commute_condition_strings(matched_prediction.model, condition_str)
                        # print("Updated prediction: {}".format(matched_prediction))

                        # Check for overlapping conditions, this would be a contradiction and we remove this Type
                        if check_conditions_overlap(current_predictions, matched_prediction):
                            self.predictions[action][attribute][effect.type()] = None
                        else:
                            pass
                            # print("No overlap")

                    else:
                        # A new effect has been observed.
                        # If the condition does not overlap and existing condition, add the new prediction (TODO: why?)

                        logging.info("Did not find matching prediction")
                        models = [p.model for p in current_predictions]
                        logging.info("Models: {}".format(models))

                        # Search for overlapping conditions
                        # TODO: why does this one compare cond/c, and c/cond, while the other only does one
                        overlap = False
                        for c in models:
                            if condition_matches(condition_str, c) or condition_matches(c, condition_str):
                                # print("Found overlap, removing: {}, {}".format(condition_s, c))
                                overlap = True
                                break

                        if overlap:
                            self.predictions[action][attribute][effect.type] = None
                        else:
                            # Now we can add the new prediction to the list
                            logging.info("No overlap")

                            current_predictions.append(DoormaxRule(condition_str, effect))

                            # Make sure it got added
                            logging.info(f"New predictions: {self.predictions[action][attribute][effect.type]}")

                            # Check if there are more than k predictions for this action/attribute/type
                            # If there are, that's not the real effect type, so remove it from the running
                            # TODO: figure this out. Is k = 1? Or is it a bigger fudge factor?
                            if len(self.predictions[action][attribute][effect.type]) > self.k:
                                # print("Too many effects, removing")
                                self.predictions[action][attribute][effect.type] = None

    def compute_possible_transitions(self, state: int, action: int, debug=False) -> List[Transition]:
        """
        Returns the effects (transitions) of taking the action given the condition
        If unknown, return None
        """
        condition = self.env.get_condition(state)
        condition_str = boolean_arr_to_string(condition)
        # print("Action: {}".format(action))
        # print("Condition: {}".format(condition_str))

        for failure_condition in self.failure_conditions[action]:
            if condition_matches(failure_condition, condition_str):
                # The current condition is a failure condition. No change
                return [Transition(JointNoEffect(), 1.0)]

        # Otherwise, check all effects and attributes
        # TODO: Is this wrong? Should applied effects be higher up??

        # Store effects for each attribute to plug into a joint effect later
        effects = []

        for attribute in range(self.num_atts):
            applied_effects = []

            # print(f"Att: {attribute}")
            for e_type in EffectType:
                # print(e_type)
                # If we have predictions that match the state we are currently in,
                # then we know those effects will happen
                current_predictions = self.predictions[action][attribute][e_type]
                # print("Current_predictions: {}".format(current_predictions))

                # If none, not a real effect, continue
                if current_predictions is None:
                    continue

                for pred in self.predictions[action][attribute][e_type]:
                    if condition_matches(pred.model, condition_str):
                        # print("Matching condition: {}, {}".format(pred.model, condition_s))
                        applied_effects.append(pred.effect)

            # print(applied_effects)
            # If e is empty or there are incompatible effects, we don't know what will happen, return max_reward
            if len(applied_effects) == 0 or incompatible_effects(applied_effects):
                return None  # None represents max reward
            else:
                # Otherwise, return the effects that will occur as transitions
                # Need to combine all effects into one joint effect.

                # print("Effects for this state: {}".format(applied_effects))
                # Convert effects to joint effect
                effects.append(applied_effects[0])  # Only one valid effect will get through

        # Create teh joint effect and say it has a probability of 1.0
        joint_effect = JointEffect(att_list=list(range(self.num_atts)), eff_list=effects)
        return [Transition(joint_effect, 1.0)]

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
        ret = ""

        for action in range(self.num_actions):
            ret += self.env.get_action_name(action) + "\n"
            for attribute in range(self.num_atts):
                non_empty_effects = []
                for e_type in EffectType:
                    effects = self.predictions[action][attribute][e_type]
                    if effects is not None and len(effects) != 0:
                        non_empty_effects.append(effects)

                # Only print attribute and effects if there is something to see
                if len(non_empty_effects) != 0:
                    ret += "{}: {}\n".format(self.env.get_att_name(attribute), non_empty_effects)

        print(ret)
