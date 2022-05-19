from typing import List, Union
import logging

from effects.effect import JointEffect, EffectType, JointNoEffect
from algorithm.transition_model import TransitionModel
from algorithm.doormax.doormax_rule import DoormaxRule

from algorithm.doormax.utils import find_matching_prediction, boolean_arr_to_string


class DoormaxRuleset(TransitionModel):
    """Tracks all conditions and effects for each action/attribute pair"""

    def __init__(self, num_actions: int, num_state_vars: int, k: int = 1):
        self.num_actions = num_actions
        self.num_state_vars = num_state_vars

        # Maximum number of different predictions allowed for an effect type
        self.k = k

        # List of effect predictions for each attribute and action
        self.predictions = dict()

        self.init_data_structures()

    def init_data_structures(self):
        for action in range(self.num_actions):
            self.predictions[action] = dict()

            # For all attributes and effects, set predictions for the combination to null
            for attribute in range(self.num_state_vars):

                # Setup empty prediction list
                self.predictions[action][attribute] = dict()

                for e_type in EffectType:
                    # List of predictions starts empty
                    self.predictions[action][attribute][e_type] = []

    # TODO
    def add_experience(self, action, condition: List[bool], obs_list: List[Union[List[int], JointEffect]]):
        """Records experience of state action transition"""

        condition_str = boolean_arr_to_string(condition)

        logging.info("Adding experience")
        logging.info(f"action: {action}")
        logging.info(f"condition: {condition_str}")
        logging.info(f"observation: {obs_list}")

        if type(obs_list[0]) is JointNoEffect:
            logging.info("Is a failure condition")
            # If the states are the same, this is a failure condition for the action (nothing changed)

            # TODO: remove all matches conditions (to prevent duplicates? why?)
            # Print thrice, to check for change
            # print(self.failure_conditions[action])
            self.failure_conditions[action] = [c for c in self.failure_conditions[action] if not condition_matches(condition_str, c)]
            # print(self.failure_conditions[action])
            self.failure_conditions[action].append(condition_str)
            logging.info(self.failure_conditions[action])
        else:
            logging.info("Not a failure condition")
            # Look through all effects to all attributes
            for joint_effect in obs_list:
                # print()
                attribute = list(joint_effect.value.keys())[0]
                effect = joint_effect.value[attribute]
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
                    if self.check_conditions_overlap(current_predictions, matched_prediction):
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

    def get_prediction(self, condition: List[bool], action: int):
        """Ask the meteorologists for their best predictions. If any is not ready, return nothing"""
        pass

    def print_action_predictions(self, condition: List[bool]):
        pass

    def print_parent_predictions(self, condition: List[bool], action: int):
        pass

    def __str__(self):
        return '\n'.join([f'Action {action}:\n' + '\n'.join([str(m) for m in met.values()]) for action, met in self.best_meteorologists.items()])
