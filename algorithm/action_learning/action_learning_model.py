from typing import List, Union
import logging
import pickle
import numpy as np

from effects.effect import JointEffect, JointNoEffect
from algorithm.transition_model import TransitionModel
from common.structures import Transition


class ActionLearningModel(TransitionModel):
    """Tracks all conditions and effects for each action/attribute pair"""

    def __init__(self, env, doormax_model: TransitionModel):
        self.env = env

        self.num_inputs = self.env.get_condition_size()
        self.num_actions = self.env.get_num_actions()
        self.num_atts = self.env.NUM_ATT

        self.doormax_model = doormax_model

        logging.info(f"Real action map {self.env.get_action_map()}")

        # Stores current probabilities that each action is each other action. Each row is different action's belief
        self.action_map_belief = np.ones((self.num_actions, self.num_actions)) / self.num_actions
        logging.info("Starting belief action map")
        self.print_model()

    def add_experience(self, action: int, state: int, obs: List[Union[List[int], JointEffect]]):
        """Records experience of state action transition"""

        # Say we execute our action 0. Let A = our action and O = the outcome we observe after taking action A.
        # X represents an action we think it is. P(A=X | O) = P(O | A=X) * P(A=X) / P(O)
        #
        # For example, an outcome could be moving in a direction or not moving. P(A=X) is the prior,
        # P(O) is the sum of the probabilities of the actions that could have caused that effect
        # (can be more than one if surrounded by walls, for example). P(O | A=X) is usually one for movement actions.
        logging.debug(f"Adding experience: {action}, {state}, {obs}")

        # What we currently believe this action represents
        priors = self.action_map_belief[action]

        print(f"Priors for action {action}")
        print(priors)

        # Need to calculate P(observation | action = X). For an action X, this is 1 if executing that action in this
        # state would have created a matching transition. 0 otherwise

        likelihood = np.zeros(self.num_actions)

        for action_i in range(self.num_actions):
            # The transition that would have occurred if we took this action
            transition = self.compute_possible_transitions(state, action_i)
            likelihood[action_i] = 1.0 if self.transitions_match(transition, obs) else 0.0

        print(f"Likelihood for each action: {likelihood}")

        # Update priors
        posterior = priors * likelihood
        posterior = posterior / np.sum(posterior)  # Equivalent to dividing by P(O), normalize
        print(f"Posterior {posterior}")

        self.action_map_belief[action] = posterior

        # In case knowledge of one action ruled out another
        self.cross_action_belief_update(action)

        print("Current model")
        self.print_model()
        print()

    def transitions_match(self, transition, observation) -> bool:
        """Returns true if a potential transition matches an observation"""

        # Due to the mess that is my data types, Transition is Transition object consisting of a list
        # of JointEffects and probabilities, observation is a dict of lists of effects for each att

        effect = transition[0].effect

        # If nothing happened, and we predict JointNoEffect, that is a match
        if not observation:
            return type(effect) == JointNoEffect
        else:
            # If the effect is JointNoEffect, this is clearly not a match because an attribute changed
            if type(effect) == JointNoEffect:
                return False

            # We need to check if the transition matches, which because observation can be all possible effects
            # i.e Increment(1) and SetTo(3), (one of which is fake), if any match that is a match

            # Check that the effect for every attribute matches
            # Make sure there are no contradictions for any attribute
            for att in range(self.num_atts):
                assumed_changed = effect.value[att]
                actual_change = observation[att]
                
                # If the change isn't there, that's a contradiction. Here are some examples
                # NoChange [NoChange] -> good
                # NoChange [Increment(1), SetToNumber(1)] -> bad
                # SetToNumber(4) [NoChange] -> bad
                # NoChange [NoChange] -> good
                if assumed_changed not in actual_change:
                    return False

            # If no contradictions, this matches and is a possible effect
            return True

    def cross_action_belief_update(self, action):
        # Because action maps are one to one, if we know what one action is, that rules out what other actions are

        best_guess = np.argmax(self.action_map_belief[action])

        # If we are certain that is the correct guess, wipe that column except for that one
        # Then update the rows to still sum to one
        if self.action_map_belief[action, best_guess] == 1:
            self.action_map_belief[:, best_guess] = 0
            self.action_map_belief[action, best_guess] = 1
            # Need to have rows sum to 1, so need to transpose to get the alignment
            self.action_map_belief = (self.action_map_belief.T / np.sum(self.action_map_belief, axis=1)).T

    def compute_possible_transitions(self, state: int, action: int) -> List[Transition]:
        """
        Returns the effects (transitions) of taking the action given the condition
        If unknown, return None
        """
        return self.doormax_model.compute_possible_transitions(state, action)

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
        for i in range(self.num_actions):
            print(f"{i}: {self.action_map_belief[i]}")

        best = {i: np.argmax(self.action_map_belief[i]) for i in range(self.num_actions)}
        print(f"Best Guess {best}")
        print(f"Actual map {self.env.get_action_map()}")

    def save(self, filepath):
        logging.info(f"Saving ActionLearningModel to {filepath}")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def has_correct_action_model(self):
        best = {i: np.argmax(self.action_map_belief[i]) for i in range(self.num_actions)}
        return best == self.env.get_action_map()

    def get_action_map_belief(self) -> np.ndarray:
        return self.action_map_belief
