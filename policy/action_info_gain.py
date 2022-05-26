import random
import numpy as np

from effects.effect import JointNoEffect
from policy.policy import Policy
from algorithm.action_learning.action_learning_model import ActionLearningModel


class ActionInfoGain(Policy):
    max_steps = 100
    epsilon = 0.01

    def __init__(self, actions: int, action_model: ActionLearningModel):
        self.num_actions = actions
        self.action_model = action_model

    def choose_action(self, curr_state: int, is_learning: bool = True) -> int:

        scores = [self.information_gain_of_action(curr_state, a) for a in range(self.num_actions)]

        print(f"Action scores: {scores}")

        # If no actions get us any information gain, take a random action
        # TODO: take a random action that we know changes the state. todo: go towards state with higher info gain
        if np.count_nonzero(scores) == 0:
            return random.randint(0, self.num_actions-1)

        # We want the action with the highest information gain
        return np.argmax(scores)

    def info_gain_of_state(self, state: int) -> float:
        """Returns the total info gain over all actions for a state"""
        return sum([self.information_gain_of_action(state, a) for a in range(self.num_actions)])

    def information_gain_of_action(self, state, action):
        """Returns the expected entropy of taking an action"""

        # The entropy of action is the sum over the possible outcomes, p(outcome) * -log(p(outcome))
        # Assume each effect has an equally distributed chance, but some outcomes can have higher probabilities
        # i.e. if we are surrounded by walls, then there could be a 3/4 chance of not moving.

        prior = self.action_model.get_action_map_belief()[action]

        outcome_counts = {}

        num_outcomes = 0
        for a in range(self.num_actions):
            # This action can't occur, remove it's effects from the pool of possible effects

            if prior[a] == 0:
                continue

            possible_result = self.action_model.compute_possible_transitions(state, a)

            effect = possible_result[0].effect  # Transition to effect with 100% probability (deterministic world)

            # Create a dictionary of all possible outcomes and their probability (count) of occurring
            if effect not in outcome_counts:
                outcome_counts[effect] = 1
            else:
                outcome_counts[effect] += 1

            num_outcomes += 1

        print(f"Action {action} prior {prior}")
        print(outcome_counts)

        # For each outcome, how much information will we have if that is the real outcome?
        # Calculate the entropy of the ending probability distribution
        expected_information = 0
        for outcome in outcome_counts.keys():

            # TODO: Use either only in joint effects or only in obs
            if type(outcome) == JointNoEffect:
                obs = {}
            else:
                obs = {att: [outcome.value[att]] for att in range(4)}

            likelihood = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                # The transition that would have occurred if we took this action
                transition = self.action_model.compute_possible_transitions(state, a)
                likelihood[a] = 1.0 if self.action_model.transitions_match(transition, obs) else 0.0

            posterior = prior * likelihood
            posterior /= np.sum(posterior)
            print(f"Result posterior for {outcome}: {posterior}")

            # TODO Instead of particular probabilities of actions, maybe we have the count of remaining actions
            # As a proxy for information? I.E 4/6, 2/6, 1/6
            # TODO: if the posterior does not change, then we are unable to gain any information from doing that action
            # Should we look at information gain or total information?
            # If we can't gain any info, try to navigate to places where we can.
            # Otherwise, pick random actions?
            # Total information gain of a state

            # Remove zeros to avoid errors in log
            posterior = posterior[posterior != 0]

            # Entropy of distribution is sum -p*log(p). Using log2 so we can think in terms of bits
            entropy = np.sum(-posterior * np.log2(posterior))

            print(f"Entropy {entropy}")

            # Keep weighted average of entropy to find expected entropy/information
            expected_information += (outcome_counts[outcome] / num_outcomes) * entropy

            print(f"Expected info: {expected_information}")

        # Now that we are done, lets remove 0's from prior and calculate how much entropy there was to start
        prior = prior[prior != 0]
        prior_entropy = np.sum(-prior * np.log2(prior))
        print(f"Prior entropy {prior_entropy}")
        print()

        # Information gain (prior has greater entropy, in bits, because we gain info when taking actions)
        expected_information = prior_entropy - expected_information
        return expected_information
