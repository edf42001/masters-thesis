import math
import numpy as np
import random

from algorithm.transition_model import TransitionModel
from policy.policy import Policy


class ValueIteration(Policy):
    max_steps = 100
    epsilon = 0.01

    def __init__(self, states: int, actions: int, discount_factor: float, rmax: float, model: TransitionModel):
        self.rmax = rmax
        self.discount_factor = discount_factor
        # self.discounted_rmax = rmax / (1 - discount_factor)
        self.num_states = states
        self.num_actions = actions
        self.model = model

        # Table to store each state's value
        init_value = 0.0
        self.values = np.full(self.num_states, init_value, dtype='float32')

    def choose_action(self, curr_state: int, is_learning: bool = True) -> int:
        # TESTING HACK
        # return random.randint(0, self.num_actions-1)

        # Do value iteration to find the best action (only when learning) (otherwise use precomputed)
        if is_learning:
            self.value_iteration(curr_state)

        # Choose the best action for the current state
        return self.best_value_action(curr_state)

    def value_iteration(self, curr_state: int):
        precomputed_transitions = {}
        iterations = 0

        # Loop until no values change by more than epsilon
        delta = self.epsilon + 1
        while delta > self.epsilon:  # TODO: this needs to be flipped
            delta = 0

            # Iterate over all states
            for state in range(self.num_states):
                current_state_value = self.values[state]

                # In taxi world certain states are unreachable. For example, the destination and pickup location
                # are static and never change, so we don't need to do value iteration on them
                # THis is a hack, why are we passing the action as 0?
                if self.model.unreachable_state(curr_state, state) or self.model.end_of_episode(state):
                    continue

                # For each action we can take in this state, get the next state, or None if the transition is unknown
                # Use precomputed transitions to save processing. Use these to update the values for the next states
                next_values = np.zeros(self.num_actions)
                for action in range(self.num_actions):
                    state_action = (state, action)
                    if state_action not in precomputed_transitions:
                        transition = self.model.compute_possible_transitions(state, action)
                        precomputed_transitions[state_action] = transition
                    else:
                        transition = precomputed_transitions[state_action]

                    # We can't predict the transition, assume max reward (optimism in face of uncertainty)
                    if transition is None:
                        next_values[action] = self.discount_factor * self.rmax
                        continue

                    # We know what will happen, compute next state.
                    # If probabalistic, this could be a list of transition, would take into account probabilites
                    next_state = self.model.next_state(state, transition[0].effect)

                    # If this new state is a terminal state, we use a hack to update the values
                    if self.model.end_of_episode(next_state):
                        # TODO? Which is correct?
                        # next_values[action] = self.discount_factor * self.model.get_reward(state, next_state, action)
                        next_values[action] = self.model.get_reward(state, next_state, action)
                    else:
                        # Otherwise, what is the reward for this transition + discounted value of next state
                        next_values[action] = self.model.get_reward(state, next_state, action) + self.discount_factor * self.values[next_state]

                # The value of this state becomes the optimal reward + next value
                optimal_value = np.max(next_values)
                self.values[state] = optimal_value
                delta = max(delta, abs(optimal_value - current_state_value))

            iterations += 1

    def best_value_action(self, state: int) -> int:
        """
        From a current state and the value table, which action will have the best total value?
        """
        # Return the best action to take in current state
        # TODO: In case of tie, randomly select from best
        # There is some redundancy here in recomputing the transitions
        # best_actions = []
        # best_q = -float('inf')
        # for action, q_value in enumerate(self.q_values[state]):
        #     if math.isclose(q_value, best_q):
        #         best_actions.append(action)
        #     elif q_value > best_q:
        #         best_actions = [action]
        #         best_q = q_value

        # return random.choice(best_actions)

        best_value = -100
        best_action = None

        best_values = []
        for action in range(self.num_actions):
            transition = self.model.compute_possible_transitions(state, action)
            if transition is None:
                value = self.rmax
            else:
                # See old doormax_taxi.py for descriptions of weird edge cases here
                # Need to figure out if the reward should be taken into account
                next_state = self.model.next_state(state, transition[0].effect)
                value = self.model.get_reward(state, next_state, action) + self.values[next_state]

                if self.model.end_of_episode(state):
                    value = self.model.get_reward(state, next_state, action)

            if value > best_value:
                best_value = value
                best_action = action

            # This is just for debugging and printing
            best_values.append(value)

        # print("Values of actions: " + str(best_values))
        return best_action
