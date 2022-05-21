import math
import numpy as np
import random

from algorithm.transition_model import TransitionModel
from policy.policy import Policy


class QValueIteration(Policy):
    max_steps = 100
    epsilon = 0.001

    def __init__(self, states: int, actions: int, discount_factor: float, rmax: float, model: TransitionModel):
        self.rmax = rmax
        self.discount_factor = discount_factor
        self.discounted_rmax = rmax / (1 - discount_factor)
        self.num_states = states
        self.num_actions = actions
        self.model = model

        init_value = self.discounted_rmax
        self.q_values = np.full((states, actions), init_value, dtype='float32')

    def choose_action(self, curr_state: int, is_learning: bool = True) -> int:
        # TESTING HACK
        # return random.randint(0, self.num_actions-1)

        # Do q iteration to find the best action
        if is_learning:
            self.q_value_iteration()

        # Choose the best action for the current state
        return self.max_q_action(curr_state)

    def q_value_iteration(self):
        precomputed_transitions = {}
        steps = 0
        while True:
            delta = 0
            for state in range(self.num_states):
                for action in range(self.num_actions):
                    old_q = self.q_values[state][action]

                    # Get list of possible effects from model
                    state_action = (state, action)
                    if state_action not in precomputed_transitions:
                        possible_transitions = self.model.compute_possible_transitions(state, action)
                        precomputed_transitions[state_action] = possible_transitions
                    else:
                        possible_transitions = precomputed_transitions[state_action]

                    # If there is not enough experience, the model will return no transitions
                    # Assume optimistic transition to itself for maximum value
                    if not possible_transitions:
                        new_q = self.discounted_rmax
                    else:
                        # Compute the new q value using next states
                        q_reward, new_q = 0, 0
                        for transition in possible_transitions:
                            effect, prob = transition.effect, transition.prob
                            next_state = self.model.next_state(state, effect)

                            q_reward += prob * self.model.get_reward(state, next_state, action)
                            new_q += prob * self.max_q_value(next_state)

                        new_q *= self.discount_factor
                        new_q += q_reward

                    self.q_values[state][action] = new_q
                    delta = max(delta, abs(new_q - old_q))

            steps += 1

            # Stop value iteration once there are no significant updates to Q values
            if delta < self.epsilon or steps >= self.max_steps:
                break

    def max_q_value(self, state: int) -> float:
        return np.max(self.q_values[state])

    def max_q_action(self, state: int) -> int:
        # Return the best action to take in current state
        # In case of tie, randomly select from best
        best_actions = []
        best_q = -float('inf')
        for action, q_value in enumerate(self.q_values[state]):
            if math.isclose(q_value, best_q):
                best_actions.append(action)
            elif q_value > best_q:
                best_actions = [action]
                best_q = q_value

        return random.choice(best_actions)
