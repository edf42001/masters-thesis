from typing import Dict
import numpy as np
import random

from policy.policy import Policy


class DistanceMetricOptionDiscoverPolicy(Policy):
    """For now, just does basic Q learning"""

    def __init__(self, num_states: int, num_actions: int, params):
        self.discount_factor = params['discount_factor']
        self.learning_rate = params['learning_rate']
        self.decay = params['e_decay']
        self.min_rate = params['min_e']

        self.num_states = num_states
        self.num_actions = num_actions

        init_value = 0
        self.q_values = np.full((num_states, num_actions), init_value, dtype='float32')

    def add_experience(self, state: int, action: int, reward: float, next_state: int, terminal: bool):
        """Update Q values using Q-Learning backup"""
        old_q = self.q_values[state][action]
        new_q = reward if terminal else reward + self.discount_factor * np.max(self.q_values[next_state])

        lr = self.learning_rate
        self.q_values[state][action] = (1 - lr) * old_q + lr * new_q

    def update(self):
        """Does learning rate decay, by multiplying by a constant"""
        self.learning_rate = max(self.min_rate, self.decay * self.learning_rate)

    def choose_action(self, curr_state: int, explore: bool = True) -> int:
        if explore:
            # Chose a random action from our options
            return np.random.randint(low=0, high=self.num_actions)
        else:
            # Return max q value
            return int(np.argmax(self.q_values[curr_state]))

    def save(self, filename):
        np.save(filename, self.q_values)

    def output(self):
        print(self.q_values)
