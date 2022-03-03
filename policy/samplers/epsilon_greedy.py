import random

from policy.samplers.sampler import Sampler


class EpsilonGreedy(Sampler):

    def __init__(self, params: dict):

        self.epsilon = params['epsilon']
        self.decay = params['e_decay']
        self.min_epsilon = params['min_e']

    def sample(self) -> bool:
        return random.random() < self.epsilon

    def update(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
