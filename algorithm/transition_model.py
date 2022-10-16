from typing import List

from common.structures import Transition
from symbolic_stochastic_domains.symbolic_classes import Outcome
from environment.environment import Environment


class TransitionModel:
    """A transition model computes possible transitions from a given state"""
    env: Environment = None

    def compute_possible_transitions(self, state: int, action: int) -> List[Transition]:
        raise NotImplementedError()

    def get_reward(self, state: int, next_state: int, action: int) -> float:
        raise NotImplementedError()

    def next_state(self, state: int, effect: Outcome) -> int:
        raise NotImplementedError()

    def print_model(self):
        raise NotImplementedError()
