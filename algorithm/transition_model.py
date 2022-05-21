from typing import List

from common.structures import Transition
from effects.effect import JointEffect


class TransitionModel:
    """A transition model computes possible transitions from a given state"""
    env = None

    def compute_possible_transitions(self, state: int, action: int) -> List[Transition]:
        raise NotImplementedError()

    def get_reward(self, state_index: int, next_state_index: int, action_index: int) -> float:
        raise NotImplementedError()

    def next_state(self, state: int, effect: JointEffect) -> int:
        raise NotImplementedError()

    def print_model(self):
        raise NotImplementedError()
