from typing import List, Union

from effects.effect import JointEffect
from algorithm.transition_model import TransitionModel


class DoormaxRuleset(TransitionModel):
    """Tracks all conditions and effects for each action/attribute pair"""

    def __init__(self):
        pass

    # TODO
    def add_experience(self, action, condition: List[bool], obs_list: List[Union[List[int], JointEffect]]):
        """Distribute the current observations to all relevant Meteorologists"""
        for p in self.parents:
            # Slice condition to match subset of terms expected by parent
            # Change type from list of boolean to a single string
            aux_input = tuple([condition[i] for i in p])
            self.meteorologists[action][p].add_experience(aux_input, obs_list)

    def get_prediction(self, condition: List[bool], action: int):
        """Ask the meteorologists for their best predictions. If any is not ready, return nothing"""
        pass

    def print_action_predictions(self, condition: List[bool]):
        pass

    def print_parent_predictions(self, condition: List[bool], action: int):
        pass

    def __str__(self):
        return '\n'.join([f'Action {action}:\n' + '\n'.join([str(m) for m in met.values()]) for action, met in self.best_meteorologists.items()])
