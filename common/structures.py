from typing import Union, List

from effects.effect import JointEffect


class Transition:
    """A transition is a list of effects to be applied to a state, along with a probability of that occurring"""
    def __init__(self, e: Union[List[int], JointEffect], p: float):
        self.effect = e
        self.prob = p

    def __repr__(self):
        return f'Transition <Effect {self.effect}, p {self.prob}>'
