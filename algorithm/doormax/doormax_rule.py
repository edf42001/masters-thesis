from typing import List, Union, Tuple

from effects.effect import Effect, JointEffect


class DoormaxRule:
    """
    A single doormax rule tracks an action / state variable pair, and determines what conditions have what
    effects on that state variable
    """

    def __init__(self, model: str, effect: Effect):
        self.model = model
        self.effect = effect

        self.action = 0
        self.variable = 0

    def add_experience(self, setting: Tuple[bool], obs_list: List[Union[List[int], JointEffect]]):
        """Add experience to the Predictor corresponding to current input"""
        # Convert binary input to integer index
        for obs in obs_list:
            self.predictors[setting].add_observation(obs)

    def has_prediction(self, setting: Tuple[bool] = None):
        """Check if any or all Predictors have enough experience"""
        if setting:
            return self.predictors[setting].has_prediction()
        else:
            return all(p.has_prediction() for p in self.predictors.values())

    def get_prediction(self, setting: Tuple[bool]):
        """Get list of possible effects and probabilities for current input"""
        return self.predictors[setting].get_prediction()

    def __str__(self):
        return "({} -> {})".format(self.model, self.effect)

    def __repr__(self):
        return self.__str__()
