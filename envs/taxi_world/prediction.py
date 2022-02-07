from envs.taxi_world.effects import Effect


class Prediction:
    """
    A prediction is a pair where the model is a condition
    that needs to be true for the effect to occur
    """
    def __init__(self, model: str, effect: Effect):
        self.model = model
        self.effect = effect

    def __repr__(self):
        return "({} -> {})".format(self.model, self.effect)


# class PredictionSet:
#     def __init__(self, model: str, effect: Effect):
#         self.model = model
#         self.effect = effect
#
#     def __repr__(self):
#         return "({} -> {})".format(self.model, self.effect)
