from typing import List, Union

from algorithm.doormax.doormax_ruleset import DoormaxRuleset
from algorithm.transition_model import TransitionModel
from effects.effect import JointEffect
from environment.environment import Environment


class Doormax(TransitionModel):
    # This is dumb why don't we just factor this into the doormax_ruleset

    def __init__(self, env: Environment, use_outcomes: bool = False):
        self.env = env
        self.use_outcomes = use_outcomes

        self.max_parents = self.env.get_max_parents()
        self.num_inputs = self.env.get_condition_size()
        self.num_actions = self.env.get_num_actions()
        self.num_state_vars = self.env.NUM_ATT

        self.model = DoormaxRuleset(self.num_actions, self.num_state_vars, self.env.ACTION_NAMES, self.env.ATT_NAMES)

    def add_experience(self, action: int, state: int, obs: List[Union[List[int], JointEffect]]):
        condition = self.env.get_condition(state)
        self.model.add_experience(action, condition, obs)

    def compute_possible_transitions(self, state: int, action: int):
        condition = self.env.get_condition(state)
        return self.model.get_prediction(condition, action)

    def get_reward(self, state: int, next_state: int, action: int):
        """Assumes all rewards are known in advance"""
        return self.env.get_reward(state, next_state, action)

    def next_state(self, state: int, observation) -> int:
        if self.use_outcomes:
            return self.env.apply_outcome(state, observation)
        else:
            return self.env.apply_effect(state, observation)

    def print_action_predictions(self, state: int):
        condition = self.env.get_condition(state)
        self.model.print_action_predictions(condition)

    def print_parent_predictions(self, state: int, action: int):
        condition = self.env.get_condition(state)
        self.model.print_parent_predictions(condition, action)

    def print_model(self):
        self.model.print_model(self.env)
