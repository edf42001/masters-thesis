import logging

from algorithm.simulator import Simulator
from environment.environment import Environment
from algorithm.transition_model import TransitionModel
from policy.policy import Policy


class ActionLearner(Simulator):
    def __init__(self, env: Environment, model: TransitionModel, planner: Policy, visualize: bool = False):
        self.env = env
        self.model = model
        self.planner = planner

        self.visualize = visualize

        self.curr_state = self.env.get_state()
        self.last_episode_steps = -1
        self.last_episode_reward = -1

    def run_single_episode(self, max_steps: int, is_learning: bool):
        steps, total_reward = 0, 0
        while steps < max_steps and not self.env.end_of_episode():
            print(f"Step {steps}")
            self.curr_state = self.env.get_state()

            # Choose an action with the policy. Different actions can be taken if learning/executing optimal policy
            action = self.choose_action(is_learning)

            print(f"Taking action {action}")

            # Perform action and observe the next state
            observation = self.env.step(action)
            reward = self.env.get_last_reward()
            next_state = self.env.get_state()

            # Display environment if need be
            if self.visualize:
                self.env.visualize()

            if is_learning:
                self.model.add_experience(action, self.curr_state, observation)

            # Update bookkeeping
            total_reward += reward
            steps += 1
            self.curr_state = next_state

            # End when we have learned the correct model
            if self.model.has_correct_action_model():
                print("Correct action model learned, stopping early")
                break

        # Final values to return for episode
        self.last_episode_steps = steps
        self.last_episode_reward = total_reward

    def choose_action(self, is_learning: bool):
        return self.planner.choose_action(self.curr_state, is_learning)
