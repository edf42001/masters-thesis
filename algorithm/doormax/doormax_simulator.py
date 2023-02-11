import logging

from algorithm.simulator import Simulator
from environment.environment import Environment
from policy.policy import Policy


class DoormaxSimulator(Simulator):
    def __init__(self, env: Environment, model, planner: Policy, visualize: bool = False):
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
            self.curr_state = self.env.get_state()

            # Choose an action with the policy. Different actions can be taken if learning/executing optimal policy
            action = self.choose_action(is_learning)

            # Perform action and observe the next state
            observation = self.env.step(action)
            reward = self.env.get_last_reward()
            next_state = self.env.get_state()

            logging.debug(self.env.get_action_name(action))
            logging.debug(self.env.get_condition(self.curr_state))
            logging.debug(observation)

            # Display environment if need be
            if self.visualize:
                self.env.visualize()

            if is_learning:
                self.model.add_experience(action, self.curr_state, observation)

            # Update bookkeeping
            total_reward += reward
            steps += 1
            self.curr_state = next_state

        # Final values to return for episode
        self.last_episode_steps = steps
        self.last_episode_reward = total_reward

        print("Final model:")
        self.model.print_model()

    def choose_action(self, is_learning: bool):
        return self.planner.choose_action(self.curr_state, is_learning)
