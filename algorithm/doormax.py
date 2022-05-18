import random

from algorithm.simulator import Simulator
from environment.environment import Environment


class Doormax(Simulator):
    def __init__(self, env: Environment, visualize: bool = False, all_goals: bool = False):
        self.env = env
        self.visualize = visualize
        self.all_goals = all_goals

        self.curr_state = self.env.get_state()
        self.last_episode_steps = -1
        self.last_episode_reward = -1
        self.max_steps = -1
        self.is_learning = True

    def run_single_episode(self, max_steps: int, is_learning: bool):
        steps, total_reward = 0, 0

        while steps < max_steps and not self.env.end_of_episode():
            self.curr_state = self.env.get_state()

            # Choose an action with the policy. Different actions can be taken if learning/executing optimal policy
            action = self.choose_action(is_learning)
            print(self.env.get_action_name(action))
            # Perform action and observe the next state
            self.env.step(action)
            reward = self.env.get_last_reward()
            next_state = self.env.get_state()

            # Display environment if need be
            if self.visualize:
                self.env.visualize()

            # Update bookkeeping
            total_reward += reward
            steps += 1
            self.curr_state = next_state

        # Final values to return for episode
        self.last_episode_steps = steps
        self.last_episode_reward = total_reward

    def choose_action(self, is_learning: bool):
        # # Decide if explore or exploit
        # if is_learning:
        #     explore = self.the_sampler.sample()
        #     self.the_sampler.update()
        # else:
        #     explore = False
        #
        # return self.the_policy.choose_action(self.curr_state, explore=explore)
        # Random action for now
        return random.randint(0, self.env.NUM_ACTIONS - 1)

