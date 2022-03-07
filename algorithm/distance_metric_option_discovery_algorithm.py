from algorithm.simulator import Simulator
from environment.environment import Environment
from policy.hierarchical_policy import Policy
from policy.samplers.sampler import Sampler


class DistanceMetricOptionDiscoveryAlgorithm(Simulator):

    def __init__(self, env: Environment, policy: Policy, sampler: Sampler, visualize: bool = False):
        self.env = env
        self.policy = policy
        self.sampler = sampler
        self.visualize = visualize

        self.curr_state = self.env.get_state()
        self.last_episode_steps = -1
        self.last_episode_reward = -1

    def run_single_episode(self, max_steps: int, is_learning: bool):
        steps, episode_reward = 0, 0

        # Visualize initial state
        if self.visualize:
            self.env.visualize()

        while steps < max_steps and not self.env.end_of_episode():
            self.curr_state = self.env.get_state()
            curr_action = self.choose_action(is_learning)

            # Perform action and observe next state
            next_state, reward = self.env.step(curr_action)
            # reward = self.env.get_last_reward()
            # next_state = self.env.get_state()
            terminal = self.env.end_of_episode()

            if self.visualize:
                self.env.visualize()

            if is_learning:
                self.policy.add_experience(self.curr_state, curr_action, reward, next_state, terminal)
                self.policy.update()

            # Update vars
            episode_reward += reward
            steps += 1
            self.curr_state = next_state

        self.last_episode_steps = steps
        self.last_episode_reward = episode_reward

    def choose_action(self, is_learning: bool):
        # Decide if explore or exploit
        if is_learning:
            explore = self.sampler.sample()
            self.sampler.update()
        else:
            explore = False

        return self.policy.choose_action(self.curr_state, explore=explore)