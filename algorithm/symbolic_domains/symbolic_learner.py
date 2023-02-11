from algorithm.simulator import Simulator
from algorithm.symbolic_domains.symbolic_model import SymbolicModel
from environment.environment import Environment
from policy.policy import Policy


class SymbolicLearner(Simulator):
    def __init__(self, env: Environment, model: SymbolicModel, planner: Policy, visualize: bool = False, delay: int = 100):
        self.env = env
        self.model = model
        self.planner = planner

        self.visualize = visualize
        self.delay = delay

        self.curr_state = self.env.get_state()
        self.last_episode_steps = -1
        self.last_episode_reward = -1

    def run_single_episode(self, max_steps: int, is_learning: bool):
        steps, total_reward = 0, 0
        while steps < max_steps and not self.env.end_of_episode():
            self.curr_state = self.env.get_state()

            # Choose an action with the policy. Different actions can be taken if learning/executing optimal policy
            action = self.choose_action(is_learning)

            print(f"Step {steps} taking action {action}")

            # Perform action and observe the effects / get next state
            _, observation, _ = self.env.step(action)
            reward = self.env.get_last_reward()
            next_state = self.env.get_state()

            # Display environment if need be
            if self.visualize:
                self.env.visualize(delay=self.delay)

            # TODO: for consistency, this should take in state instead of literals
            if is_learning:
                self.model.add_experience(action, self.curr_state, observation)

            # Update bookkeeping
            total_reward += reward
            steps += 1
            self.curr_state = next_state

        # Final values to return for episode
        self.last_episode_steps = steps
        self.last_episode_reward = total_reward

        print(f"Total reward {self.last_episode_reward}")

    def choose_action(self, is_learning: bool):
        return self.planner.choose_action(self.curr_state, is_learning)
