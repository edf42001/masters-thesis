import pickle
import time
from typing import Tuple


class Runner:
    def __init__(self):
        self.name = None
        self.pkl_name = None
        self.exp_num = None

        self.max_steps = None
        self.num_episodes = None
        self.eval_episodes = None
        self.eval_timer = None
        self.visualize = None

        self.env = None
        self.policy = None
        self.sampler = None
        self.learner = None
        self.plot = None

    def run_experiment(self, save_policy: bool = False, show: bool = False, save_training: bool = False):
        total_steps = 0
        start_time = time.time()
        for _ in range(self.num_episodes):
            steps, reward = self.run_episode(True)
            # self.plot.update(steps, reward)
            total_steps += steps
        end_time = time.time()

        if save_policy:
            self.policy.save(f'{self.name}_policy.npy')

        print('----Experiment Complete----')
        print(f'Total steps:        {total_steps}')
        print('Steps per second: {:.2f}'.format(total_steps / (end_time - start_time)))

        rewards, steps = self.plot.finalize(show=show)
        with open(f'{self.name}_{self.pkl_name}_{self.exp_num}.pkl', 'wb') as f:
            pickle.dump((rewards, steps), f)

        if save_training:
            rewards, steps = self.plot.get_training()
            with open(f'training_{self.name}_{self.pkl_name}_{self.exp_num}.pkl', 'wb') as f:
                pickle.dump((rewards, steps), f)

    def run_episode(self, is_learning: bool, max_steps: int = None) -> Tuple[int, float]:
        """Runs an episode and returns steps taken and reward"""

        self.env.restart()
        max_steps = max_steps or self.max_steps

        start_time = time.time()
        self.learner.run_single_episode(max_steps, is_learning=is_learning)
        end_time = time.time()

        steps = self.learner.get_last_episode_steps()
        reward = self.learner.get_last_episode_reward()

        if is_learning:
            print(f'Steps:            {steps}')
            print('Steps per second: {:.2f}'.format(steps / (end_time - start_time)))

        return steps, reward
