
from matplotlib import pyplot as plt

from runners.runner import Runner


class Plot:

    def __init__(self, runner: Runner, eval_episodes: int, eval_timer: int, eval_states=None):
        """Creates 4 plots:
             * training reward against episode number
             * training steps against episode number
             * avg eval reward against episode number
             * avg eval reward against episode number
            If eval_states is provided, run eval episodes starting from given states
        """
        self.runner = runner
        self.eval_episodes = eval_episodes
        self.eval_timer = eval_timer
        self.eval_time = eval_timer
        self.eval_states = eval_states

        if eval_states:
            self.eval_episodes = len(eval_states)

        self.train_steps = []
        self.train_rewards = []

        self.eval_steps = []
        self.eval_rewards = []

        self.cumulative_steps = []
        self.cumulative_reward = []
        self.steps_between_eval = 0

        # Turned off plots for now
        # plt.ion()
        # fig = plt.figure()
        #
        # self.ax_train_r = fig.add_subplot(221)
        # self.ax_train_r.set_title('Train: Cumulative reward')
        #
        # self.ax_train_s = fig.add_subplot(222)
        # self.ax_train_s.set_title('Train: Episode lengths')
        #
        # self.ax_eval_r = fig.add_subplot(223)
        # self.ax_eval_r.set_title('Eval: Cumulative reward')
        #
        # self.ax_eval_s = fig.add_subplot(224)
        # self.ax_eval_s.set_title('Eval: Episode lengths')

    def update(self, steps: int, reward: float):
        """Provide results of training step. If eval timer is 0, runs evaluation using current policy"""
        self.train_rewards.append(reward)
        self.train_steps.append(steps)
        self.steps_between_eval += steps

        # self.ax_train_r.clear()
        # self.ax_train_r.plot(self.train_rewards)
        # self.ax_train_r.set_title('Train: Cumulative reward')
        #
        # self.ax_train_s.clear()
        # self.ax_train_s.plot(self.train_steps)
        # self.ax_train_s.set_title('Train: Episode lengths')
        #
        # plt.pause(0.02)

        self.eval_time -= 1
        if self.eval_time == 0:
            self.eval_time = self.eval_timer

            total_steps, total_reward = 0, 0
            if self.eval_states:
                for state in self.eval_states:
                    s, r = self.runner.run_episode(False, max_steps=1, init_state=state)
                    total_steps += s
                    total_reward += r
            else:
                for _ in range(self.eval_episodes):
                    s, r = self.runner.run_episode(False)
                    total_steps += s
                    total_reward += r

            total_reward /= self.eval_episodes
            total_steps /= self.eval_episodes

            self.eval_rewards.append(total_reward)
            self.eval_steps.append(total_steps)

            prev_steps = 0 if not self.cumulative_steps else self.cumulative_steps[-1]
            self.cumulative_steps.append(prev_steps + self.steps_between_eval)
            best_reward = total_reward if not self.cumulative_reward else max(total_reward, self.cumulative_reward[-1])
            self.cumulative_reward.append(best_reward)
            self.steps_between_eval = 0

            # self.ax_eval_r.clear()
            # self.ax_eval_r.plot(self.cumulative_steps, self.eval_rewards)
            # self.ax_eval_r.set_title('Eval: Cumulative reward')
            #
            # self.ax_eval_s.clear()
            # self.ax_eval_s.plot(self.cumulative_steps, self.eval_steps)
            # self.ax_eval_s.set_title('Eval: Episode lengths')
            #
            # plt.pause(0.02)

    def finalize(self, show: bool = False):
        # plt.ioff()
        # if show:
        #     plt.show()
        # plt.close()

        return self.cumulative_reward, self.cumulative_steps

    def get_training(self):
        return self.train_rewards, self.train_steps
