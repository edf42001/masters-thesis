class Simulator:

    last_episode_steps: int = None
    last_episode_reward: int = None

    def run_single_episode(self, max_steps: int, is_learning: bool):
        raise NotImplementedError()

    def choose_action(self, is_learning: bool) -> int:
        raise NotImplementedError()

    def get_last_episode_steps(self):
        return self.last_episode_steps

    def get_last_episode_reward(self):
        return self.last_episode_reward
