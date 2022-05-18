class Policy:
    def add_experience(self, state: int, action: int, reward: float, next_state: int, terminal: bool):
        raise NotImplementedError()

    def choose_action(self, curr_state: int, is_learning: bool = True) -> int:
        raise NotImplementedError()

    def update(self):
        # Optional hook for updating learning hyperparameters
        pass
