import random


class CoinWorld:
    """There are n coins. The action flip-all flips all to heads or all to tails (50% chance of either)"""
    def __init__(self):
        self.n = 2
        self.coins = [False] * self.n

    def reset(self):
        self.coins = [random.random() < 0.5 for _ in range(self.n)]

    def step(self):
        if random.random() < 0.5:
            self.coins = [False] * self.n
        else:
            self.coins = [True] * self.n

    def state(self):
        return self.coins[0], self.coins[1]
