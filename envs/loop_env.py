import random


# Simulation of the Loop environment. Provides an "env" interface
class LoopEnv(object):
    def __init__(self):
        self.state = 0

    def step(self, action):
        # step(action) -> (next_state,reward,is_terminal,debug_info)
        # In the loop world there are two loops of length 5 joined in the center state.
        # If the agent takes action A, they go into the right loop, at which point all actions lead
        # around the loop until it gets back to the start, where it gets a reward of 1
        # If the agent takes B, it goes to the left loop, all actions B take it around the loop until it gets
        # to the start and receives reward of 2, but actions A will take it back to start with 0 reward.
        # Right loop is 0-4, left is 0, 5-8.

        reward = 0  # The returned reward
        is_terminal = False
        debug = 0  # Debug information, we don't use it, but need to match format

        # Branching point
        if self.state == 0:
            if action == 0:
                self.state = 1
            elif action == 1:
                self.state = 5
        # Right loop
        elif self.state == 1 or self.state == 2 or self.state == 3:
            self.state += 1

        # Last state of right loop
        elif self.state == 4:
            self.state = 0

            # Get reward for completing loop
            reward = 1

        # Left loop
        elif self.state == 5 or self.state == 6 or self.state == 7:
            if action == 0:
                self.state = 0
            elif action == 1:
                self.state += 1

        # Last state of left loop
        elif self.state == 8:
            self.state = 0

            # Reward for going around left loop
            reward = 2

        # Return these variables to matches ai gym format
        return self.state, reward, is_terminal, debug

    def get_state(self):
        return self.state

    def num_states(self):
        return 9

    def num_actions(self):
        return 2

    def num_rewards(self):
        return 3


if __name__ == "__main__":
    env = LoopEnv()
    state = env.get_state()

    # Test the env
    for i in range(30):
        action = random.randint(0, 1)
        next_state, reward, _, _ = env.step(action)

        print("{}: {}->{}, {}".format(action, state, next_state, reward))

        state = next_state
