import random

# Simulation of the line environment. Provides an "env" interface
class LineEnv(object):
    def __init__(self):
        self.state = 0

    def step(self, action):
        # step(action) -> (next_state,reward,is_terminal,debug_info)
        # If the agent takes action "A" (0) they progress down the line
        # with probability 0.9, otherwise they slide back to state 0.
        # Action "B" (1) takes them back to state 0. At the end of the
        # line they get a +10 reward and return to state 0.
        
        reward = 0  # The returned reward
        prev_state = self.state  # Previous state
        is_terminal = False
        debug = 0

        if action == 1:
            self.state = 0
        elif action == 0:
            if self.state == 4:
                self.state = 0
                reward = 10
            elif random.random() < 0.9:
                self.state += 1
            else:
                self.state = 0
        
        # Return these variables to matches ai gym format
        return self.state, reward, is_terminal, debug
