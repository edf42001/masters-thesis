import random

# Simulation of the line environment. Provides an "env" interface
class LineEnv(object):
    def __init__(self):
        self.state = 0

    def step(self, action):
        # step(action) -> (next_state,reward,is_terminal,debug_info)
        # The agent chooses action A or B. With prob 80%, it does the action it chooses,
        # 20% it does the other. A brings it towards the end of the chain, whre it can get +10.
        # B takes it to the beginning at it gets only +2
        # Action will be 0 for A or 1 for B
        
        reward = 0  # The returned reward
        is_terminal = False
        debug = 0  # Debug information, we don't use it, but need to match format

        # Swap action between 0-1 if it "slips"
        if random.random() < 0.2:
            action = 1 - action

        if action == 1:
            self.state = 0
            reward = 2
        else:
            if self.state == 4:
                self.state = 4
                reward = 10
            else:
                self.state += 1
                reward = 0
        
        # Return these variables to matches ai gym format
        return self.state, reward, is_terminal, debug
