import logging
import random
import numpy as np

from environment.symbolic_door_world import SymbolicDoorWorld

if __name__ == "__main__":
    random.seed(1)

    env = SymbolicDoorWorld()

    print(env.visualize())

    for i in range(20):
        action = random.randint(0, env.get_num_actions()-1)
        obs = env.step(action)

        print(f"Took action {action}")
        print(env.visualize())
        # print(f"Observed {obs}")

        print("Current literals")
        literals = env.get_literals(env.curr_state)
        for pred in literals:
            if pred.value:
                print(pred)
        print()
