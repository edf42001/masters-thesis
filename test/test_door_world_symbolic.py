import logging
import random
import numpy as np

from environment.symbolic_door_world import SymbolicDoorWorld
from symbolic_stochastic_domains.symbolic_classes import Example, ExampleSet

if __name__ == "__main__":
    random.seed(1)

    env = SymbolicDoorWorld()

    print(env.visualize())

    example_set = ExampleSet()

    for i in range(20):
        literals = env.get_literals(env.curr_state)
        action = random.randint(0, env.get_num_actions()-1)

        obs = env.step(action)

        print(f"Took action {action}")
        print(env.visualize())
        print(f"Observed {obs}")

        example = Example(action, literals, obs)

        print("Training example:")
        print(example)
        print("-----")

        example_set.add_example(example)

        print("Example set:")
        print(example_set)
        print()
