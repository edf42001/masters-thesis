import random
import graphviz
import time

from environment.sokoban_world import SokobanWorld

if __name__ == "__main__":
    # For testing
    random.seed(1)

    env = SokobanWorld(stochastic=False)


    state = env.get_state()
    print(state)
    factored_state = env.get_factored_state(state)
    print(factored_state)
    new_state = env.get_flat_state(factored_state)
    print(state)

    for i in range(5):
        env.visualize()
        env.draw_world(env.get_factored_state(env.get_state()), delay=2000)
        action = random.randint(0, env.get_num_actions()-1)
        env.step(action)


