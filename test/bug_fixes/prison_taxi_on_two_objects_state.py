"""
Created on 3/7/23 by Ethan Frank

Created to test and debug issue https://github.com/edf42001/masters-thesis/issues/23
"""
from environment.prison_world import Prison


def main():
    env = Prison()
    # state = 373018
    state = 1306138  # The state before the agent picks up the destination

    factored_state = env.get_factored_state(state)
    literals = env.get_literals(state)
    print(factored_state)
    print(literals)
    print(env.STATE_ARITIES)
    env.restart(factored_state)
    env.visualize(2000)


if __name__ == "__main__":
    main()
