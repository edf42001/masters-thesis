import random

from environment.symbolic_door_world import SymbolicDoorWorld

if __name__ == "__main__":
    random.seed(1)

    env = SymbolicDoorWorld(stochastic=False)

    for i in range(env.get_num_states()):
        env.restart(init_state=env.get_factored_state(i))
        print("State: ")
        print(env.visualize())

        print(f"Literals: {[lit for lit in env.get_literals(i) if lit.value]}")

        print("Result of left action")
        env.step(0)
        print(env.visualize())

        env.restart(init_state=env.get_factored_state(i))
        print("Result of right action")
        env.step(1)
        print(env.visualize())
        print()
