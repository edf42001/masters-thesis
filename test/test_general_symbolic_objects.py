from environment.symbolic_door_world import SymbolicDoorWorld
from environment.symbolic_heist import SymbolicHeist
"""
This file is used to test the system for describing any generic world using symbolic objects and predicates
Currently am testing with the taxi world
"""

if __name__ == "__main__":
    env = SymbolicHeist()

    curr_state = env.get_state()
    print(f"Current state: {env.get_factored_state(curr_state)}")
    objects = env.get_object_list(curr_state)
    print(f"Current objects: {objects}")
