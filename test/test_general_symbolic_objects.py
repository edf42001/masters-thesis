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
    literals = env.get_literals(curr_state)
    print(f"Literals: {literals}")
    print(f"Len = {len(literals)}")
    print(f"Num states: {env.get_num_states()}")

    examples = set()
    for state in range(env.get_num_states()):
        literals = env.get_literals(state)
        examples.add(str(literals))

    # TODO: I think there are issues with my open and on code.
    # Predicates need to know that touchleft(key1) and touchLeft(key2) will have the same effect (dynamic objects)
    # Perhaps the object class and the object id can be stored differently? How will that effect hashing?
    # Making it so all the keys have the same name reduces the number of states from 13568 to only 232
    # Also reduces the len of the literal list from 58 to 22
    print(f"Different states: {len(examples)}")
