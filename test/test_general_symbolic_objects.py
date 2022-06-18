"""
This file is used to test the system for describing any generic world using symbolic objects and predicates
Currently am testing with the taxi world
"""
import random

from environment.symbolic_heist import SymbolicHeist

if __name__ == "__main__":
    random.seed(1)

    env = SymbolicHeist(stochastic=False, use_outcomes=False)

    curr_state = env.get_state()
    print(f"Current state: {env.get_factored_state(curr_state)}")
    objects = env.get_object_list(curr_state)
    print(f"Current objects: {objects}")
    literals, bindings = env.get_literals(curr_state)
    print(f"Literals: {literals}")
    print(f"Len = {len(literals)}")
    print(f"Num states: {env.get_num_states()}")

    env.draw_world(curr_state, delay=500)

    actions = [env.A_NORTH, env.A_SOUTH, env.A_EAST, env.A_WEST, env.A_NORTH, env.A_PICKUP, env.A_UNLOCK,
               env.A_SOUTH, env.A_EAST, env.A_NORTH, env.A_NORTH, env.A_WEST, env.A_WEST,
               env.A_SOUTH, env.A_UNLOCK, env.A_SOUTH]
    #
    for action in actions:
        observation = env.step(action)
        print(f"Observation: {observation}")
        curr_state = env.get_state()
        literals, bindings = env.get_literals(curr_state)
        print(f"Literals: {literals}")
        print(f"Bindings: {bindings}")
        env.draw_world(curr_state, delay=0)

    # examples = set()
    # for state in range(env.get_num_states()):
    #     literals, bindings = env.get_literals(state)
    #     examples.add(str(literals))

    # Predicates need to know that touchleft(key1) and touchLeft(key2) will have the same effect (dynamic objects)
    # Perhaps the object class and the object id can be stored differently? How will that effect hashing?
    # Making it so all the keys have the same name reduces the number of states from 13568 to only 232
    # Also reduces the len of the literal list from 58 to 22
    # Update: now that I have fixed the predicates, there are 328 unique states. Still pretty small.
    # TODO: expand further. For example, taxi touching a block touching a door which is open. Limit to depth of three
    # This will cause exponential growth, and will increase the space again, but hopefully not more than the 13568
    # Perhaps run the code I have that generates literals, recursively replacing the source object, for each object.
    # print(f"Different states: {len(examples)}")
