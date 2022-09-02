"""
Created on 8/30/22 by Ethan Frank

Test script to generate possible permutations of object names in the world given an object map belief
"""

import itertools

if __name__ == "__main__":

    # Objects that are actually currently in the environment
    state_objects = {"idpyo", "dpamn"}

    # Current object map belief
    object_map = {'idpyo': ['pass', 'dest'],
                  'pumzg': ['wall'],
                  'dpamn': ['pass', 'dest', 'wall']}

    mappings_to_choose_from = (object_map[unknown_object] for unknown_object in state_objects)

    permutations = itertools.product(*mappings_to_choose_from)

    for permutation in permutations:
        # Remove ones where there is a duplicate assignment. Two objects can not be mapped to the same
        if len(set(permutation)) != len(permutation):
            continue
        print(permutation)
