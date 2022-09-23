"""
Created on 9/23/22 by Ethan Frank

Various helper functions
"""

import random


def random_string_generator(length):
    return ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(length))
