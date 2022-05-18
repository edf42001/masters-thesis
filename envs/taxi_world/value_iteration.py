import numpy as np
import pickle

from envs.taxi_world.taxi_world_env import ACTION
# from envs.taxi_world.doormax_taxi import DoormaxTaxi

import cProfile
import pstats

"""
Does value iteration on doormax object oriented taxi world transition probabilities
to solve the whole mdp. Or just from this state??
"""


def solve_mdp_value_iteration(curr_state, doormax, discount_rate):
    NUM_STATES = doormax.env.get_num_states()  # x, y, passenger location (+ in taxi), goal location
    NUM_ACTIONS = doormax.env.get_num_actions()

    curr_state_hash = doormax.env.state_hash(curr_state)

    # Store values from value iteration
    values = np.zeros(NUM_STATES)

    # Do iteration until max change is small
    epsilon = 0.1
    iterations = 0  # Iterations counter
    max_change = 100
    print("Value iteration current state: {}".format(curr_state))
    while max_change > epsilon:
        max_change = 0

        # For every state, use bellman equation to update its value
        for s in range(NUM_STATES):
            current_value = values[s]

            # New value of this state is the max reward it can get for moving + discount value of that next state
            state = doormax.env.reverse_state_hash(s, pickup=curr_state[2])

            # # Hack to fix the fact that passenger in taxi passenger location are weird together, leading to bugs
            # # where it is set to 0 but this messes with the conditions
            # if state[4]:
            #     state[2] = curr_state[2]

            # Hacky way to check: Is this a state where the current goal / pickup is different?
            # in which case we don't care and don't have to propogate the values because transition prob to these states
            # is 0
            if state[3] != curr_state[3] or state[2] != curr_state[2]:
                # print(state)
                continue

            # Predict next states with object oriented transitions, or None for unknown
            next_states = doormax.predict_next_states(state)

            # if s == curr_state_hash:
                # print("At current state {}, {}! Next states: {}".format(state, curr_state, next_states))

            next_values = np.zeros(NUM_ACTIONS)
            for action in list(ACTION):
                next_state = next_states[action]

                # We can't predict the transition, assume max reward (optimism in face of uncertainty)
                if next_state is None:
                    next_values[action.value] = discount_rate * 15
                    # print("State was none! {}: {}->{}".format(action, state, next_state))
                    continue

                # # HACK AGAIN BECAUSE NOW HE's looping around on himself (the run ends when you deliver)
                if action == ACTION.DROPOFF and state[4] and not next_state[4]:
                    # TODO? WHich?
                    # next_values[action.value] = discount_rate * doormax.rewards[s, action.value]
                    next_values[action.value] = doormax.rewards[s, action.value]
                else:
                    # Otherwise, what is the reward for this transition + discounted value of next state
                    next_values[action.value] = doormax.rewards[s, action.value] + discount_rate * values[doormax.env.state_hash(next_state)]

                # if s == doormax.env.state_hash((0, 0, curr_state[2], curr_state[3], False)):
                    # print("a, r, v, s: {}, {}, {}, {}".format(action, doormax.rewards[s, action.value], values[doormax.env.state_hash(next_state)], next_state))

            # if s == curr_state_hash:
            # print("Current, Next values: {}, {}".format(current_value, next_values))
            optimal_value = np.max(next_values)
            values[s] = optimal_value

            change = abs(optimal_value - current_value)
            if change > max_change:
                # print("Change is big: {}, {}, {}: {}".format(next_values, optimal_value, current_value, state))
                max_change = change

        # debug_values_false = np.zeros((5, 5))
        # debug_values_true = np.zeros((5, 5))
        # for i in range(0, 5):
        #     for j in range(0, 5):
        #         state = (i, j, curr_state[2], curr_state[3], False)
        #         debug_values_false[4-j, i] = values[doormax.env.state_hash(state)]
        #         state = (i, j, curr_state[2], curr_state[3], True)
        #         debug_values_true[4-j, i] = values[doormax.env.state_hash(state)]
        #
        # print(debug_values_false)
        # print(debug_values_true)
        # print(max_change)
        iterations += 1

    return values


if __name__ == "__main__":
    # Load our object for testing
    with open("doormax.pkl", "rb") as f:
        doormax = pickle.load(f)

    # Try running a value iteration solver for optimal actions
    state = doormax.env.get_state()
    print("Current state: {}".format(state))

    discount_rate = 0.5

    # np.random.seed(seed=None)
    # number = np.random.randint(0, 500)
    # state = doormax.env.reverse_state_hash(number)
    # state_hash = doormax.env.state_hash(state)
    # print(number, state, state_hash)

    # profiler = cProfile.Profile()
    # profiler.enable()
    import time
    start = time.time()
    values = solve_mdp_value_iteration(state, doormax, 0.9)
    print("end: {}".format(time.time() - start))

    next_states = doormax.predict_next_states(state)
    for action in list(ACTION):
        next_state = next_states[action]
        print(values[doormax.env.state_hash(next_state)])

    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.dump_stats('solve_mdp_value_iteration.prof')

