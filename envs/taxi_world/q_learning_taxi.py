import numpy as np

from envs.taxi_world.taxi_world_env import TaxiWorldEnv, ACTION


# METHODS
def select_action(q_table, state, epsilon):
    if np.random.random() < epsilon:
        # Pick random action
        return np.random.choice(list(ACTION))
    else:
        # Pick best action
        index = state_hash(state)
        return ACTION(np.argmax(q_table[index]))


def state_hash(state):
    # 5 for taxi x, 5 for taxi y, 5 for current passenger, 4 for destination
    # Need to combine pickup location (index 2) with passenger in taxi (index 4) to make 5 states instead of 8

    # Set this to 4 if passenger is in taxi otherwise 0-3 for the pickup index
    passenger_loc_index = 4 if state[4] else state[2]

    return state[0] + \
           5 * state[1] + \
           25 * passenger_loc_index + \
           125 * state[3]


# BEGIN CODE
# Create the env
env = TaxiWorldEnv()
NUM_STATES = 500  # 25 places taxi can be in, 5 places for passenger (4 pickup, 1 in taxi), 4 places for destination
NUM_ACTIONS = len(list(ACTION))

load = True
if not load:
    q_table = np.zeros((NUM_STATES, NUM_ACTIONS))
else:
    q_table = np.load("../../np_save_data/taxi_world_sarsa_q_values.npy")


# How many iterations to iterate for, and iteration counter
NUM_ITERATIONS = 10000
iterations = 0

# Q learning parameters
if not load:
    epsilon = 0.1
else:
    epsilon = 0

learning_rate = 0.2
discount_rate = 0.6

# Store state
state = env.get_state()
next_state = env.get_state()

running_reward = 0

# Main learning loop
while iterations < NUM_ITERATIONS:
    # Step 1: Observer current state s
    state = env.get_state()

    # Step 2: Choose action a according to exploration policy, based on prediction of T(s' | s, a)
    # returned by predictTransition(s, a)
    action = select_action(q_table, state, epsilon)

    # Step env
    env.draw_taxi(delay=200)
    reward, done = env.step(action)

    # Step 3: observe new state s'
    next_state = env.get_state()

    # Step 4: Update q values
    index = state_hash(state)
    next_index = state_hash(next_state)
    q_table[index, action.value] = (1 - learning_rate) * q_table[index, action.value] + learning_rate * (reward + discount_rate * np.max(q_table[next_index]))

    running_reward = 0.99 * running_reward + 0.01 * reward
    state = next_state
    iterations += 1

    if iterations % 200 == 0:
        print(running_reward)

# Save q values
np.save("../../np_save_data/taxi_world_sarsa_q_values.npy", q_table)
