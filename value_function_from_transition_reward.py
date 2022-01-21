import numpy as np
import matplotlib.pyplot as plt

# This script is to practice writing code to calculate (with dynamic programming)
# The value funcition of states/actions (q function) of the simple chain environemt
# First, we need a data structure to hold the data. How about an array of states, 1-N, each entry a list of tuples,
# each tuple a next state, and another list for rewards? These are sparse, so anything not mentioned does not exist.
# I include the first state in the tuple just for fun

NUM_STATES = 5
NUM_ACTIONS = 2

# Hashed index method. Good if not sparse?
transitions = [
    # Action 0 transition probabilities
    # (next_state, prob)
    [(1, 0.8), (0, 0.2)],
    [(2, 0.8), (0, 0.2)],
    [(3, 0.8), (0, 0.2)],
    [(4, 0.8), (0, 0.2)],
    [(4, 0.8), (0, 0.2)],

    # Action 1 transition probabilities
    [(0, 0.8), (1, 0.2)],
    [(0, 0.8), (2, 0.2)],
    [(0, 0.8), (3, 0.2)],
    [(0, 0.8), (4, 0.2)],
    [(0, 0.8), (4, 0.2)],
]

rewards = [
    # Action 0 reward probabilities
    # (reward, prob)
    [(0, 0.8), (2, 0.2)],
    [(0, 0.8), (2, 0.2)],
    [(0, 0.8), (2, 0.2)],
    [(0, 0.8), (2, 0.2)],
    [(10, 0.8), (2, 0.2)],

    # Action 1 transition probabilities
    [(2, 0.8), (0, 0.2)],
    [(2, 0.8), (0, 0.2)],
    [(2, 0.8), (0, 0.2)],
    [(2, 0.8), (0, 0.2)],
    [(2, 0.8), (10, 0.2)],
]


def sa_hash(state, action):
    # Index into a 10 size array
    return state + NUM_STATES * action


# Solve this with DP: Iteratively go over each state and do:
# Q(s, a) = E[R(s, a)] + discount * sum_over_next_states(T(s, a, s')*max_over_a(Q(s', a))
q_table = np.zeros(10)

NUM_ITERATIONS = 100  # Iterations to do
discount = 0.90
iterations = 0

q_tables = np.zeros((NUM_ITERATIONS, 10))

while iterations < NUM_ITERATIONS:
    # Each iteration, iterate over each action in the q table
    # from front to back. Does order matter?
    for i in range(len(q_table)):
        # The state/action pair is encoded in i, which can be thought of as already hashed

        # Calculate expected reward
        # For the state/action pair, extract the reward probabilites and values to calculate expected reward
        expected_reward = sum([reward[0] * reward[1] for reward in rewards[i]])

        # Next, sum rewards over next states probabilistically
        transition = transitions[i]
        # TODO: Make max of next states work for when there are more than two next states
        next_expected_q_value = sum([prob * max(q_table[sa_hash(next_state, 0)], q_table[(sa_hash(next_state, 1))])
                                     for (next_state, prob) in transition])
        q_table[i] = expected_reward + discount * next_expected_q_value

    # Store q table history
    q_tables[iterations] = q_table

    iterations += 1

plt.plot(q_tables)
plt.title("Q Table values over iterations")
plt.show()


