import numpy as np
import matplotlib.pyplot as plt


def q_values_from_transition_reward_iterative(transitions, rewards, discount):
    # This script is to practice writing code to calculate (with dynamic programming)
    # The value funcition of states/actions (q function) of the simple chain environemt
    # First, we need a data structure to hold the data:
    # Transitions and rewards are just big 3D arrays for s, a, next state / reward

    # Solve this with DP: Iteratively go over each state and do:
    # Q(s, a) = E[R(s, a)] + discount * sum_over_next_states(T(s, a, s')*max_over_a(Q(s', a))


    NUM_STATES = transitions.shape[0]
    NUM_ACTIONS = transitions.shape[1]
    NUM_REWARDS = rewards.shape[2]

    q_table = np.zeros(NUM_STATES * NUM_ACTIONS)

    NUM_ITERATIONS = 50  # Iterations to do
    iterations = 0

    q_tables = np.zeros((NUM_ITERATIONS, 10))

    while iterations < NUM_ITERATIONS:
        # Each iteration, iterate over each action in the q table
        # from front to back. Does order matter?
        for i in range(len(q_table)):
            s = i % 5
            a = int(i / 5)
            # print(s, a)
            # The state/action pair is encoded in i, which can be thought of as already hashed

            # Calculate expected reward
            # For the state/action pair, extract the reward probabilities and values to calculate expected reward
            # Reward value is the same as the prob index so multiply the probs by the nums 0 to N
            s_a_reward = rewards[s, a, :]
            expected_reward = sum(s_a_reward * np.arange(0, len(s_a_reward)))

            # Next, sum expected q values over next states probabilistically
            next_state_probs = transitions[s, a, :]
            # print(next_state_probs)
            # TODO: Make max of next states work for when there are more than two next states
            next_average_q_value = sum([next_state_probs[i] * max(q_table[[i, i+NUM_STATES]]) for i in range(NUM_STATES)])
            q_table[i] = expected_reward + discount * next_average_q_value

        # Store q table history
        q_tables[iterations] = q_table

        iterations += 1


if __name__ == "__main__":
    print("No op for now")
    # plt.plot(q_tables)
    # plt.title("Q Table values over iterations")
    # plt.show()
