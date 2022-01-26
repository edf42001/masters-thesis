from envs.line_env import LineEnv

import numpy as np
import random
from q_distribution_from_sampled_mdps import generate_q_tables_from_sampled_mpds,\
    estimate_value_of_perfect_information_from_sampled_q_values

import matplotlib.pyplot as plt
import cProfile
import pstats

# Make it easier to read everything
np.set_printoptions(precision=2, suppress=True)

# Create environment
env = LineEnv()

NUM_STATES = env.num_states()  # States in env
NUM_ACTIONS = env.num_actions()  # Actions per state (uniform in this case)
NUM_REWARDS = env.num_rewards()  # Rewards go from 0 to 10

TRAINING_EPOCHS = int(4E2)  # How long to train for
iterations = 0  # How many steps the agent has taken so far

# Number of mdp to sample for naive global sampling
K = 5

# Do we use learning rate or are q values based only on the sampling?
discount = 0.9
learning_rate = 0.1

# Starting state of env
state = env.get_state()

# Store transition and reward counts
transition_table = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES))
reward_table = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_REWARDS))

q_table = np.zeros(NUM_STATES * NUM_ACTIONS)

# Store rewards over time
rewards = np.zeros(TRAINING_EPOCHS)


# Do the training
while iterations < TRAINING_EPOCHS:
    # Create dirilect posteriors from priors (Assume all start at one)
    transition_alphas = transition_table + 0.1
    reward_alphas = reward_table + 0.1

    # From the alphas, generate K MDPs and from those MDPs, calculate their Q table values
    sampled_q_tables = generate_q_tables_from_sampled_mpds(transition_alphas, reward_alphas, discount, K)
    expected_q = sampled_q_tables.mean(axis=0)

    # We now have a distribution of q values. From these, calculate
    # the Value of Perfect Information (for all q values? Just for this state? Can they be reused?)
    VPI = estimate_value_of_perfect_information_from_sampled_q_values(sampled_q_tables)

    # TODO: Compare to basic q learning. Make faster. Check if q values are approaching what they should
    # Ask prof ray for advice? Implement loop domain. Check other implementations and compare.

    # Want to maximize vpi - cost of choosing vpi action =
    # maximize VPI + E[q(s, a)]
    # print(expected_q)
    # print(VPI)
    action = np.argmax((expected_q + VPI).reshape(NUM_ACTIONS, NUM_STATES)[:, state])

    # Choose action randomly for testing purposes
    # action = random.randint(0, 1)

    # Step environment
    next_state, reward, _, _ = env.step(action)

    # Q table bellman update
    q_index = state + NUM_STATES * action
    next_best_q = max(q_table[next_state], q_table[next_state + NUM_STATES])
    q_table[q_index] = (1 - learning_rate) * q_table[q_index] + learning_rate * (reward + discount * next_best_q)

    print("{}: {}->{}, {}".format(action, state, next_state, reward))

    # Store the number of transitions and rewards discovered
    transition_table[state, action, next_state] += 1
    reward_table[state, action, reward] += 1

    # Update current state
    state = next_state

    # Store rewards over time
    rewards[iterations] = reward
    iterations += 1


print(sum(rewards))
plt.plot(rewards)
plt.show()

# To read this reshaped table, imagine it's a 5x5 (ignore the two rows per row). Across top is first state,
# then down is second state, the row is the action (0 or 1). Perhaps this should be reordered (action, state, state)?
# Interesting, not the reshape goes in the opposite direction from the order the multiplication for the hash is done
# print(transition_parameter_table.reshape((NUM_STATES, NUM_ACTIONS, NUM_STATES)))
# print(reward_parameter_table.reshape((NUM_REWARDS, NUM_ACTIONS, NUM_STATES)))
# print("Parameter tables detailing reward and transition counts")
# print(transition_parameter_table)
# print(reward_parameter_table)

# To extract the transition probabilites for a single state/action pair,
# take the subset of transition_dirichlet_alphas like so: