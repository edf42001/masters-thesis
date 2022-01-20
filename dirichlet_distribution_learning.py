from envs.line_env import LineEnv

import numpy as np
import random
import matplotlib.pyplot as plt


def transition_hash(s, a, s_p):
    # Parameters of the MDP are for each state, action, next state probability
    # These need to be stored in a table so we hash each s, a, s_p
    # This will assign a unique int to each combo
    return s + \
           a * NUM_STATES + \
           s_p * NUM_STATES * NUM_ACTIONS


def reward_hash(s, a, r):
    # How to store these values? Can't rewards be any real number? Or are they discrete?
    # But then don't we still need a range on them? What about negative numbers?
    # I guess let's just use discrete bins from 0 to 10? The bins could be smaller and shifted?
    # Seems like a waste of space but ok
    return s + \
           a * NUM_STATES + \
           r * NUM_STATES * NUM_ACTIONS


env = LineEnv()

NUM_STATES = 5  # States in env
NUM_ACTIONS = 2  # Actions per state (uniform in this case)
NUM_REWARDS = 11  # Rewards go from 0 to 10


TRAINING_EPOCHS = int(5E3)  # How long to train for
iterations = 0  # How many steps the agent has taken

# Starting state of env
state = env.get_state()

# TODO: THis needs to be sparsified
transition_dirichlet_alphas = np.ones(NUM_STATES * NUM_ACTIONS * NUM_STATES)
reward_dirichlet_alphas = np.ones(NUM_STATES * NUM_ACTIONS * NUM_REWARDS)

transition_parameter_table = np.zeros(NUM_STATES * NUM_ACTIONS * NUM_STATES)
reward_parameter_table = np.zeros(NUM_STATES * NUM_ACTIONS * NUM_REWARDS)

while iterations < TRAINING_EPOCHS:

    # Choose action randomly for testing purposes
    action = random.randint(0, 1)

    # Step environment
    next_state, reward, _, _ = env.step(action)

    # print("{}: {}->{}, {}".format(action, state, next_state, reward))

    # These currently store the same info, will they be different later?
    transition_parameter_table[transition_hash(state, action, next_state)] += 1
    transition_dirichlet_alphas[transition_hash(state, action, next_state)] += 1

    reward_parameter_table[reward_hash(state, action, reward)] += 1
    reward_dirichlet_alphas[transition_hash(state, action, next_state)] += 1

    # Update current state
    state = next_state

    iterations += 1


# To read this reshaped table, imagine it's a 5x5 (ignore the two rows per row). Across top is first state,
# then down is second state, the row is the action (0 or 1). Perhaps this should be reordered (action, state, state)?
# Interesting, not the reshape goes in the opposite direction from the order the multiplication for the hash is done
print(transition_parameter_table.reshape((NUM_STATES, NUM_ACTIONS, NUM_STATES)))
print(reward_parameter_table.reshape((NUM_REWARDS, NUM_ACTIONS, NUM_STATES)))

mdp_sample = np.random.dirichlet(transition_dirichlet_alphas)
print(mdp_sample.reshape((NUM_STATES, NUM_ACTIONS, NUM_STATES)))
print(sum(mdp_sample))
