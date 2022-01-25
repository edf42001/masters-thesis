from envs.line_env import LineEnv

import numpy as np
import random
from value_function_from_transition_reward import q_values_from_transition_reward_iterative
from q_distribution_from_sampled_mdps import sample_mdp_from_params

import matplotlib.pyplot as plt

# Make it easier to read everything
np.set_printoptions(precision=2, suppress=True)


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

transition_parameter_table = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES))
reward_parameter_table = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_REWARDS))

# TODO: THis needs to be sparsified
transition_dirichlet_alphas = np.ones((NUM_STATES, NUM_ACTIONS, NUM_STATES))
reward_dirichlet_alphas = np.ones((NUM_STATES, NUM_ACTIONS, NUM_REWARDS))

while iterations < TRAINING_EPOCHS:

    # Choose action randomly for testing purposes
    action = random.randint(0, 1)

    # Step environment
    next_state, reward, _, _ = env.step(action)

    # print("{}: {}->{}, {}".format(action, state, next_state, reward))

    # These currently store the same info, will they be different later?
    transition_parameter_table[state, action, next_state] += 1
    transition_dirichlet_alphas[state, action, next_state] += 1

    reward_parameter_table[state, action, reward] += 1
    reward_dirichlet_alphas[state, action, reward] += 1

    # Update current state
    state = next_state

    iterations += 1


# To read this reshaped table, imagine it's a 5x5 (ignore the two rows per row). Across top is first state,
# then down is second state, the row is the action (0 or 1). Perhaps this should be reordered (action, state, state)?
# Interesting, not the reshape goes in the opposite direction from the order the multiplication for the hash is done
# print(transition_parameter_table.reshape((NUM_STATES, NUM_ACTIONS, NUM_STATES)))
# print(reward_parameter_table.reshape((NUM_REWARDS, NUM_ACTIONS, NUM_STATES)))
print("Parameter tables detailing reward and transition counts")
print(transition_parameter_table)
print(reward_parameter_table)

# To extract the transition probabilites for a single state/action pair,
# take the subset of transition_dirichlet_alphas like so:

# Generate samples:
# print("Sampled mdps:")
# print(transition_dirichlet_alphas)
# for i in range(4):
#     mdp_sample = sample_mdp_from_params(transition_dirichlet_alphas, NUM_STATES, NUM_ACTIONS)
#     print(mdp_sample)


transitions = sample_mdp_from_params(transition_dirichlet_alphas)
rewards = sample_mdp_from_params(reward_dirichlet_alphas)
np.save('np_save_data/transitions.npy', transitions)
np.save('np_save_data/rewards.npy', rewards)

np.save('np_save_data/transition_dirichlet_alphas.npy', transition_dirichlet_alphas)
np.save('np_save_data/reward_dirichlet_alphas.npy', reward_dirichlet_alphas)

discount = 0.9
print(transitions)
print(rewards)
q_values = q_values_from_transition_reward_iterative(transitions, rewards, discount)
print('Q values')
print(q_values)
print('Baseline Q values')
print(np.load('np_save_data/true_q_values_gamma_0_9.npy'))
# Could also sample whole thing at once, then normalize?
# Or is each transition probability supposed to be independent?


# Work on value of imperfect information here
# If bad action becomes better, gain is q*(s,a) - E[q(s, a)]
# If a good action becomes worse, gain of performing other action is E[q(s, a2)] - q*(s, a1))
# Cost of this action is difference between it and best action
# Thus, try to maximize VPI(s, a) - max_a'(E(q) - stuff)
# Because max action value is always the same, try to maximize
# E[q(s, a)] + VPI(s, a)


def calc_value_perfect_information(s, a):
    # Sum of gains over all different values the new q values could be (x)
    # weighted by the probability that it is that value
    # TODO
    return 0
