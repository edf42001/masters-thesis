import numpy as np
import matplotlib.pyplot as plt
# import cProfile
# import pstats


from value_function_from_transition_reward import q_values_from_transition_reward_iterative


def sample_mdp_from_params(params):
    # This function samples a transition table from dirichlet distribution
    # This can be used for either transition or reward probabilities.
    # For each transition, need to sample separately. Sadly, looks like we need a for loop for that right now :(

    mdp_sample = np.zeros_like(params)
    for i in range(params.shape[0]):
        for j in range(params.shape[1]):
            mdp_sample[i, j, :] = np.random.dirichlet(params[i, j, :])

    return mdp_sample


def calculate_q_distribution_from_sampled_mpds(transition_alphas, reward_alphas, discount, K):
    # Sample K mdps from a dircichlet distribution based on the transitions and rewards tables
    # Uses the results to calculate q value distributions, and returns the results
    # To test, we do lots of plots

    NUM_STATES = transition_alphas.shape[0]
    NUM_ACTIONS = transition_alphas.shape[1]

    # Store K mdps, in list pairs one for transition one for rewards
    mdp_transitions = []
    mdp_rewards = []

    # To store the resulting q values for these mdps
    q_tables = np.zeros((K, NUM_STATES * NUM_ACTIONS))

    # Sample K mdps
    for _ in range(K):
        mdp_transitions.append(sample_mdp_from_params(transition_alphas))
        mdp_rewards.append(sample_mdp_from_params(reward_alphas))

    for k in range(K):
        q_tables[k] = q_values_from_transition_reward_iterative(mdp_transitions[k], mdp_rewards[k], discount)

    # Calculate mean and variance. Very basic analysis.
    mean = q_tables.mean(axis=0)
    variance = q_tables.var(axis=0)

    # Return the q tables and data so we can analyze the distribution
    return q_tables, mean, variance


if __name__ == "__main__":

    # Make it easier to read everything
    np.set_printoptions(precision=2, suppress=True)

    # profiler = cProfile.Profile()
    # profiler.enable()

    # Load test data
    transition_alphas = np.load('np_save_data/transition_dirichlet_alphas.npy')
    reward_alphas = np.load('np_save_data/reward_dirichlet_alphas.npy')
    discount = 0.9
    K = 50

    q_tables, mean, var = calculate_q_distribution_from_sampled_mpds(transition_alphas, reward_alphas, discount, K)

    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.dump_stats('stats.prof')

    print(q_tables)

    # Calculate mean
    print(q_tables.mean(axis=0))
    print(q_tables.var(axis=0))
