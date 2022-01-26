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


def generate_q_tables_from_sampled_mpds(transition_alphas, reward_alphas, discount, K):
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

    # Return the q tables and data so we can analyze the distribution
    return q_tables


def estimate_value_of_perfect_information_from_sampled_q_values(sampled_q_values, NUM_STATES):
    # Caluclate the mean of the samples. Is this what we assume the "current" q values are?
    # Or do we get "current" q values from regular q leaning?
    sampled_q_values_mean = sampled_q_values.mean(axis=0)

    NUM_ACTIONS = int(sampled_q_values.shape[1] / NUM_STATES)

    # Store gains for each sampled mdp
    gains = np.zeros_like(sampled_q_values)

    # print("Sampled and current q values")
    # print(sampled_q_values)
    # print(sampled_q_values_mean)
    # print()

    for i in range(len(sampled_q_values_mean)):
        state = i % NUM_STATES
        action = int(i / NUM_STATES)
        # print("s: {}, a: {}".format(state, action))
        assumed_q_values = sampled_q_values[:, state + NUM_STATES * action]
        current_q_value = sampled_q_values_mean[state + NUM_STATES * action]

        # print("Assumed values", assumed_q_values)
        # print("current q value", current_q_value)

        # Q values of actions available in this state
        q_values_actions = sampled_q_values_mean.reshape(NUM_ACTIONS, NUM_STATES)[:, state]
        # print("q_values_actions", q_values_actions)

        # Sort actions to find best ones. (don't need to sort whole array?) (values ordered from least to greatest)
        sorted_actions = np.argsort(q_values_actions)
        best_action = sorted_actions[-1]
        second_best_action = sorted_actions[-2]
        best_q = sampled_q_values_mean[state + NUM_STATES * best_action]
        second_best_q = sampled_q_values_mean[state + NUM_STATES * second_best_action]

        for k in range(len(assumed_q_values)):
            gain = calculate_information_gain(action, assumed_q_values[k], best_action, best_q, second_best_q)
            gains[k, state + NUM_STATES * action] = gain

        # print("gains", gains[:, state + 5*action])
        # print()

    # print("gains")
    # print(gains)

    # Return average expected value of information for each state
    expected_vpi = gains.mean(axis=0)
    # print(expected_vpi)

    return expected_vpi


def calculate_information_gain(action, assumed_q, best_a, best_q, second_best_q):
    # a1 and a2 are best and second best actions
    if action == best_a and assumed_q < second_best_q:
        # and q is less than second best (otherwise best would still be best)
        gain = second_best_q - assumed_q
    elif action != best_a and assumed_q > best_q:
        # and new value is better than best action (otherwise it would stay the same)
        gain = assumed_q - best_q
    else:
        gain = 0

    return gain


if __name__ == "__main__":
    # Make it easier to read everything
    np.set_printoptions(precision=2, suppress=True)

    # profiler = cProfile.Profile()
    # profiler.enable()

    # Load test data
    transition_alphas = np.load('np_save_data/transition_dirichlet_alphas.npy')
    reward_alphas = np.load('np_save_data/reward_dirichlet_alphas.npy')
    NUM_STATES = transition_alphas.shape[0]
    discount = 0.9
    K = 5

    sampled_q_tables = generate_q_tables_from_sampled_mpds(transition_alphas, reward_alphas, discount, K)

    current_q_values = np.array([20, 25, 30, 34, 39, 23, 24, 24, 25, 28])
    VPI = estimate_value_of_perfect_information_from_sampled_q_values(sampled_q_tables, NUM_STATES)

    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.dump_stats('stats.prof')

    # print(q_tables)

    # Calculate mean
    # print(q_tables.mean(axis=0))
    # print(q_tables.var(axis=0))
