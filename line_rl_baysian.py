from line_env import LineEnv

import numpy as np
import random
import matplotlib.pyplot as plt

env = LineEnv()

NUM_STATES = 5  # States in env
NUM_ACTIONS = 2  # Actions per state (uniform in this case)

q_table = np.zeros(NUM_STATES*NUM_ACTIONS)  # 5 states, 2 actions per state = 10
iterations = 0  # How many steps the agent has taken
epsilon = 0.2  # Epsilon exploration
epsilon_decay = 2E-6  # Decay epsilon per iteration
learning_rate = 0.20  # For bellman update
discount_rate = 0.95  # Discount future actions

TRAINING_EPOCHS = int(5E4)  # How long to train for


# Starting state of env
state = 0

rewards = np.zeros(TRAINING_EPOCHS)


while iterations < TRAINING_EPOCHS:

    # Choose action with epsilon random exploration
    if random.random() < epsilon:
        action = random.randint(0, 1)
    else:
        # Choose best action from available in q table
        # Because there's only two actions I'll just do this manually
        # Actions are hashed by state+5*action
        action = 0 if q_table[state] > q_table[state + NUM_STATES] else 1

    # Step environment
    next_state, reward, _, _ = env.step(action)

    # Q table bellman update
    curr_index = state + NUM_STATES * action
    next_best_q = max(q_table[next_state], q_table[next_state + NUM_STATES])

    q_table[curr_index] = (1 - learning_rate) * q_table[curr_index] + learning_rate * (reward + discount_rate * next_best_q)

    # Update current state
    state = next_state

    # Store reward
    rewards[iterations] = reward

    iterations += 1
    epsilon = max(0, epsilon - epsilon_decay)

print(q_table)


# Moving average using convolution
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


rewards_averaged = moving_average(rewards, 100)
plt.plot(rewards_averaged)
plt.plot(rewards)
plt.show()
