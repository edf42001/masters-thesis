import numpy as np

from envs.taxi_world.taxi_world_env import TaxiWorldEnv, ACTION


# METHODS
def select_action():
    # For now, pick a random action from the list
    return np.random.choice(list(ACTION))


# BEGIN CODE
# Create the env
env = TaxiWorldEnv()

# For every action, a list of failure conditions for which that action does nothing
failure_conditions = dict()

# How many iterations to iterate for, and iteration counter
NUM_ITERATIONS = 3000
iterations = 0

# Set up data structures
for action in ACTION:
    # Initialize all failure conditions to empty
    failure_conditions[action] = []

    # TODO: A lot more setup in here


# Main learning loop
while iterations < NUM_ITERATIONS:
    # Step 1: Observer current state s
    state = env.get_state()

    # Step 2: Choose action a according to exploration policy, based on prediction of T(s' | s, a)
    # returned by predictTransition(s, a)
    action = select_action()

    # Step env
    env.draw_taxi(delay=1)
    reward, done = env.step(action)

    # Step 3: observe new state s'
    new_state = env.get_state()

    # Step 4: Update model with addExperience(s, a, s', k)


    iterations += 1
