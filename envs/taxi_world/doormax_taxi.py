import numpy as np

from envs.taxi_world.taxi_world_env import TaxiWorldEnv, ACTION, Taxi
from envs.taxi_world.doormax_helpers import condition_matches, boolean_arr_to_string


class DoormaxTaxi:
    """Implements deterministic object oriented r max RL for deterministic taxi world"""

    def __init__(self, env: TaxiWorldEnv):
        # A reference to the environment
        self.env = env

        # Collection of failure conditions for actions (what conditions cause no change for given action)
        self.failure_conditions = dict()

    def init_failure_conditions(self):
        for action in ACTION:
            # Initialize all failure conditions for that action to empty
            self.failure_conditions[action] = []

    def select_action(self):
        # For now, pick a random action from the list
        return np.random.choice(list(ACTION))

    def predict_transition(self, state, action):
        """
        Predicts the next state from current state and action,
        or returns unknown (max reward) if it doesn't know
        """

        condition = self.conditions(state)

        for failure_condition in failure_conditions[action]:
            if condition_matches(failure_condition, condition):
                # The current condition is a failure condition. No change
                return state

        # Otherwise, check all effect types?

    def conditions(self, state):
        """
        Returns the conditions that are true for a given state
        Terms consist of:
        touchN/S/E/W(taxi, wall), on(taxi, passenger), on(taxi, destination), passenger.in_taxi,
        and their negations.
        """
        x = state[0]
        y = state[1]
        pickup = ["R", "G", "B", "Y"][state[2]]  # Really need to rethink these (letters vs index) for stops
        destination = ["R", "G", "B", "Y"][state[3]]
        passenger_in_taxi = state[4]

        taxi = Taxi(x, y)

        return boolean_arr_to_string([
            self.env.touching_wall(taxi, ACTION.NORTH),
            self.env.touching_wall(taxi, ACTION.EAST),
            self.env.touching_wall(taxi, ACTION.SOUTH),
            self.env.touching_wall(taxi, ACTION.WEST),
            self.env.on_destination(taxi, destination),
            self.env.on_pickup(taxi, pickup),
            passenger_in_taxi
        ])

    def add_experience(self, state, action, next_state, k):
        """Records experience of state action transition"""
        if state == next_state:
            # If the states are the same, this is a failure condition for the action (nothing changed)

            condition = self.conditions(state)

            # TODO: remove all matches conditions (to prevent duplicates? why?)
            failure_conditions[action] = [c for c in failure_conditions[action] if not condition_matches(condition, c)]
            failure_conditions[action].append(condition)


# BEGIN CODE
# Create the env
env = TaxiWorldEnv()

# For every action, a list of failure conditions for which that action does nothing
failure_conditions = dict()

# How many iterations to iterate for, and iteration counter
NUM_ITERATIONS = 3000
iterations = 0

# Set up data structures
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
