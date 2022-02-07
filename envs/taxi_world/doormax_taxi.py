import numpy as np

from envs.taxi_world.taxi_world_env import TaxiWorldEnv, ACTION, Taxi, Passenger
from envs.taxi_world.doormax_helpers import condition_matches, boolean_arr_to_string, commute_condition_strings
from envs.taxi_world.effects import Effect, Increment, SetToNumber, SetToBoolean, NoChange, get_effects
from envs.taxi_world.prediction import Prediction

# TODO: Objects should have an attribute method, then we union those
ALL_ATTRIBUTES = ["taxi.x", "taxi.y", "passenger.in_taxi"]
ALL_EFFECTS = [Increment, SetToNumber, SetToBoolean, NoChange]


class DoormaxTaxi:
    """Implements deterministic object oriented r max RL for deterministic taxi world"""

    def __init__(self, env: TaxiWorldEnv):
        # A reference to the environment
        self.env = env

        # Collection of failure conditions for actions (what conditions cause no change for given action)
        self.failure_conditions = dict()

        # List of effect predictions for each attribute and action
        self.predictions = dict()

        self.init_data_structures()

    def init_data_structures(self):
        # For every action
        for action in list(ACTION):
            # Initialize all failure conditions for that action to empty
            self.failure_conditions[action] = []

            # Setup empty prediction list
            self.predictions[action] = dict()

            # For all attributes and effects, set predictions for the combination to null
            for attribute in ALL_ATTRIBUTES:

                # Setup empty prediction list
                self.predictions[action][attribute] = dict()

                for effect_type in ALL_EFFECTS:
                    # List of predictions starts empty
                    self.predictions[action][attribute][effect_type] = []

    def select_action(self):
        # For now, pick a random action from the list
        # Just return south, for testing
        return ACTION.SOUTH
        # return np.random.choice(list(ACTION))

    def predict_transition(self, state, action):
        """
        Predicts the next state from current state and action,
        or returns unknown (max reward) if it doesn't know
        """

        condition = self.conditions(state)

        for failure_condition in self.failure_conditions[action]:
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
        passenger = Passenger(passenger_in_taxi)

        return boolean_arr_to_string([
            self.env.touching_wall(taxi, ACTION.NORTH),
            self.env.touching_wall(taxi, ACTION.EAST),
            self.env.touching_wall(taxi, ACTION.SOUTH),
            self.env.touching_wall(taxi, ACTION.WEST),
            self.env.on_destination(taxi, destination),
            self.env.on_pickup(taxi, pickup),
            passenger.in_taxi
        ])

    def find_matching_prediction(self, current_predictions, effect):
        """Searches for a matching effect type in a list of predictions"""
        for p in current_predictions:
            if p.effect == effect:
                return p

        return None

    def add_experience(self, state, action, next_state, k):
        """Records experience of state action transition"""

        # Current state condition
        condition_s = self.conditions(state)
        print("Condition: {}".format(condition_s))

        if state == next_state:
            print("Failure condition")
            # If the states are the same, this is a failure condition for the action (nothing changed)

            # TODO: remove all matches conditions (to prevent duplicates? why?)
            # Print thrice, to check for change
            print(self.failure_conditions[action])
            self.failure_conditions[action] = [c for c in self.failure_conditions[action] if not condition_matches(condition_s, c)]
            print(self.failure_conditions[action])
            self.failure_conditions[action].append(condition_s)
            print(self.failure_conditions[action])
        else:
            print("Not failure condition")
            # Look through all effects to all attributes
            for attribute in ALL_ATTRIBUTES:
                for effect in get_effects(state, next_state, attribute):
                    print()
                    print(attribute, effect)

                    # Look through predictions for current action, attribute, and e type
                    # to find a matching effect
                    current_predictions = self.predictions[action][attribute][effect.type()]
                    print("Current predictions: {}".format(current_predictions))

                    matched_prediction = self.find_matching_prediction(current_predictions, effect)

                    # Check if we already have an effect that matches for this action and attribute
                    if matched_prediction is not None:
                        # We already have a prediction for what will happen to this attribute when
                        # this action is taken. Lets update it to make the condition more accurate
                        print("Found matching prediction: {}".format(matched_prediction))

                        matched_prediction.model = commute_condition_strings(matched_prediction.model, condition_s)
                        print("Updated prediction: {}".format(matched_prediction))
                    else:
                        # A new effect has been observed.
                        # If the condition does not overlap and existing condition, add the new prediction (TODO: why?)

                        print("Did not find matching prediction")
                        models = [p.model for p in current_predictions]
                        print("Models: {}".format(models))

                        # Search for overlapping conditions
                        overlap = False
                        for c in models:
                            if condition_matches(condition_s, c) or condition_matches(c, condition_s):
                                print("Found overlap: {}, {}".format(condition_s, c))
                                overlap = True
                                break

                        if overlap:
                            # TODO: remove pred from P
                            pass
                        else:
                            # Now we can add the new prediction to the list
                            print("No overlap")

                            current_predictions.append(Prediction(condition_s, effect))

                            # Make sure it got added
                            print(self.predictions[action][attribute][effect.type()])

                            # Check if there are more than k predictions for this action/attribute/type
                            # If there are, that's not the real effect type, so remove it from the running
                            # TODO: figure this out. Is k = 1? Or is it a bigger fudge factor?
                            # if len(self.predictions[action][attribute][effect.type()]) > k:
                            #     print("Too many effects, removing")
                            #     self.predictions[action][attribute][effect.type()] = None


# Create the env
env = TaxiWorldEnv()
doormax = DoormaxTaxi(env)

# How many iterations to iterate for, and iteration counter
NUM_ITERATIONS = 2
iterations = 0

# Main learning loop
while iterations < NUM_ITERATIONS:
    # Step 1: Observer current state s
    state = env.get_state()

    # Step 2: Choose action a according to exploration policy, based on prediction of T(s' | s, a)
    # returned by predictTransition(s, a)
    action = doormax.select_action()
    # print(action)

    # Step env
    # env.draw_taxi(delay=1)
    reward, done = env.step(action)

    # Step 3: observe new state s'
    new_state = env.get_state()
    print(action, reward, done, state, new_state)

    # Step 4: Update model with addExperience(s, a, s', k)
    # Because this world is discrete, there is only one effect per action, attribute, and type
    # (Can never be (Incerement(1), AND Increment(2)) for example, so k = 1
    doormax.add_experience(state, action, new_state, k=1)

    # Newline
    print()
    print()
    print()

    iterations += 1
