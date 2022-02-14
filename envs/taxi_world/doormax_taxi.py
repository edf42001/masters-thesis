import numpy as np

from envs.taxi_world.taxi_world_env import TaxiWorldEnv, ACTION, Taxi, Passenger
from envs.taxi_world.doormax_helpers import condition_matches, boolean_arr_to_string, commute_condition_strings
from envs.taxi_world.effects import Effect, Increment, SetToNumber, SetToBoolean, NoChange, get_effects, apply_effect
from envs.taxi_world.prediction import Prediction
from envs.taxi_world.value_iteration import solve_mdp_value_iteration

# TODO: Objects should have an attribute method, then we union those
ALL_ATTRIBUTES = ["taxi.x", "taxi.y", "passenger.in_taxi"]
ALL_EFFECTS = [Increment, SetToNumber, SetToBoolean, NoChange]

# TODO: FOR TESTING ONLY
np.random.seed(1)

import cProfile
import pstats
import pickle


class DoormaxTaxi:
    """Implements deterministic object oriented r max RL for deterministic taxi world"""

    def __init__(self, env: TaxiWorldEnv):
        # A reference to the environment
        self.env = env

        # Collection of failure conditions for actions (what conditions cause no change for given action)
        self.failure_conditions = dict()

        # List of effect predictions for each attribute and action
        self.predictions = dict()

        # Rewards for each state, action, used when solving the mdp with value iteration
        self.rewards = np.zeros((env.num_states(), env.num_actions()))

        # Sets up the above data structures
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

        # return \
        #     ("1" if self.env.touching_wall(taxi, ACTION.NORTH) else "0") + \
        #     ("1" if self.env.touching_wall(taxi, ACTION.EAST) else "0") + \
        #     ("1" if self.env.touching_wall(taxi, ACTION.SOUTH) else "0") + \
        #     ("1" if self.env.touching_wall(taxi, ACTION.WEST) else "0") + \
        #     ("1" if self.env.on_destination(taxi, destination) else "0") + \
        #     ("1" if self.env.on_pickup(taxi, pickup) else "0") + \
        #     ("1" if passenger.in_taxi else "0")

    def find_matching_prediction(self, current_predictions, effect):
        """Searches for a matching effect type in a list of predictions"""
        for p in current_predictions:
            if p.effect == effect:
                return p

        return None

    def check_conditions_overlap(self, current_predictions, matched_pred):
        """
        Check if there exists some condition (but don't compare matched prediction to itself!)
        where the conditions overlap. This is a contradiction? For some reason?
        """
        for pred in current_predictions:
            if pred != matched_pred and condition_matches(matched_pred.model, pred.model):
                print("Found overlapping condition: {}, {}".format(matched_pred, pred))
                return True

        return False

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
            # print(self.failure_conditions[action])
            self.failure_conditions[action] = [c for c in self.failure_conditions[action] if not condition_matches(condition_s, c)]
            # print(self.failure_conditions[action])
            self.failure_conditions[action].append(condition_s)
            print(self.failure_conditions[action])
        else:
            print("Not failure condition")
            # Look through all effects to all attributes
            for attribute in ALL_ATTRIBUTES:
                for effect in get_effects(state, next_state, attribute):
                    # print()
                    # print(attribute, effect)

                    # Look through predictions for current action, attribute, and e type
                    # to find a matching effect
                    current_predictions = self.predictions[action][attribute][effect.type()]

                    # If ever set to None, this means this is the wrong effect type for this pair
                    if current_predictions is None:
                        # print("This condition is known to be wrong")
                        continue

                    # print("Current predictions: {}".format(current_predictions))

                    matched_prediction = self.find_matching_prediction(current_predictions, effect)

                    # Check if we already have an effect that matches for this action and attribute
                    if matched_prediction is not None:
                        # We already have a prediction for what will happen to this attribute when
                        # this action is taken. Lets update it to make the condition more accurate
                        # print("Found matching prediction")

                        matched_prediction.model = commute_condition_strings(matched_prediction.model, condition_s)
                        # print("Updated prediction: {}".format(matched_prediction))

                        # Check for overlapping conditions, this would be a contradiction and we remove this Type
                        if self.check_conditions_overlap(current_predictions, matched_prediction):
                            self.predictions[action][attribute][effect.type()] = None
                        else:
                            pass
                            # print("No overlap")

                    else:
                        # A new effect has been observed.
                        # If the condition does not overlap and existing condition, add the new prediction (TODO: why?)

                        # print("Did not find matching prediction")
                        models = [p.model for p in current_predictions]
                        # print("Models: {}".format(models))

                        # Search for overlapping conditions
                        # TODO: why does this one compare cond/c, and c/cond, while the other only does one
                        overlap = False
                        for c in models:
                            if condition_matches(condition_s, c) or condition_matches(c, condition_s):
                                # print("Found overlap, removing: {}, {}".format(condition_s, c))
                                overlap = True
                                break

                        if overlap:
                            self.predictions[action][attribute][effect.type()] = None
                        else:
                            # Now we can add the new prediction to the list
                            # print("No overlap")

                            current_predictions.append(Prediction(condition_s, effect))

                            # Make sure it got added
                            print(self.predictions[action][attribute][effect.type()])

                            # Check if there are more than k predictions for this action/attribute/type
                            # If there are, that's not the real effect type, so remove it from the running
                            # TODO: figure this out. Is k = 1? Or is it a bigger fudge factor?
                            if len(self.predictions[action][attribute][effect.type()]) > k:
                                # print("Too many effects, removing")
                                self.predictions[action][attribute][effect.type()] = None

    def predict_transition(self, state, action):
        """
        Predicts the next state from current state and action,
        or returns unknown (max reward) if it doesn't know
        """

        condition_s = self.conditions(state)
        # print("Action: {}".format(action))
        # print("Condition: {}".format(condition_s))
        # print("State: {}".format(state))

        for failure_condition in self.failure_conditions[action]:
            if condition_matches(failure_condition, condition_s):
                # The current condition is a failure condition. No change
                return state

        # Otherwise, check all effects and attributes
        # TODO: Is this wrong? Should applied effects be higher up??
        for attribute in ALL_ATTRIBUTES:
            applied_effects = []

            # print(attribute)
            for effect_type in ALL_EFFECTS:
                # print(effect_type)
                # If we have predictions that match the state we are currently in,
                # then we know those effects will happen
                current_predictions = self.predictions[action][attribute][effect_type]
                # print("Current_predictions: {}".format(current_predictions))

                # If none, not a real effect, continue
                if current_predictions is None:
                    continue

                for pred in self.predictions[action][attribute][effect_type]:
                    if condition_matches(pred.model, condition_s):
                        # print("Matching condition: {}, {}".format(pred.model, condition_s))
                        applied_effects.append(pred.effect)

            # print(applied_effects)
            # If e is empty or there are incompatible effects, we don't know what will happen, return max_reward
            if len(applied_effects) == 0 or self.incompatable_effects(applied_effects):
                # print("No effects found or incompatible")
                return None  # None represents max reward
            else:
                # print("Effects for this state: {}".format(applied_effects))
                for effect in applied_effects:
                    # TODO NEED ATTRIBUTE? (Only one attribute changes at a time so it doesn't matter??)
                    # print(state)
                    state = apply_effect(state, attribute, effect)
                    # print(state)

        # When done, if none of the failure conditions were triggered, return the resulting stat
        return state

    def predict_next_states(self, state):
        cond_s = self.conditions(state)
        # print("PREDICTING ACTIONS NOW: {}, {}".format(state, cond_s))

        next_states = dict()
        for action in list(ACTION):
            # print(action)
            next_states[action] = self.predict_transition(state, action)
            # print()

        # print(next_states)
        return next_states

    def incompatable_effects(self, effects):
        """
        Returns if any incompatable affects are in this list
        Technically, incompatable effects are those which produce different values
        for the same initial value attribute, but I will just check for equality,
        which is the same for this small set of effects
        """

        for i in range(len(effects)):
            for j in range(len(effects)):
                if i != j and effects[i] != effects[j]:
                    return True

        return False

    def select_action(self, state, discount_rate):
        # For now, pick a random action from the list
        # Just return south, for testing
        # return ACTION.SOUTH
        # Pick the best action from the solved mdp:
        values = solve_mdp_value_iteration(state, self, discount_rate)
        next_states = doormax.predict_next_states(state)
        best_value = -100
        best_action = None

        print("Select action:")
        # print("Values from value iteration: {}".format(values))
        print("Next states: {}".format(next_states))

        best_values = []
        for action in list(ACTION):
            next_state = next_states[action]
            if next_state is None:
                value = 15
            else:
                value = values[doormax.env.state_hash(next_state)]

            if value > best_value:
                best_value = value
                best_action = action

            best_values.append(value)

        print("Values of actions: {}".format(best_values))
        return best_action
        # return np.random.choice(list(ACTION))

    def print_predictions(self, predictions):
        """Prints predictions in an easy to read format"""
        for action in list(ACTION):
            print(action)
            for attribute in ALL_ATTRIBUTES:
                non_empty_effects = []
                for effect in ALL_EFFECTS:
                    effects = predictions[action][attribute][effect]
                    if effects is not None and len(effects) != 0:
                        non_empty_effects.append(effects)

                # Only print attribute and effects if there is something to see
                if len(non_empty_effects) != 0:
                    print("{}: {}".format(attribute, non_empty_effects))
            # print()


# Profiling
# profiler = cProfile.Profile()
# profiler.enable()

if __name__ == "__main__":
    # Create the env
    env = TaxiWorldEnv()
    doormax = DoormaxTaxi(env)

    # # Load from pickle for testing
    # with open("doormax.pkl", "rb") as f:
    #     doormax = pickle.load(f)
    #     env = doormax.env

    # How many iterations to iterate for, and iteration counter
    NUM_ITERATIONS = 10000
    iterations = 0

    # Main learning loop
    while iterations < NUM_ITERATIONS:
        # Step 1: Observer current state s
        state = env.get_state()

        # Step 2: Choose action a according to exploration policy, based on prediction of T(s' | s, a)
        # returned by predictTransition(s, a)
        action = doormax.select_action(state, discount_rate=0.8)

        # Figure out which action is best to pick
        next_states = doormax.predict_next_states(state)

        # Step env
        env.draw_taxi(delay=50)
        reward, done = env.step(action)

        # Step 3: observe new state s'
        new_state = env.get_state()
        print(action, reward, done, state, new_state)
        # TODO. If done, don't add experience? (or at least, the increments get messed up because the taxi moves randomly)
        # TODO: They don't get messed up if k is set to 1 but that's probably just a hack.

        # Step 4: Update model with addExperience(s, a, s', k)
        # Because this world is discrete, there is only one effect per action, attribute, and type
        # (Can never be (Incerement(1), AND Increment(2)) for example, so k = 1
        doormax.add_experience(state, action, new_state, k=1)

        # Record reward for later use
        doormax.rewards[doormax.env.state_hash(state), action.value] = reward

        # Newline
        print(iterations)
        print()
        print()
        print()

        # if iterations > 85:
        #     with open("doormax.pkl", 'wb') as f:
        #         pickle.dump(doormax, f)
        # a = input("paused: ")

        iterations += 1

    print("RESULTS:")
    doormax.print_predictions(doormax.predictions)

    # with open("doormax.pkl", 'wb') as f:
    #     pickle.dump(doormax, f)


    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.dump_stats('doormax_stats.prof')
