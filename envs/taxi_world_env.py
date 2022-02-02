import random
from enum import Enum
import numpy as np


# Enum for actions
class ACTION(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    PICKUP = 4
    DROPOFF = 5


class TaxiWorldEnv(object):
    """
    Simulation of the (deterministic) taxi world environment. Taxi world is 5x5 and there is a taxi.
    The taxi picks up people from one of 4 location RGBY, and drops them off at one of the locations,
    the target is chosen randomly. There are some walls in the way that the taxi stops if it hits them.
    The taxi gets +10 for dropping off the passenger correctly and -1 for every other step
    """

    def __init__(self):
        # Taxi's position
        self.taxi_x = 0
        self.taxi_y = 0

        # Locations of taxi destinations
        self.stops = {"R": [0, 4], "G": [4, 4], "B": [3, 0], "Y": [0, 0]}

        # Current pickup and stoplocation
        self.current_pickup = "R"
        self.current_dropoff = "G"

        # Has the taxi picked up the passenger
        self.passenger_in_taxi = False

        # Location of walls
        self.walls_vertical = np.zeros((6, 6), dtype=bool)
        self.horizontal = np.zeros((6, 6), dtype=bool)

        # Randomize starting values
        self.reset_passenger_pickup_dropoff()
        self.reset_taxi()

    def setup_walls(self):
        # Wall count starts at 0, 0, and moves horizontally and up
        vertical = [0, 1, 3, 5, 6, 7, 9, 11, 12, 17, 18, 20, 23, 24, 26, 29]
        horizontal = [0, 1, 2, 3, 4, 25, 26, 27, 28, 29]

    def step(self, action):
        # Negative reward by default
        reward = -1

        # Epoch has finished when taxi drops off passenger
        done = False

        # State changes based on action (deterministically)
        # Using .value allows us to use numbers. TODO: use enums?
        # TODO: Add effect of walls
        if action == ACTION.NORTH.value:
            self.taxi_y += 1

        elif action == ACTION.EAST.value:
            self.taxi_x += 1

        elif action == ACTION.SOUTH.value:
            self.taxi_y += -1

        elif action == ACTION.WEST.value:
            self.taxi_x += -1

        elif action == ACTION.PICKUP.value:
            square = self.stops[self.current_pickup]

            # If on the right spot and the passenger is there, pick them up
            if square[0] == self.taxi_x and square[1] == self.taxi_y and not self.passenger_in_taxi:
                self.passenger_in_taxi = True

        elif action == ACTION.DROPOFF.value:
            square = self.stops[self.current_dropoff]

            # If on the right spot and the passenger is in taxi, drop them off, +10 reward
            if square[0] == self.taxi_x and square[1] == self.taxi_y and self.passenger_in_taxi:
                # Episode restarts, randomize values
                self.passenger_in_taxi = False
                self.reset_passenger_pickup_dropoff()
                self.reset_taxi()

                reward = 10
                done = True
        else:
            print("BAD ACTION: {}".format(action))

        return reward, done

    # TODO: The episode ends when taxis drops off passenger, so this should reset the taxi position as well
    def reset_passenger_pickup_dropoff(self):
        # Pick random pickup location
        choices = list(self.stops.keys())
        self.current_pickup = np.random.choice(choices)

        # Pick random dropoff, and make sure it is not the pickup
        self.current_dropoff = np.random.choice(choices)
        while self.current_pickup == self.current_dropoff:
            self.current_dropoff = np.random.choice(choices)

    def reset_taxi(self):
        self.taxi_x = np.random.randint(0, 4)
        self.taxi_y = np.random.randint(0, 4)

    def get_state(self):
        return self.state

    def num_states(self):
        return 9

    def num_actions(self):
        return 2

    def num_rewards(self):
        return 3


if __name__ == "__main__":
    env = TaxiWorldEnv()

    NUM_STEPS = 20
    iterations = 0
    while iterations < NUM_STEPS:

        action = int(input("Enter action: "))
        reward, done = env.step(action)

        print("Reward: {}".format(reward))
        print("pickup: {}, dropoff: {}".format(env.current_pickup, env.current_dropoff))
        print("x: {}, y: {}".format(env.taxi_x, env.taxi_y))
        print("passneger_in_taxi: {}".format(env.passenger_in_taxi))


    iterations += 1
