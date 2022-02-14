import random
from enum import Enum
import numpy as np
import cv2


# Enum for actions
class ACTION(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    PICKUP = 4
    DROPOFF = 5


class Taxi:
    """Class to store info about the taxi object"""
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class Passenger:
    def __init__(self, in_taxi=False):
        self.in_taxi = in_taxi


class TaxiWorldEnv(object):
    """
    Simulation of the (deterministic) taxi world environment. Taxi world is 5x5 and there is a taxi.
    The taxi picks up people from one of 4 location RGBY, and drops them off at one of the locations,
    the target is chosen randomly. There are some walls in the way that the taxi stops if it hits them.
    The taxi gets +10 for dropping off the passenger correctly and -1 for every other step
    """

    def __init__(self):
        # Width and height in grid cells
        self.WIDTH = 5
        self.HEIGHT = 5

        # Taxi's object
        self.taxi = Taxi()

        # Passenger object
        self.passenger = Passenger()

        # Locations of taxi destinations
        self.stops = {"R": [0, 4], "G": [4, 4], "B": [3, 0], "Y": [0, 0]}

        # Current pickup and stoplocation
        self.current_pickup = "R"
        self.current_dropoff = "G"

        # Location of walls
        self.walls_vertical = np.zeros((6, 6), dtype=bool)
        self.walls_horizontal = np.zeros((6, 6), dtype=bool)
        self.setup_walls()

        # Randomize starting values
        self.reset_passenger_pickup_dropoff()
        self.reset_taxi()

        # TODO: FOR TESTING PURPOSES ONLY
        self.taxi.x = 0
        self.taxi.y = 2
        self.current_pickup = "Y"
        self.current_dropoff = "G"

    def setup_walls(self):
        # Walls are indexed by their bottom left corner. A horizontal wall and veritcal wall extend right and up from
        # that point if they are marked as existing. This means there are (Width+1)*(Height+1) points, but the rightmost
        # and upmost points aren't used for horizontal and vertical respectively.
        vertical = [0, 1, 3, 5, 6, 7, 9, 11, 12, 17, 18, 20, 23, 24, 26, 29]
        horizontal = [0, 1, 2, 3, 4, 30, 31, 32, 33, 34]

        for wall in vertical:
            x = wall % (self.WIDTH + 1)
            y = int(wall / (self.WIDTH + 1))
            self.walls_vertical[x, y] = True

        for wall in horizontal:
            x = wall % (self.WIDTH + 1)
            y = int(wall / (self.WIDTH + 1))
            self.walls_horizontal[x, y] = True

    def step(self, action):
        # Negative reward by default
        reward = -1

        # Epoch has finished when taxi drops off passenger
        done = False

        # State changes based on action (deterministically)
        if action in [ACTION.NORTH, ACTION.EAST, ACTION.SOUTH, ACTION.WEST] and \
           self.touching_wall(self.taxi, action):
            # If the action is trying to move but there is a wall in the way, literally do nothing
            pass

        elif action == ACTION.NORTH:
            self.taxi.y += 1

        elif action == ACTION.EAST:
            self.taxi.x += 1

        elif action == ACTION.SOUTH:
            self.taxi.y += -1

        elif action == ACTION.WEST:
            self.taxi.x += -1

        elif action == ACTION.PICKUP:
            # If on the right spot and the passenger is there, pick them up
            if self.on_pickup(self.taxi, self.current_pickup) and not self.passenger_in_taxi():
                self.passenger.in_taxi = True

        elif action == ACTION.DROPOFF:
            # If on the right spot and the passenger is in taxi, drop them off, +10 reward
            if self.on_destination(self.taxi, self.current_dropoff) and self.passenger_in_taxi():
                # Episode restarts, randomize values
                self.passenger.in_taxi = False
                self.reset_passenger_pickup_dropoff()
                self.reset_taxi()

                reward = 10
                done = True
        else:
            print("BAD ACTION: {}".format(action))

        return reward, done

    def touching_wall(self, taxi, direction):
        """Checks if the taxi is touching a wall in a certain direction"""
        if direction == ACTION.NORTH:
            return self.walls_horizontal[taxi.x, taxi.y + 1]
        elif direction == ACTION.EAST:
            return self.walls_vertical[taxi.x+1, taxi.y]
        elif direction == ACTION.SOUTH:
            return self.walls_horizontal[taxi.x, taxi.y]
        elif direction == ACTION.WEST:
            return self.walls_vertical[taxi.x, taxi.y]
        else:
            print("Bad direction: {}".format(direction))
            return False

    def on_destination(self, taxi, destination):
        """Check if the taxi is on the destination"""
        square = self.stops[destination]
        return square[0] == taxi.x and square[1] == taxi.y

    def on_pickup(self, taxi, pickup):
        """Check if the taxi is on passenger pickup location"""
        # TODO: and passenger isn't in taxi?
        square = self.stops[pickup]
        return square[0] == taxi.x and square[1] == taxi.y

    def passenger_in_taxi(self):
        """Is the passenger in the tax"""
        return self.passenger.in_taxi

    def reset_passenger_pickup_dropoff(self):
        # Pick random pickup location
        choices = list(self.stops.keys())
        self.current_pickup = np.random.choice(choices)

        # Pick random dropoff, and make sure it is not the pickup
        self.current_dropoff = np.random.choice(choices)
        while self.current_pickup == self.current_dropoff:
            self.current_dropoff = np.random.choice(choices)

    def reset_taxi(self):
        self.taxi.x = np.random.randint(0, 4)
        self.taxi.y = np.random.randint(0, 4)

    def draw_taxi(self, delay=100):
        GRID_SIZE = 100
        WIDTH = self.WIDTH
        HEIGHT = self.HEIGHT

        # Blank white square
        img = 255 * np.ones((HEIGHT * GRID_SIZE, WIDTH * GRID_SIZE, 3))

        # Draw pickup and dropoff zones
        colors = {"R": [0, 0, 255], "G": [0, 255, 0], "B": [255, 0, 0], "Y": [0, 128, 128]}
        for color in colors.keys():
            location = self.stops[color]
            bottom_left_x = location[0] * GRID_SIZE
            bottom_left_y = (HEIGHT - location[1]) * GRID_SIZE
            cv2.rectangle(img, (bottom_left_x, bottom_left_y), (bottom_left_x + GRID_SIZE, bottom_left_y - GRID_SIZE),
                          thickness=-1, color=colors[color])

        # Mark goal with small circle
        goal_x = self.stops[self.current_dropoff][0]
        goal_y = self.stops[self.current_dropoff][1]
        cv2.circle(img, (int((goal_x + 0.5) * GRID_SIZE), int((HEIGHT - (goal_y + 0.5)) * GRID_SIZE)), int(GRID_SIZE * 0.05),
                   thickness=-1, color=[0, 0, 0])

        # Draw taxi
        cv2.circle(img, (int((self.taxi.x + 0.5) * GRID_SIZE), int((HEIGHT - (self.taxi.y + 0.5)) * GRID_SIZE)), int(GRID_SIZE * 0.3),
                   thickness=-1, color=[0, 0, 0])

        # Draw passenger
        if self.passenger_in_taxi():
            pass_x = self.taxi.x
            pass_y = self.taxi.y
        else:
            pass_x = self.stops[self.current_pickup][0]
            pass_y = self.stops[self.current_pickup][1]

        cv2.circle(img, (int((pass_x + 0.5) * GRID_SIZE), int((HEIGHT - (pass_y + 0.5)) * GRID_SIZE)), int(GRID_SIZE * 0.2),
                   thickness=-1, color=[0.5, 0.5, 0.5])

        # Draw horizontal and vertical walls
        for i in range((self.WIDTH + 1) * (self.HEIGHT + 1)):
            x = i % (self.WIDTH + 1)
            y = int(i / (self.WIDTH + 1))

            if self.walls_vertical[x, y]:
                cv2.line(img, (x * GRID_SIZE, (HEIGHT - y) * GRID_SIZE), (x * GRID_SIZE, (HEIGHT - (y+1)) * GRID_SIZE),
                         thickness=3, color=[0, 0, 0])

            if self.walls_horizontal[x, y]:
                cv2.line(img, (x * GRID_SIZE, (HEIGHT - y) * GRID_SIZE), ((x+1) * GRID_SIZE, (HEIGHT - y) * GRID_SIZE),
                         thickness=3, color=[0, 0, 0])

        cv2.imshow("Taxi World", img)
        cv2.waitKey(delay)

    def num_states(self):
        # One state for taxi.x, taxi.y, 1 for every passenger location (pickups +1 for in taxi), one for every dropoff
        return self.WIDTH * self.HEIGHT * (len(self.stops) + 1) * len(self.stops)

    def num_actions(self):
        return len(list(ACTION))

    def num_rewards(self):
        return 3

    def get_state(self):
        """State is a tuple of (taxi.x, taxi.y, pickup, dropoff, passenger.in_taxi"""

        stop_letter_to_index = {"R": 0, "G": 1, "B": 2, "Y": 3}

        passenger_loc_index = stop_letter_to_index[self.current_pickup]
        destination_loc_index = stop_letter_to_index[self.current_dropoff]

        return self.taxi.x, self.taxi.y, passenger_loc_index, destination_loc_index, self.passenger_in_taxi()

    def state_hash(self, state):
        """Converts a state (tuple) into unique hash (number) for indexing"""
        # 5 for taxi x, 5 for taxi y, 5 for current passenger, 4 for destination
        # Need to combine pickup location (index 2) with passenger in taxi (index 4) to make 5 states instead of 8

        # Set this to 4 if passenger is in taxi otherwise 0-3 for the pickup index
        passenger_loc_index = 4 if state[4] else state[2]

        return state[0] + \
               5 * state[1] + \
               25 * passenger_loc_index + \
               125 * state[3]

    def reverse_state_hash(self, state_hash, pickup):
        """Converts a state hash (number) into a state (tuple)"""
        x = state_hash % 5
        y = int(state_hash / 5) % 5
        passenger_loc = int(state_hash / 25) % 5
        destination_loc = int(state_hash / 125)

        # Hack to fix the fact that passenger in taxi passenger location are weird together, leading to bugs
        # where it is set to 0 but this messes with the conditions
        passenger_in_taxi = (passenger_loc == 4)
        pickup = passenger_loc if not passenger_in_taxi else pickup  # It doesn't matter, the taxi doesn't need to go to pickup if the passenger is in the taxi already (THIS IS A FALSE STATEMENT)

        return x, y, pickup, destination_loc, passenger_in_taxi


if __name__ == "__main__":
    env = TaxiWorldEnv()

    action_map = list(ACTION)

    NUM_STEPS = 20
    iterations = 0
    while iterations < NUM_STEPS:

        env.draw_taxi()
        action = action_map[int(input("Enter action: "))]
        reward, done = env.step(action)

        print("Reward: {}".format(reward))
        print("pickup: {}, dropoff: {}".format(env.current_pickup, env.current_dropoff))
        print("x: {}, y: {}".format(env.taxi.x, env.taxi.y))
        print("passenger_in_taxi: {}".format(env.passenger_in_taxi()))

    iterations += 1
