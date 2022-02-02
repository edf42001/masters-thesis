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
        self.walls_horizontal = np.zeros((6, 6), dtype=bool)
        self.setup_walls()

        # Randomize starting values
        self.reset_passenger_pickup_dropoff()
        self.reset_taxi()

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
        # Using  allows us to use numbers. TODO: use enums?
        if action in [ACTION.NORTH, ACTION.EAST, ACTION.SOUTH, ACTION.WEST] and \
           self.touching_wall(action):
            # If the action is trying to move but there is a wall in the way, literally do nothing
            pass

        elif action == ACTION.NORTH:
            self.taxi_y += 1

        elif action == ACTION.EAST:
            self.taxi_x += 1

        elif action == ACTION.SOUTH:
            self.taxi_y += -1

        elif action == ACTION.WEST:
            self.taxi_x += -1

        elif action == ACTION.PICKUP:
            square = self.stops[self.current_pickup]

            # If on the right spot and the passenger is there, pick them up
            if square[0] == self.taxi_x and square[1] == self.taxi_y and not self.passenger_in_taxi:
                self.passenger_in_taxi = True

        elif action == ACTION.DROPOFF:
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

    def draw_taxi(self):
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
        cv2.circle(img, (int((self.taxi_x + 0.5) * GRID_SIZE), int((HEIGHT - (self.taxi_y + 0.5)) * GRID_SIZE)), int(GRID_SIZE * 0.3),
                   thickness=-1, color=[0, 0, 0])

        # Draw passenger
        if self.passenger_in_taxi:
            pass_x = self.taxi_x
            pass_y = self.taxi_y
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
        cv2.waitKey(2)

    def touching_wall(self, direction):
        """Checks if the taxi is touching a wall in a certain direction"""
        if direction == ACTION.NORTH:
            return self.walls_horizontal[self.taxi_x, self.taxi_y + 1]
        elif direction == ACTION.EAST:
            return self.walls_vertical[self.taxi_x+1, self.taxi_y]
        elif direction == ACTION.SOUTH:
            return self.walls_horizontal[self.taxi_x, self.taxi_y]
        elif direction == ACTION.WEST:
            return self.walls_vertical[self.taxi_x, self.taxi_y]
        else:
            print("Bad direction: {}".format(direction))
            return False

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

    # TODO: Way to automate this?
    action_map = [ACTION.NORTH, ACTION.EAST, ACTION.SOUTH, ACTION.WEST, ACTION.PICKUP, ACTION.DROPOFF]

    NUM_STEPS = 20
    iterations = 0
    while iterations < NUM_STEPS:

        env.draw_taxi()
        action = action_map[int(input("Enter action: "))]
        reward, done = env.step(action)

        print("Reward: {}".format(reward))
        print("pickup: {}, dropoff: {}".format(env.current_pickup, env.current_dropoff))
        print("x: {}, y: {}".format(env.taxi_x, env.taxi_y))
        print("passneger_in_taxi: {}".format(env.passenger_in_taxi))

    iterations += 1
