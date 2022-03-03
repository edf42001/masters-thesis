from environment.hierarchy.hierarchy import Hierarchy


class TaxiHierarchy(Hierarchy):  # TODO: Fix undefined reference to hierarchy

    # Primitive actions
    A_NORTH = 0
    A_EAST = 1
    A_SOUTH = 2
    A_WEST = 3
    A_PICKUP = 4
    A_DROPOFF = 5

    # Non-primitive actions (navigate to 4 locations, get / put passenger, root of all tasks)
    A_NAVIGATE_0 = 6
    A_NAVIGATE_1 = 7
    A_NAVIGATE_2 = 8
    A_NAVIGATE_3 = 9
    A_GET = 10
    A_PUT = 11
    A_ROOT = 12

    def __init__(self, env):  # TODO: Say type is environment here
        self.env = env  # Store reference to env

        # List of actions available to each subtask
        self.children = {
            # Primitives
            self.A_NORTH: [],
            self.A_EAST: [],
            self.A_SOUTH: [],
            self.A_WEST: [],
            self.A_PICKUP: [],
            self.A_DROPOFF: [],

            # Non-primitives
            self.A_NAVIGATE_0: [self.A_NORTH, self.A_EAST, self.A_SOUTH, self.A_WEST],
            self.A_NAVIGATE_1: [self.A_NORTH, self.A_EAST, self.A_SOUTH, self.A_WEST],
            self.A_NAVIGATE_2: [self.A_NORTH, self.A_EAST, self.A_SOUTH, self.A_WEST],
            self.A_NAVIGATE_3: [self.A_NORTH, self.A_EAST, self.A_SOUTH, self.A_WEST],
            self.A_GET: [self.A_PICKUP, self.A_NAVIGATE_0, self.A_NAVIGATE_1, self.A_NAVIGATE_2, self.A_NAVIGATE_3],
            self.A_PUT: [self.A_DROPOFF, self.A_NAVIGATE_0, self.A_NAVIGATE_1, self.A_NAVIGATE_2, self.A_NAVIGATE_3],
            self.A_ROOT: [self.A_GET, self.A_PUT]
        }

        super().__init__()

    def is_terminated(self, action: int, state: int, done: bool = False) -> bool:
        x, y, passenger, dest = self.env.get_factored_state(state)
        pos = (x, y)

        # Determine when an action, primitive or not, has finished
        # primitive actions finish immediately. I assume done is end of episode? Will need to check
        # Navigates are done when the taxi ends up on the location
        # Don't yet understand about the put or get
        if done or self.is_primitive(action):
            return True
        elif action == self.A_ROOT:
            return done
        elif action == self.A_PUT:
            return passenger != self.env.NUM_LOCATIONS
        elif action == self.A_GET:
            return passenger >= self.env.NUM_LOCATIONS
        elif action == self.A_NAVIGATE_0:
            return pos == self.env.locations[0]
        elif action == self.A_NAVIGATE_1:
            return pos == self.env.locations[1]
        elif action == self.A_NAVIGATE_2:
            return pos == self.env.locations[2]
        elif action == self.A_NAVIGATE_3:
            return pos == self.env.locations[3]
