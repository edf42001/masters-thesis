from enum import Enum


class SymbolicObject:
    """Represents a generic object in a symbolic world"""
    def __init__(self, name=""):
        self.name = name

    def __repr__(self):
        # Print the object type plus our unique id
        return f"{type(self).__name__}({self.name})"


class Door(SymbolicObject):
    def __init__(self, name="", x=0, open=False):
        super().__init__(name)

        self.x = x
        self.open = open


class Switch(SymbolicObject):
    def __init__(self, name="", x=0):
        super().__init__(name)

        self.x = x


class Goal(SymbolicObject):
    def __init__(self, name="", x=0):
        super().__init__(name)

        self.x = x


class Gem(SymbolicObject):
    def __init__(self, name, location, state):
        super().__init__(name)

        self.location = location
        self.state = state


class Wall(SymbolicObject):
    def __init__(self, name, locations):
        super().__init__(name)

        # One wall object handles all the walls in the world
        self.locations = locations


class Lock(SymbolicObject):
    def __init__(self, name, location, open):
        super().__init__(name)

        # One wall object handles all the walls in the world
        self.location = location
        self.open = open


class Key(SymbolicObject):
    def __init__(self, name, location, state):
        super().__init__(name)

        # One wall object handles all the walls in the world
        self.location = location
        self.state = state  # 0: on the ground, 1: in the taxi, 2: gone, used up


class Taxi(SymbolicObject):
    def __init__(self, name, location):
        super().__init__(name)

        # TODO: need to see if holding something?
        self.location = location


class Passenger(SymbolicObject):
    def __init__(self, name, location, state):
        super().__init__(name)

        self.location = location
        self.state = state  # In taxi or on pickup location


class Destination(SymbolicObject):
    def __init__(self, name, location, state):
        super().__init__(name)

        # The destinations don't move so I don't know if they need a state?
        # Or they can just store the location.
        self.location = location
        self.state = state


class PredicateType(Enum):
    # These are currently used for heist
    # Door world may use some as well but I haven't looked at it for a while
    TOUCH_LEFT = 1
    TOUCH_RIGHT = 2
    TOUCH_UP = 3
    TOUCH_DOWN = 4
    ON = 5
    IN = 6
    OPEN = 7


class Predicate:
    """
    Grounded predicates have a name, object references, and truth value

    """
    type = None
    object1 = ""
    object2 = ""
    value = False

    hash = 0

    def __init__(self, p_type, object1, object2, value):
        self.type = p_type
        self.value = value
        self.object1 = object1
        self.object2 = object2
        self.hash = hash((self.type, self.value, self.object1, self.object2))

    def __repr__(self):
        """For example, predicate on with args block1, table that is false will be ~on(block1, table)"""
        squiggle = "" if self.value else "~"
        name = type(self).__name__

        return f"{squiggle}{name}({self.object1}, {self.object2})"

    @staticmethod
    def create(p_type: PredicateType, o1: SymbolicObject, o2: SymbolicObject, **kwargs):
        """Factory method for creating predicates of specific type"""
        if p_type == PredicateType.TOUCH_LEFT:
            return TouchLeft(p_type, o1.name, o2.name, TouchLeft.evaluate(o1, o2, **kwargs))
        elif p_type == PredicateType.TOUCH_RIGHT:
            return TouchRight(p_type, o1.name, o2.name, TouchRight.evaluate(o1, o2, **kwargs))
        elif p_type == PredicateType.TOUCH_UP:
            return TouchUp(p_type, o1.name, o2.name, TouchUp.evaluate(o1, o2, **kwargs))
        elif p_type == PredicateType.TOUCH_DOWN:
            return TouchDown(p_type, o1.name, o2.name, TouchDown.evaluate(o1, o2, **kwargs))
        elif p_type == PredicateType.ON:
            return On(p_type, o1.name, o2.name, On.evaluate(o1, o2))
        elif p_type == PredicateType.IN:
            return In(p_type, o1.name, o2.name, In.evaluate(o1, o2))
        elif p_type == PredicateType.OPEN:
            return Open(p_type, o1.name, o2.name, Open.evaluate(o1, o2))
        else:
            raise ValueError(f'Unrecognized effect type: {p_type}')

    def copy(self):
        # Create a copy of this class with the same instance variables. type(self)() calls this Classes' constructor
        return type(self)(self.type, self.object1, self.object2, self.value)

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return (
                self.type == other.type and
                self.value == other.value and
                self.object1 == other.object1 and
                self.object2 == other.object2
        )


class Open(Predicate):
    @staticmethod
    def evaluate(o1, o2):
        return o1.open


class TouchLeft(Predicate):
    @staticmethod
    def evaluate(o1, o2, **kwargs):
        # I don't want predicates to work through walls, but now I have to pass walls in as a kwarg :(
        touching_wall = o1.location in kwargs['walls']['W']

        # If there is a wall in the way, we can't touch the thing
        if touching_wall and not type(o2) is Wall:
            return False

        if type(o2) is Wall:
            return touching_wall
        elif type(o2) is Key:  # I could use an or, but for some reason the gem's state is 0 when it's on the ground
            # Key needs to be on the ground to be touching it
            x, y = o1.location
            return (x - 1, y) == o2.location and o2.state == 1
        elif type(o2) is Gem:
            x, y = o1.location
            return (x - 1, y) == o2.location and o2.state == 1
        else:
            # If these used x and y this could work for 1D as well
            x, y = o1.location
            return (x - 1, y) == o2.location


class TouchRight(Predicate):
    @staticmethod
    def evaluate(o1, o2, **kwargs):
        # I don't want predicates to work through walls, but now I have to pass walls in as a kwarg :(
        touching_wall = o1.location in kwargs['walls']['E']

        # If there is a wall in the way, we can't touch the thing
        if touching_wall and not type(o2) is Wall:
            return False

        if type(o2) is Wall:
            return touching_wall
        elif type(o2) is Key:
            # Key needs to be on the ground to be touching it
            x, y = o1.location
            return (x + 1, y) == o2.location and o2.state == 1
        elif type(o2) is Gem:
            x, y = o1.location
            return (x + 1, y) == o2.location and o2.state == 1
        else:
            x, y = o1.location
            return (x + 1, y) == o2.location


class TouchUp(Predicate):
    @staticmethod
    def evaluate(o1, o2, **kwargs):
        # I don't want predicates to work through walls, but now I have to pass walls in as a kwarg :(
        touching_wall = o1.location in kwargs['walls']['N']

        # If there is a wall in the way, we can't touch the thing
        if touching_wall and not type(o2) is Wall:
            return False

        if type(o2) is Wall:
            return touching_wall
        elif type(o2) is Key:
            # Key needs to be on the ground to be touching it
            x, y = o1.location
            return (x, y + 1) == o2.location and o2.state == 1
        elif type(o2) is Gem:
            x, y = o1.location
            return (x, y + 1) == o2.location and o2.state == 1
        else:
            x, y = o1.location
            return (x, y + 1) == o2.location


class TouchDown(Predicate):
    @staticmethod
    def evaluate(o1, o2, **kwargs):
        # I don't want predicates to work through walls, but now I have to pass walls in as a kwarg :(
        touching_wall = o1.location in kwargs['walls']['S']

        # If there is a wall in the way, we can't touch the thing
        if touching_wall and not type(o2) is Wall:
            return False

        if type(o2) is Wall:
            return touching_wall
        elif type(o2) is Key:
            # Key needs to be on the ground to be touching it
            x, y = o1.location
            return (x, y - 1) == o2.location and o2.state == 1
        elif type(o2) is Gem:
            x, y = o1.location
            return (x, y - 1) == o2.location and o2.state == 1
        else:
            x, y = o1.location
            return (x, y - 1) == o2.location


class On(Predicate):
    @staticmethod
    def evaluate(o1, o2):
        if type(o2) is Wall:
            return False  # Can't be on a wall
        elif type(o2) is Key:
            # We are only on it if it is on the ground, not in the taxi nor non-existent
            return o2.state == 1 and o1.location == o2.location
        elif type(o2) is Gem:
            # We can't be on the gem when it is being held
            return o2.state == 1 and o1.location == o2.location
        else:
            return o1.location == o2.location


class In(Predicate):
    @staticmethod
    def evaluate(o1, o2):
        # Only Key or Gem or Passenger can be in the taxi
        if type(o2) in [Key, Gem, Passenger] and o2.state == 0:
            return True
        else:
            return False
