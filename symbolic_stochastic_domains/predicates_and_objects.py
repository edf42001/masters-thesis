from enum import Enum


class SymbolicObject:
    """Represents a generic object in a symbolic world"""
    def __init__(self, name=""):
        self.name = name

    def __repr__(self):
        # Print the object type plus our unique id
        return f"{type(self).__name__}({self.name})"


class Taxi(SymbolicObject):
    def __init__(self, name="", x=0):
        super().__init__(name)

        self.x = x


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


class Wall(SymbolicObject):
    def __init__(self, name="", locations=None):
        super().__init__(name)

        # One wall object handles all the walls in the world
        self.locations = locations


class Gem2D(SymbolicObject):
    def __init__(self, name, location, state):
        super().__init__(name)

        self.location = location
        self.state = state


class Wall2D(SymbolicObject):
    def __init__(self, name, locations):
        super().__init__(name)

        # One wall object handles all the walls in the world
        self.locations = locations


class Lock2D(SymbolicObject):
    def __init__(self, name, location, open):
        super().__init__(name)

        # One wall object handles all the walls in the world
        self.location = location
        self.open = open


class Key2D(SymbolicObject):
    def __init__(self, name, location, state):
        super().__init__(name)

        # One wall object handles all the walls in the world
        self.location = location
        self.state = state  # 0: on the ground, 1: in the taxi, 2: gone, used up


class Taxi2D(SymbolicObject):
    def __init__(self, name, location):
        super().__init__(name)

        # TODO: need to see if holding something?
        self.location = location


class PredicateType(Enum):
    # These are currently used in door world
    TOUCH_LEFT = 0
    TOUCH_RIGHT = 1
    ON = 2
    OPEN = 3

    # These are currently used for heist
    TOUCH_LEFT2D = 4
    TOUCH_RIGHT2D = 5
    TOUCH_UP2D = 6
    TOUCH_DOWN2D = 7
    ON2D = 8
    IN = 9


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
    def create(p_type: PredicateType, o1: SymbolicObject, o2: SymbolicObject):
        """Factory method for creating predicates of specific type"""
        if p_type == PredicateType.TOUCH_LEFT:
            return TouchLeft(p_type, o1.name, o2.name, TouchLeft.evaluate(o1, o2))
        elif p_type == PredicateType.TOUCH_RIGHT:
            return TouchRight(p_type, o1.name, o2.name, TouchRight.evaluate(o1, o2))
        elif p_type == PredicateType.ON:
            return On(p_type, o1.name, o2.name, On.evaluate(o1, o2))
        elif p_type == PredicateType.OPEN:
            return Open(p_type, o1.name, o2.name, Open.evaluate(o1, o2))
        elif p_type == PredicateType.TOUCH_LEFT2D:
            return TouchLeft2D(p_type, o1.name, o2.name, TouchLeft2D.evaluate(o1, o2))
        elif p_type == PredicateType.TOUCH_RIGHT2D:
            return TouchRight2D(p_type, o1.name, o2.name, TouchRight2D.evaluate(o1, o2))
        elif p_type == PredicateType.TOUCH_UP2D:
            return TouchUp2D(p_type, o1.name, o2.name, TouchUp2D.evaluate(o1, o2))
        elif p_type == PredicateType.TOUCH_DOWN2D:
            return TouchDown2D(p_type, o1.name, o2.name, TouchDown2D.evaluate(o1, o2))
        elif p_type == PredicateType.TOUCH_UP2D:
            return TouchUp2D(p_type, o1.name, o2.name, TouchUp2D.evaluate(o1, o2))
        elif p_type == PredicateType.ON2D:
            return On2D(p_type, o1.name, o2.name, On2D.evaluate(o1, o2))
        elif p_type == PredicateType.IN:
            return In(p_type, o1.name, o2.name, In.evaluate(o1, o2))
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


class TouchLeft(Predicate):
    @staticmethod
    def evaluate(o1, o2):
        if type(o2) is Wall:
            # Because walls are static and cover the whole world, they are treated separately
            return o1.x in o2.locations
        else:
            return o1.x == o2.x + 1


class TouchRight(Predicate):
    @staticmethod
    def evaluate(o1, o2):
        if type(o2) is Wall:
            # Because walls are static and cover the whole world, they are treated separately
            return (o1.x + 1) in o2.locations
        else:
            return o1.x == o2.x - 1


class On(Predicate):
    @staticmethod
    def evaluate(o1, o2):
        # Taxi can't be on a wall
        if type(o2) is Wall:
            return False
        else:
            return o1.x == o2.x


class Open(Predicate):
    @staticmethod
    def evaluate(o1, o2):
        return o1.open


# Whoops, the other coordinate needs to be equal
class TouchLeft2D(Predicate):
    @staticmethod
    def evaluate(o1, o2):
        # The wall object refers to all walls and is handled separately
        if type(o2) is Wall2D:
            return o1.location in o2.locations['W']
        else:
            # If these used x and y this could work for 1D as well
            x, y = o1.location
            return (x - 1, y) == o2.location


class TouchRight2D(Predicate):
    @staticmethod
    def evaluate(o1, o2):
        if type(o2) is Wall2D:
            return o1.location in o2.locations['E']
        else:
            x, y = o1.location
            return (x + 1, y) == o2.location


class TouchUp2D(Predicate):
    @staticmethod
    def evaluate(o1, o2):
        if type(o2) is Wall2D:
            return o1.location in o2.locations['N']
        else:
            x, y = o1.location
            return (x, y + 1) == o2.location


class TouchDown2D(Predicate):
    @staticmethod
    def evaluate(o1, o2):
        if type(o2) is Wall2D:
            return o1.location in o2.locations['S']
        else:
            x, y = o1.location
            return (x, y - 1) == o2.location


class On2D(Predicate):
    @staticmethod
    def evaluate(o1, o2):
        if type(o2) is Wall2D:
            return False  # Can't be on a wall
        elif type(o2) is Key2D:
            # If the key is no longer there, even if we are at the location we aren't on it, Nor if it is in the taxi
            return o2.state == 0 and o1.location == o2.location
        else:
            return o1.location == o2.location


class In(Predicate):
    @staticmethod
    def evaluate(o1, o2):
        # Only Key or Gem can be in the taxi
        if type(o2) is Key2D:
            return o2.state == 1
        elif type(o2) is Gem2D:
            return o2.state == 1
        else:
            return False
