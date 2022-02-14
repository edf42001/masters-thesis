"""
Helper classes for effects done to attributes.
For example if the taxi moved to the right to pos 4, the effect helper would return increment(1) and set_to(4)
"""


class Effect(object):
    """Generic Effect object"""
    def __init__(self):
        pass

    def type(self):
        """Returns the type of the object (currently the class type)"""
        return self.__class__

    def apply(self, initial_value):
        """Applies this effect to a value"""
        return initial_value


class Increment(Effect):
    def __init__(self, amount):
        super().__init__()
        self.amount = amount

    def apply(self, initial_value):
        return initial_value + self.amount

    def __str__(self):
        return "Increment({})".format(self.amount)

    def __eq__(self, obj):
        return isinstance(obj, Increment) and obj.amount == self.amount


class SetToNumber(Effect):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def apply(self, initial_value):
        return self.value

    def __str__(self):
        return "SetToNumber({})".format(self.value)

    def __eq__(self, obj):
        return isinstance(obj, SetToNumber) and obj.value == self.value


class SetToBoolean(Effect):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def apply(self, initial_value):
        return self.value

    def __str__(self):
        return "SetToBoolean({})".format(self.value)

    def __eq__(self, obj):
        return isinstance(obj, SetToBoolean) and obj.value == self.value


class NoChange(Effect):
    def __init__(self):
        super().__init__()

    def apply(self, initial_value):
        return initial_value

    def __str__(self):
        return "NoChange()"

    def __eq__(self, obj):
        return isinstance(obj, NoChange)


def get_effects(state, next_state, attribute):
    # TODO: This should act on objects and get effects from dictionaries. Instead of these if elses
    taxi_x = state[0]
    taxi_y = state[1]
    passenger_in_taxi = state[4]

    taxi_x_next = next_state[0]
    taxi_y_next = next_state[1]
    passenger_in_taxi_next = next_state[4]

    if attribute == "taxi.x":
        if taxi_x_next == taxi_x:
            # Could return NoChange()? Sadly I think it needs to return NoChange, so we can tell the differences
            # between actions whose effects we don't know and actions whose effects lead to no change
            return [NoChange()]
        else:
            return [Increment(taxi_x_next - taxi_x), SetToNumber(taxi_x_next)]
    elif attribute == "taxi.y":
        if taxi_y_next == taxi_y:
            return [NoChange()]
        else:
            return [Increment(taxi_y_next - taxi_y), SetToNumber(taxi_y_next)]
    elif attribute == "passenger.in_taxi":
        if passenger_in_taxi_next == passenger_in_taxi:
            return [NoChange()]
        else:
            # Could also have an (invert) here
            return [SetToBoolean(passenger_in_taxi_next)]
    else:
        print("Bad Attribute: {}".format(attribute))


def apply_effect(state, attribute, effect):
    # TODO: This should act on objects and get effects from dictionaries. Instead of these if elses
    taxi_x = state[0]
    taxi_y = state[1]
    passenger_in_taxi = state[4]

    if attribute == "taxi.x":
        taxi_x = effect.apply(taxi_x)
    elif attribute == "taxi.y":
        taxi_y = effect.apply(taxi_y)
    elif attribute == "passenger.in_taxi":
        passenger_in_taxi = effect.apply(passenger_in_taxi)
    else:
        print("Bad Attribute: {}".format(attribute))

    # Return the new state with modified values
    return taxi_x, taxi_y, state[2], state[3], passenger_in_taxi