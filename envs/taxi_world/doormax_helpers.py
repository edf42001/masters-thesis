from taxi_world_env import TaxiWorldEnv, ACTION


class TaxiState:
    """
    Encodes important information about state of taxi world. Knowledge specific to env
    such as where the pickup and destination are, and whether the current taxi position is near a wall,
    are handled by the env
    """

    def __init__(self, taxi_x, taxi_y, passenger_in_taxi, pickup, destination):
        self.taxi_x = taxi_x
        self.taxi_y = taxi_y
        self.passenger_in_taxi = passenger_in_taxi
        self.pickup = pickup
        self.destination = destination


class TaxiTerms:
    """
    Structure to organize taxi terms. Terms consist of:
    touchN/S/E/W(taxi, wall), on(taxi, passenger), on(taxi, destination), passenger.in_taxi,
    and their negations.

    I don't really know how to organize this so for now it will be a bunch of methods.
    Eventually there could be a "Term" class

    """
    def __init__(self, env):
        self.terms = self.boolean_arr_to_string([
            self.touch_north(env),
            self.touch_east(env),
            self.touch_south(env),
            self.touch_west(env),
            self.on_destination(env),
            self.on_passenger(env),
            self.passenger_in_taxi(env),
        ])

    def boolean_arr_to_string(self, str):
        return ["1" if value else "0" for value in str]

    def touch_north(self, env):
        return env.touching_wall(ACTION.NORTH)

    def touch_east(self, env):
        return env.touching_wall(ACTION.EAST)

    def touch_south(self, env):
        return env.touching_wall(ACTION.SOUTH)

    def touch_west(self, env):
        return env.touching_wall(ACTION.WEST)

    def on_destination(self, env: TaxiWorldEnv):
        return env.stops[env.current_dropoff] == [env.taxi_x, env.taxi_y]

    def on_passenger(self, env):
        return env.stops[env.current_pickup] == [env.taxi_x, env.taxi_y]

    def passenger_in_taxi(self, env: TaxiWorldEnv):
        return env.passenger_in_taxi


def condition_matches(c1, c2):
    """
    If a condition matches, it means the are the same and the same effect could happen
    This is if all the terms are the same or one of the terms is "doesn't matter"
    """

    # TODO: What if the star is in s2, not s1?
    for s1, s2 in zip(c1, c2):
        if (s1 != "*") and (s1 != s2):
            return False

    return True


def commute_condition_strings(c1, c2):
    """
    Commute two strings. Basically, comparing the conditions of action/effect
    pairs to decide which state variables matter (0, 1) or don't matter (*)
    """
    ret = ""
    for s1, s2 in zip(c1, c2):
        if s1 == "0" and s2 == "0":
            ret += "0"
        elif s1 == "1" and s2 == "1":
            ret += "1"
        # TODO: Could everything below this be replace with else: add "*"
        elif (s1 == "0" and s2 == "1") or (s1 == "1" and s2 == "0"):
            ret += "*"
        elif s1 == "*" or s2 == "*":
            ret += "*"
        else:
            print("Bad commute string conditions")

    return ret


if __name__ == "__main__":
    c1 = "1001001"
    c2 = "1000001"
    ret = commute_condition_strings(c1, c2)
    print(ret)

    print(condition_matches(c2, c1))
