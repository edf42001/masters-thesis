def boolean_arr_to_string(arr):
    """Array of booleans to 0 or 1 string"""
    return ["1" if value else "0" for value in arr]


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
