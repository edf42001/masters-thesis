from enum import Enum
from typing import List, Union


class EffectType(Enum):
    INCREMENT = 0
    SET_TO_NUMBER = 1
    # SET_TO_BOOL = 2
    NO_CHANGE = 2  # 3  # Can't have NO_CHANGE because it gets autofilled even if there is a change in get_effects


class Effect:
    """
    Specifies one numerical effect applied to one state variable
    Note that a boolean can be represented by a zero or a one, so numbers work here too
    """
    type = None
    value = None
    hash = None

    def __eq__(self, other):
        return self.type == other.type and self.value == other.value

    def __hash__(self):
        return self.hash

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def create(e_type: EffectType, s_var: Union[int, float], next_s_var: Union[int, float]):
        """Factory for creating effect of specific type"""
        if e_type == EffectType.INCREMENT:
            return Increment(s_var, next_s_var)
        elif e_type == EffectType.SET_TO_NUMBER:
            return SetToNumber(s_var, next_s_var)
        elif e_type == EffectType.NO_CHANGE:
            return NoChange()
        # elif e_type == EffectType.SET_TO_BOOL:
        #     return SetToBool(next_s_var)
        else:
            raise ValueError(f'Unrecognized effect type: {e_type}')

    def apply_to(self, s_var: Union[int, float]) -> float:
        """Returns the result of applying this effect to s_var"""
        raise NotImplementedError()


class NoChange(Effect):
    def __init__(self):
        pass
        self.type = EffectType.NO_CHANGE
        self.hash = hash(self.type)  # Will be constant

    """No Change Effect: the state variable does not change"""
    def apply_to(self, s_var: Union[int, float]):
        return s_var

    def __str__(self):
        return 'NoChange'


class Increment(Effect):
    """Increment Effect: determines the numerical difference of a variable between states"""
    def __init__(self, s_var: Union[int, float], next_s_var: Union[int, float]):
        self.type = EffectType.INCREMENT
        self.value = next_s_var - s_var
        self.hash = hash((self.type, self.value))

    def apply_to(self, s_var: Union[int, float]):
        return s_var + self.value

    def __str__(self):
        return f'Increment({self.value})'


class SetToNumber(Effect):
    """Set-To Effect: sets the state variable to the value in next_s_var"""
    def __init__(self, s_var: Union[int, float], next_s_var: Union[int, float]):
        self.type = EffectType.SET_TO_NUMBER
        self.value = next_s_var
        self.hash = hash((self.type, self.value))

    def apply_to(self, s_var: Union[int, float]):
        return self.value

    def __str__(self):
        return f'SetToNumber({self.value})'


# class SetToBool(Effect):
#     """Set-To Effect: sets the state variable to the value in next_s_var"""
#     def __init__(self, s_var: Union[int, float], next_s_var: Union[int, float]):
#         self.type = EffectType.SET_TO_BOOL
#         self.value = next_s_var != 0  # False is 0, anything else is true
#         self.hash = hash((self.type, self.value))
#
#     def apply_to(self, s_var: Union[int, float]):
#         return self.value
#
#     def __str__(self):
#         return f'SetToBool({self.value})'


class JointEffect:
    """
    Maps each state var to a particular effect.
    If a state var does not appear, it is assumed to be constant
    """
    def __init__(self, att_list: List[int], eff_list: List[Effect]):
        d, temp = {}, []
        for a, e in zip(att_list, eff_list):
            # Only keep effects that are not NoEffect
            if e.type:
                d[a] = e
                temp.append((a, e.type, e.value))
        self.value = d
        self.hash = hash(frozenset(temp))

    def __eq__(self, other):
        # Make sure that the same number of attributes are included in each joint effect
        if len(self.value) != len(other.value):
            return False

        # Make sure that each attribute has the same effect
        # Checking the length allows us to only iterate over the keys of one
        for a, e in self.value.items():
            try:
                if other.value[a] != e:
                    return False
            except KeyError:
                return False
        return True

    def __hash__(self):
        return self.hash

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        # Sometimes this is empty from dallans code
        if not self.value.items():
            return '?'
        else:
            return '(' + ' '.join(f'<{a}, {str(e)}>' for a, e in self.value.items()) + ')'

    def is_empty(self) -> bool:
        """Check if this joint effect has no change to state"""
        return not self.value

    def apply_to(self, state: List[Union[int, float]]):
        """Applies the joint effect to a state in-place"""
        for att, effect in self.value.items():
            state[att] = effect.apply_to(state[att])


class JointNoEffect(JointEffect):
    """No Effect: the entire state does not change"""
    def __init__(self):
        super().__init__([], [])

    def __str__(self):
        return '<JointNoEffect>'

    def __repr__(self):
        return self.__str__()
