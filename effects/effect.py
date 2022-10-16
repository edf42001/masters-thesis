from enum import Enum
from typing import Union


class EffectType(Enum):
    INCREMENT = 0
    SET_TO_NUMBER = 1
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


class NoiseEffect:
    """No Effect: the entire state does not change"""
    def __init__(self):
        super().__init__([], [])
