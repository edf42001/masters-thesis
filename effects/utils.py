from typing import List

from effects.effect import EffectType, Effect, NoChange


def get_effects(att: int, s1: List[int], s2: List[int], is_bool: bool = False) -> List[Effect]:
    """Returns a list of effects of each type that would transform attribute att
    in s1 into its value in s2"""
    if s1[att] == s2[att]:
        return [NoChange()]
    elif is_bool:
        return [Effect.create(EffectType.SET_TO_NUMBER, s1[att], s2[att])]
    else:
        # Could modify this with  if e_type != EffectType.NO_CHANGE to ignore No_Change effects
        # Have modified this to only return effects of increment unless is bool
        # Basically, we need to know only the correct effects
        #  if e_type == EffectType.INCREMENT
        return [Effect.create(e_type, s1[att], s2[att]) for e_type in EffectType if e_type != EffectType.NO_CHANGE]
