import itertools
from typing import List

from effects.effect import EffectType, Effect, JointEffect, JointNoEffect


def get_effects(att: int, s1: List[int], s2: List[int], is_bool: bool = False) -> List[Effect]:
    """Returns a list of effects of each type that would transform attribute att
    in s1 into its value in s2"""
    if s1[att] == s2[att]:
        return []
    elif is_bool:
        return [Effect.create(EffectType.SET_TO_NUMBER, s1[att], s2[att])]
    else:
        # Could modify this with  if e_type != EffectType.NO_CHANGE to ignore No_Change effects
        # Have modified this to only return effects of increment unless is bool
        # Basically, we need to know only the correct effects
        return [Effect.create(e_type, s1[att], s2[att]) for e_type in EffectType if e_type == EffectType.INCREMENT]


def eff_joint(curr_state: List[int], next_state: List[int], is_bool: bool = False) -> List[JointEffect]:
    """Returns a list of all possible JointEffects that would transform the
    current state into the next state"""
    # For each attribute, get list of possible single effects
    att_list, possible_eff_list, = [], []
    for att in range(len(curr_state)):
        # Is bool used to be set_only. Basically I think this is for categorical variables
        # Like passenger in taxi, or keys being held, non-existent, or there.
        possible_eff = get_effects(att, curr_state, next_state, is_bool=is_bool)
        if not possible_eff:
            # Skip attributes that do not change
            continue
        else:
            att_list.append(att)
            possible_eff_list.append(possible_eff)

    # Return possible joint effects
    if len(att_list) == 0:
        # No attributes change
        return [JointNoEffect()]
    elif len(att_list) == 1:
        # Only one attribute changes, the joint effects only involve this att
        return [JointEffect(att_list, [eff]) for eff in possible_eff_list[0]]
    else:
        # Multiple attributes change, need to cross product possible effects
        joint_eff_list = itertools.product(*possible_eff_list)
        return [JointEffect(att_list, eff_list) for eff_list in joint_eff_list]