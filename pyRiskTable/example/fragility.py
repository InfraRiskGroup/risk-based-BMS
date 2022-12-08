import numpy as np
from scipy import stats

from pyRiskTable.example.constants import PGA_S, PGA_M, PGA_E, PGA_C, PGA_SIGMA
from pyRiskTable.example.constants import PGD_S, PGD_M, PGD_E, PGD_C, PGD_SIGMA


# example fragility functions (ground shaking)
def fragility_curve(damage_state='s', scale=1.0):
    """Return example fragility curve based on Basoz and Mander (1999) Example 1. The IM is PGA.

    Args:
        damage_state (str, optional): Damage state. Support names in HAZUS and their first letters. Defaults to 's'.
        scale (float, optional): Scaler to median PGA. Use when modeling deterioration. Defaults to 1.0.

    Returns:
        func (function): Fragility function. Vector-supporting function that returns probabilities at different IMs.
    """

    median_s = scale*PGA_S
    median_m = scale*PGA_M
    median_e = scale*PGA_E
    median_c = scale*PGA_C
    
    if damage_state.lower() in ('s', 'slight'):
        func = lambda im: stats.lognorm.cdf(im, PGA_SIGMA, scale=median_s)
    elif damage_state.lower() in ('m', 'moderate'):
        func = lambda im: stats.lognorm.cdf(im, PGA_SIGMA, scale=median_m)
    elif damage_state.lower() in ('e', 'extensive'):
        func = lambda im: stats.lognorm.cdf(im, PGA_SIGMA, scale=median_e)
    elif damage_state.lower() in ('c', 'complete', 'collapse'):
        func = lambda im: stats.lognorm.cdf(im, PGA_SIGMA, scale=median_c)
    
    return func


def fragility_curve_second(damage_state='s', scale=1.0):
    """Return example fragility curve for secondary hazard
    based on HAZUS liquefation-induced ground deformation model (Table 7-6 of v4.2 tech manual).

    Args:
        damage_state (str, optional): Damage state. Support names in HAZUS and their first letters. Defaults to 's'.
        scale (float, optional): Scaler to median PGD. Not used for liquefaction. Defaults to 1.0.

    Returns:
        func (function): Fragility function. Vector-supporting function that returns probabilities at different IMs.
    """

    if damage_state.lower() in ('s', 'slight'):
        func = lambda im: stats.lognorm.cdf(im, PGD_SIGMA, scale=PGD_S)
    elif damage_state.lower() in ('m', 'moderate'):
        func = lambda im: stats.lognorm.cdf(im, PGD_SIGMA, scale=PGD_M)
    elif damage_state.lower() in ('e', 'extensive'):
        func = lambda im: stats.lognorm.cdf(im, PGD_SIGMA, scale=PGD_E)
    elif damage_state.lower() in ('c', 'complete', 'collapse'):
        func = lambda im: stats.lognorm.cdf(im, PGD_SIGMA, scale=PGD_C)
    
    return func