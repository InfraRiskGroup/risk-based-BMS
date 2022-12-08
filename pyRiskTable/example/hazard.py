import numpy as np
import math as mt
from scipy import stats

from pyRiskTable.example.constants import RATE, M_HAZARD, R_HAZARD, SIGMA_HAZARD
from pyRiskTable.example.constants import PGD_SIGMA, LQ_CLASS, WATER_DEPTH


# example hazard curve

def hazard_curve(im):
    """Example hazard curve based on Baker (2013), section 2.3.

    Args:
        im (float or 1D array): IM(s) of interest

    Returns:
        hazard_rate (float or 1D array): annual rate of exceedance of the hazard curve
    """
    
    ln_median = -0.152 + 0.859*M_HAZARD - 1.803*mt.log(R_HAZARD+25)

    hazard_rate = RATE*(1-stats.lognorm.cdf(im, SIGMA_HAZARD, scale=mt.exp(ln_median)))

    return hazard_rate


def hazard_likelihood(im):
    """Example hazard likelihood at different IM(s).
    It is the gradient of the example hazard curve in Baker (2013), section 2.3.

    Args:
        im (float or 1D array): IM(s) of interest

    Returns:
        hazard_like (float or 1D array): hazard likelihood(s) at IM(s)
    """

    ln_median = -0.152 + 0.859*M_HAZARD - 1.803*mt.log(R_HAZARD+25)

    hazard_like = RATE*stats.lognorm.pdf(im, SIGMA_HAZARD, scale=mt.exp(ln_median))

    return hazard_like


def inv_hazard_curve(prob):
    """Find the IM associated with an annual rate of exceedance on the hazard curve.

    Args:
        prob (float or array): annual rate(s) of exceedance

    Returns:
        im (float or array): IMs at the annual rates
    """

    ln_median = -0.152 + 0.859*M_HAZARD - 1.803*mt.log(R_HAZARD+25)

    im = stats.lognorm.ppf(1-prob/RATE, SIGMA_HAZARD, scale=mt.exp(ln_median))

    return im


def second_probability(im1, sc=LQ_CLASS, magnitude=M_HAZARD, water_depth=WATER_DEPTH):
    """return probability of occurrence of the secondary hazard.
    Use HAZUS ground lateral spreading as an example (4.2.2.1.2)

    Args:
        im1 (float or 1D array): Primary hazard intensity. PGA in g
        sc (int, optional): susceptibility class from 0 (None) to 5 (Very high). Defaults to 3 (Moderate).
        magnitude (float, optional): EQ magnitude. Defaults to 7.5.
        water_depth (_type_, optional): depth to the groundwater in feet. Defaults to 0.07/0.022.

    Returns:
        prob_lq (float or 1D array): probability of liquefaction given PGA
    """

    im1 = np.asarray(im1)
    scalar_input = False
    if im1.ndim == 0:
        im1 = im1[np.newaxis]   # Makes scalar im_a 1D array
        scalar_input = True

    lq_probability = {
        5: lambda a: 9.09*a - 0.82,
        4: lambda a: 7.67*a - 0.92,
        3: lambda a: 6.67*a - 1.00,
        2: lambda a: 5.57*a - 1.18,
        1: lambda a: 4.16*a - 1.08,
        0: lambda a: 0,
    }
    lq_pml = {5: 0.25, 4: 0.20, 3: 0.10, 2: 0.05, 1: 0.02, 0: 0}

    M, dw = magnitude, water_depth
    kM = 0.0027*M**3 - 0.0267*M**2 - 0.2055*M + 2.9188
    kw = 0.022*dw + 0.93

    p_standard = lq_probability[sc](im1)
    p_standard[p_standard<0] = 0.0
    p_standard[p_standard>1] = 1.0

    prob_lq = p_standard / (kM*kw) * lq_pml[sc]
    prob_lq[prob_lq<0] = 0.0
    prob_lq[prob_lq>1] = 1.0

    if scalar_input:
        return prob_lq[0]

    return prob_lq


def d_second_hazard_curve(im1, im2, sc=LQ_CLASS, magnitude=M_HAZARD, sigma=PGD_SIGMA):
    """ PDF of lateral spreading (PGD) based on a modified version of the HAZUS model (4.2.2.1.3.1)
    Modifications include: (a) remove the cap on normalized PGA (4.0) and (b) the expected value is
    treated as the median value. Additionally, it is assumed that PGD follow lognormal distribution
    with an assumed dispersion factor (COV) sigma = 0.2.

    Args:
        im1 (float or 1D array): PGA in g
        im2 (float or 1D array): PGD in inch. Must of the same shape as im_1
        sc (int, optional): Susceptiability class. Defaults to 3.
        magnitude (float, optional): EQ magnitude. Defaults to 7.5.
        sigma (float, optional): Assumed dispersion factor (COV). Defaults to 0.2.

    Returns:
        im2_pdf: PDF of PGD
    """


    im1 = np.asarray(im1)
    im2 = np.asarray(im2)
    scalar_input = False
    if im1.ndim == 0:
        im1 = im1[np.newaxis]   # Makes scalar im_a 1D array
        im2 = im2[np.newaxis]   # Makes scalar im_a 1D array
        scalar_input = True

    assert len(im1) == len(im2), "primary and secondary IM must have the same length"

    PGA_trigger = {5: 0.09, 4: 0.12, 3: 0.15, 2: 0.21, 1: 0.26, 0: np.inf}
    M = magnitude
    kdelta = 0.0086*M**3 - 0.0914*M**2 + 0.4698*M - 0.9835

    x = im1 / PGA_trigger[sc]
    PGD_standard = np.zeros_like(x)

    mask0 = x<1
    mask1 = (x>=1) & (x<2)
    mask2 = (x>=2) & (x<3)
    mask3 = x>=3

    PGD_standard[mask0] = 0
    PGD_standard[mask1] = 12*x[mask1] -12
    PGD_standard[mask2] = 18*x[mask2] - 24
    PGD_standard[mask3] = 70*x[mask3] - 180

    PGD_median = kdelta * PGD_standard

    im2_pdf = np.zeros_like(im2)
    for i, (im, pgd) in enumerate(zip(im2, PGD_median)):
        if pgd == 0:
            im2_pdf[i] = 0
        else:
            im2_pdf[i] = stats.lognorm.pdf(im, sigma, scale=pgd)
    
    if scalar_input:
        return im2_pdf[0]

    return im2_pdf


def im2d_likelihood(im1, im2, sc=LQ_CLASS, magnitude=M_HAZARD, water_depth=WATER_DEPTH, sigma=PGD_SIGMA):
    """Return the likelihood of the joint event of IM1=im_1 and IM2=im_2
    where IM1 and IM2 are the IMs of primary and secondary hazards.

    Args:
        im1 (float or 1D array): PGA in g
        im2 (float or 1D array): PGD in inch. Must of the same shape as im_1

    Returns:
        im_likelihood: event likelihood
    """

    pdf_IM1 = hazard_likelihood(im1)

    p_second_occur = second_probability(im1, sc=sc, magnitude=magnitude, water_depth=water_depth)

    pdf_IM2 = d_second_hazard_curve(im1, im2, sc=sc, magnitude=magnitude, sigma=sigma)

    im_likelihood = pdf_IM1 * p_second_occur * pdf_IM2

    return im_likelihood