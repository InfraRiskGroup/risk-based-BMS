import numpy as np


def expected_consequence(im, fragility_funcs=None, cq_array=None):
    """Expected consequences given IM(s)

    Args:
        im (float or 1D array): IM values
        fragility_funcs (list): list of fragility functions organized in the order of S, M, E, C damage states
        cq_array (1D array): array of DS-dependent consequences organized in the order of S, M, E, C damage states

    Returns:
        risk (float or 1D array): expected consequences at IM(s)
    """

    assert fragility_funcs is not None, "Must provide a list of fragility functions"
    assert len(fragility_funcs) == len(cq_array), f"fragility_funcs and cq_array must have the same length"

    cdf_array = np.array([func(im) for func in fragility_funcs])
    prob_array = -np.diff(cdf_array, axis=0, append=0)

    risk = prob_array.T @ cq_array

    return risk


def risk_integrand(im, hazard_func=None, fragility_funcs=None, cq_array=None):
    """Integrand of risk integral at IM(s)

    Args:
        im (float or 1D array): IM values
        hazard_func (function): hazard likelihood function with respect to IM. Must support 1D array inputs
        fragility_funcs (list): list of fragility functions organized in the order of S, M, E, C damage states
        cq_array (1D array): array of DS-dependent consequences organized in the order of S, M, E, C damage states

    Returns:
        risks (float or 1D array): risk integrand values
    """

    assert hazard_func is not None, "Must provide hazard_func"

    conditional_risks = expected_consequence(im, fragility_funcs, cq_array)
    likelihoods = hazard_func(im)
    risks = conditional_risks * likelihoods
    
    return risks


def second_risk_integrand(im1, im2, hazard_func=None, fragility_funcs=None, cq_array=None):
    """Integrand of risk integral for secondary hazard risk assessment. Use HAZUS liquefaction as an example.

    Args:
        im1 (float or 1D array): PGA in g
        im2 (float or 1D array): PGD in inch. Must of the same shape as im_1
        hazard_func (function): hazard likelihood function with respect to IM1 and IM2. Must support two 1D array inputs
        fragility_funcs (list): list of fragility functions (secondary hazard) organized in the order of S, M, E, C damage states

    Returns:
        risks (float or 1D array): risk integrand values
    """

    assert hazard_func is not None, "Must provide hazard_func"

    risk_im = expected_consequence(im2, fragility_funcs=fragility_funcs, cq_array=cq_array)
    like_im = hazard_func(im1, im2)
    risks = risk_im * like_im

    return risks