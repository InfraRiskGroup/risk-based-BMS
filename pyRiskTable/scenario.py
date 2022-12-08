import warnings
import numpy as np
from scipy.special import roots_legendre


class _AccuracyWarning(Warning):
    pass


def generate_primary_event(a, b, likelihood_func=None, kw_likelihood={}, vec_likelihood=False,
                           consequence_func=None, kw_consequence={}, vec_consequence=False,
                           tol=1e-8, rtol=1e-8, min_order=1, max_order=50):
    """Conduct Gauss-Legendre quadrature based on error and/or order (i.e., number of integration points).
    This function is adapted and simplified based on the `scipy` version.

    Args:
        a (float): lower limit of IM level under consideration.
        b (float): upper limit of IM level under consideration.
        likelihood_func (function): hazard likelihood function.
        kw_likelihood (dict, optional): keyward arguments of `likelihood_func`. Defaults to {}.
        vec_likelihood (bool, optional): Whether `likelihood_func` accepts 1D array input. Defaults to False.
        consequene_func (function): **expected** consequene function.
        kw_consequene (dict, optional): keyward arguments of `consequene_func`. Defaults to {}.
        vec_consequene (bool, optional): Whether `consequene_func` accepts 1D array input. Defaults to False.
        tol (float, optional): tolerance on integral difference between consecutive integration points. Defaults to 1e-8.
        rtol (float, optional): relative tolerance on integral difference between consecutive integration points. Defaults to 1e-8.
        min_order (int, optional): minimum Gauss-Legendre order (inclusive). Defaults to 1.
        max_order (int, optional): maximum Gauss-Legendre order (inclusive). Defaults to 50.

    Returns:
        ims (1D array): IM levels for risk approximation
        scalers (1D array): weights to scale the consequences
        likes (1D array): hazard likelihoods at IMs
        cqs (1D array): expected consequences at IMs
        val (float): approximated risk value
        err (float): absolute error between the last two orders
    """
    assert likelihood_func is not None, "Must provide likelihood_func"
    assert consequence_func is not None, "Must provide consequence_func"

    if vec_likelihood:
        likelihood_vfunc = lambda im: likelihood_func(im, **kw_likelihood)
    else:
        likelihood_vfunc = np.vectorize(lambda im: likelihood_func(im, **kw_likelihood))

    if vec_consequence:
        consequence_vfunc = lambda im: consequence_func(im, **kw_consequence)
    else:
        consequence_vfunc = np.vectorize(lambda im: consequence_func(im, **kw_consequence))

    val = np.inf
    err = np.inf
    for n in range(min_order, max_order+1):
        x, w = roots_legendre(n)
        x = np.real(x)

        ims = (b-a)*(x+1)/2.0+a
        scalers = (b-a)/2.0 * w

        likes = likelihood_vfunc(ims)
        cqs = consequence_vfunc(ims)
        newval = np.sum(scalers*likes*cqs, axis=-1)

        err = abs(newval-val)
        val = newval

        if err < tol or err < rtol*abs(val):
            break
    else:
        if max_order > min_order:
            warnings.warn(f"max order ({max_order:d}) reached. Latest difference = {err:e}", _AccuracyWarning)

    return ims, scalers, likes, cqs, val, err



def generate_secondary_event(a, b, c, d, likelihood_func=None, kw_likelihood={}, vec_likelihood=False,
                             consequence_func=None, kw_consequence={}, vec_consequence=False,
                             tol=1e-8, rtol=1e-8, min_order=1, max_order=50):
    """Conduct 2D Gauss-Legendre quadrature for cascading events based on error and/or order (i.e., number of integration points).
    This function is adapted and simplified based on the `scipy` version.

    Args:
        integrand (function): integrand function, return risk given an intensity measure (IM).
        a (float): lower limit of primary IM level under consideration.
        b (float): upper limit of primary IM level under consideration.
        c (float): lower limit of secondary IM level under consideration.
        d (float): upper limit of secondary IM level under consideration.
        likelihood_func (function): hazard likelihood function. Must take two arguments, IM of primary and IM of secondary
        kw_likelihood (dict, optional): keyward arguments of `likelihood_func`. Defaults to {}.
        vec_likelihood (bool, optional): Whether `likelihood_func` accepts 1D array input. Defaults to False.
        consequene_func (function): **expected** consequene function.
        kw_consequene (dict, optional): keyward arguments of `consequene_func`. Defaults to {}.
        vec_consequene (bool, optional): Whether `consequene_func` accepts 1D array input. Defaults to False.
        tol (float, optional): tolerance on integral difference between consecutive integration points. Defaults to 1e-8.
        rtol (float, optional): relative tolerance on integral difference between consecutive integration points. Defaults to 1e-8.
        min_order (int, optional): minimum Gauss-Legendre order (inclusive). Defaults to 1.
        max_order (int, optional): maximum Gauss-Legendre order (inclusive). Defaults to 50.

    Returns:
        im (2D array): IM levels for risk approximation (column 1 is primary IM; column 2 is secondary IM).
        scaler (2D array): weights to scale the consequences (the product of each row is the weight of the hazard scenario).
        likes (1D array): hazard likelihoods at IMs
        cqs (1D array): expected consequences at IMs
        val (float): approximated risk value
        err (float): absolute error between the last two orders
    """

    assert likelihood_func is not None, "Must provide likelihood_func"
    assert consequence_func is not None, "Must provide consequence_func"

    if vec_likelihood:
        likelihood_vfunc = lambda im1, im2: likelihood_func(im1, im2, **kw_likelihood)
    else:
        likelihood_vfunc = np.vectorize(lambda im1, im2: likelihood_func(im1, im2, **kw_likelihood))

    if vec_consequence:
        consequence_vfunc = lambda im: consequence_func(im, **kw_consequence)
    else:
        consequence_vfunc = np.vectorize(lambda im: consequence_func(im, **kw_consequence))

    val = np.inf
    err = np.inf
    for n in range(min_order, max_order+1):
        pts1, pts1_w = roots_legendre(n)
        pts1 = np.real(pts1)
        pts2, pts2_w = pts1, pts1_w

        im1 = (b-a)*(pts1+1)/2.0+a
        im2 = (d-c)*(pts2+1)/2.0+c
        im1, im2 = np.meshgrid(im1, im2)
        im1 = im1.ravel()
        im2 = im2.ravel()

        w1 = (b-a)/2.0 * pts1_w
        w2 = (d-c)/2.0 * pts2_w
        w1, w2 = np.meshgrid(w1, w2)
        w1 = w1.ravel()
        w2 = w2.ravel()

        # Jdet = (b-a)*(d-c)/4.0

        cqs = consequence_func(im2, **kw_consequence)
        likes = likelihood_func(im1, im2, **kw_likelihood)
        newval = np.sum(w1*w2*likes*cqs, axis=-1)

        err = abs(newval-val)
        val = newval

        if err < tol or err < rtol*abs(val):
            break
    else:
        if max_order > min_order:
            warnings.warn(f"max order ({max_order:d}) reached. Latest difference = {err:e}", _AccuracyWarning)

    im = np.vstack([im1.flatten(), im2.flatten()]).T
    scaler = np.vstack([w1.flatten(), w2.flatten()]).T
    
    return im, scaler, likes, cqs, val, err
