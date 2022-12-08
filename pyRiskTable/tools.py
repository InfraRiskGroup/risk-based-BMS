# TODO: add custom hazard and fragility functions
# currently, there is no turn-key implementation for cascading hazard yet
# since it requires a defintiion of 2d hazard curve (i.e., conditional probability of secondary intensity)

import pandas as pd
import numpy as np
 
 
def export_primary_scenarios(ims=None, scalers=None, likes=None, cqs=None,
                             hazard_curve=None, vec_func=False,
                             filepath=None):

    scenario_df = pd.DataFrame()
    scenario_df['Likelihood'] = likes
    scenario_df['Intensity'] = ims
    scenario_df['Consequences'] = cqs
    scenario_df['Weights'] = scalers

    if hazard_curve is not None:
        if vec_func:
            vfunc = hazard_curve
        else:
            vfunc = np.vectorize(hazard_curve)
            
        rates = vfunc(ims)
        scenario_df['Return Period'] = 1/rates
    
    if filepath is None:
        filepath = "tmp.csv"
    scenario_df.to_csv(filepath)


def risk_from_primary_scenarios(filepath=None, new_consequence=None):

    assert filepath is not None, "Must provide filepath"
    
    scenario_df = pd.read_csv(filepath)
    
    likes = scenario_df['Likelihood']

    if new_consequence is None:
        cqs = scenario_df['Consequences']
    else:
        cqs = new_consequence
    
    scalers = scenario_df['Weights']
    
    risk = np.sum(likes*cqs*scalers)

    return risk


def user_hazard_likelihood(filepath=None, im_key='Intensity', like_key='Likelihood'):

    assert filepath is not None, "Must provide filepath"

    hazard_df = pd.read_csv(filepath)
    
    func = lambda im: np.interp(im, hazard_df[im_key], hazard_df[like_key])
    
    return func
    

def user_fragility_curve(filepath=None, im_key='IM', ds_key='Slight'):

    assert filepath is not None, "Must provide filepath"

    fragility_df = pd.read_csv(filepath)
    
    func = lambda im: np.interp(im, fragility_df[im_key], fragility_df[ds_key])
    
    return func