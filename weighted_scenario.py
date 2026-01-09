# %%
import os
import numpy as np
import pandas as pd
import geopandas as gpd

from pyRiskTable.tools import fragility_curve, user_hazard_likelihood
from pyRiskTable.scenario import generate_primary_event
from pyRiskTable.risk import expected_consequence
from pyRiskTable.tools import export_primary_scenarios
from pyRiskTable.example.constants import CQ_S, CQ_M, CQ_E, CQ_C


if __name__ == "__main__":
    result_folder = 'OR-data'
    nsc_min, nsc_max = 10, 10
    cq_ratio_array = np.array([CQ_S, CQ_M, CQ_E, CQ_C])
    
    # import bridge id
    bridge_gdf = gpd.read_file("./bridge-info/bridges_w_soil.gpkg", layer='bridges_w_soil')
    bridge_info = bridge_gdf[[
        '8 - Structure Number',
        'CAT29 - Deck Area (sq. ft.)',
    ]]

    # # keep the first five records for testing
    # bridge_info = bridge_info.head(5)

    # # restart from a bridge id
    # index = bridge_info[bridge_info['8 - Structure Number'] == '08083A006 35617'].index[0]
    # bridge_info = bridge_info.iloc[(index+1):]

    risk_df = pd.DataFrame(columns=[
        'Structure Number', 'Deck Area (sq. ft.)',
        'Risk (sq. ft.)', 'Risk (ratio)'
    ])

    for index, row in bridge_info.iterrows():
        bridge_id = row['8 - Structure Number'].strip()
        deck_area = row['CAT29 - Deck Area (sq. ft.)']
        cq_array = deck_area * cq_ratio_array

        # import hazard curves and create interpolated curves
        # The last exceed could be NaN, rn it's handled by
        # extrapolation default of cubic spline ('not-a-knot')
        hazard_df = pd.read_csv(f"./USGS-data/Sa1_{bridge_id}.csv")
        
        hazard_func = user_hazard_likelihood(
            hazard_df,
            im_key='Ground Motion (g)',
            like_key='Annual Frequency of Exceedence',
            like_type='exceedence',
            method='cubic_spline', space='log',
        )
        
        # import fragility parameters and create fragility models
        fragility_df = pd.read_csv(f"./{result_folder}/{bridge_id}/hazus-fragility.csv")
        m1, m2, m3, m4 = fragility_df[['Slight', 'Moderate', 'Extensive', 'Complete']].values.ravel()
        d = fragility_df['Dispersion'].values[0]

        fragility1_func = fragility_curve(m1, d)
        fragility2_func = fragility_curve(m2, d)
        fragility3_func = fragility_curve(m3, d)
        fragility4_func = fragility_curve(m4, d)

        # generate scenarios and estimate risk
        im_lb, im_ub = hazard_df['Ground Motion (g)'].min(), hazard_df['Ground Motion (g)'].max()

        ims, scalers, likes, cqs, val, err = generate_primary_event(im_lb, im_ub,
            likelihood_func=hazard_func, kw_likelihood={}, vec_likelihood=True,
            consequence_func=expected_consequence,
            kw_consequence={
                'fragility_funcs': [fragility1_func, fragility2_func,
                                    fragility3_func, fragility4_func],
                'cq_array': cq_array
            },
            vec_consequence=True,
            rtol=0.05, min_order=nsc_min, max_order=nsc_max)

        print(f'The estimated risk = {val}')

        # save scnarios
        os.makedirs(f"./{result_folder}/{bridge_id}", exist_ok=True)
        filepath = os.path.join(f"./{result_folder}/{bridge_id}", "scenarios.csv")
        export_primary_scenarios(
            ims, scalers, likes, cqs,
            hazard_curve=hazard_func,
            vec_func=True, filepath=filepath
        )
        print(f"Bridge {bridge_id} scenarios saved")

        # append to risk_df
        risk_df.loc[index, 'Structure Number'] = bridge_id
        risk_df.loc[index, 'Deck Area (sq. ft.)'] = deck_area
        risk_df.loc[index, 'Risk (sq. ft.)'] = val
        risk_df.loc[index, 'Risk (ratio)'] = val/deck_area

    # save risk_df
    risk_df.to_csv("./{result_folder}/risk-summary.csv", index=False)

# %%
