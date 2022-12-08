# constants used in the examples

# hazard curve v_IM (im): use Section 2.3 of Baker: Introduction to PSHA
RATE, M_HAZARD, R_HAZARD, SIGMA_HAZARD = 0.01, 6.5, 10, 0.57    # R in km

# fragility curve (pristine): Use example 1 (Table 11) in HAZUS report (Basoz and Mander 1999)
PGA_S, PGA_M, PGA_E, PGA_C = 0.30, 0.36, 0.49, 0.71
PGA_SIGMA = 0.6

# damage paramters: damage ratios are based on Table 22 in HAZUS report (Basoz and Mander 1999)
NDS = 4    # 4 damage states
CQ_S, CQ_M, CQ_E, CQ_C = 0.12, 0.19, 0.48, 1.00

# Liquefaction paramters: based on HAZUS v4.2 (Table 7-6)
PGD_S, PGD_M, PGD_E, PGD_C = 3.9, 3.9, 3.9, 13.8    # in inches, and only E and C damage states
PGD_SIGMA = 0.2
LQ_CLASS = 5
WATER_DEPTH = 0.0