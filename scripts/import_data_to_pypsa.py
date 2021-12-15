import pandas as pd
from prepare_input_files import scenario_name, reference_year

dataPath = f'pypsa/{scenario_name}/data/{reference_year}/'

# buses
buses = pd.read_csv(dataPath + 'buses.csv', index_col='name').dropna(axis=1, how='all').dropna()
cbf_buses = pd.read_csv(dataPath + 'cbf_buses.csv', index_col='name').dropna(axis=1, how='all').dropna()

# lines
lines = pd.read_csv(dataPath + 'lines.csv', index_col='name').dropna(axis=1, how='all').dropna()

# cross border flow
cbf_links_in = pd.read_csv(dataPath + 'cbf_links_in.csv', index_col='name').dropna(axis=1, how='all').dropna()
cbf_links_out = pd.read_csv(dataPath + 'cbf_links_out.csv', index_col='name').dropna(axis=1, how='all').dropna()

# Load
load = pd.read_csv(dataPath + 'load.csv', parse_dates=['t'], index_col='t').dropna(axis=1, how='all').dropna()

# Thermal units
gen_jwcd = pd.read_csv(dataPath + 'installed_units_jwcd.csv', index_col='name').dropna(axis=1, how='all').dropna()
gen_njwcd = pd.read_csv(dataPath + 'installed_units_njwcd.csv', index_col='name').dropna(axis=1, how='all').dropna()
gen_chp = pd.read_csv(dataPath + 'installed_units_chp.csv', index_col='name').dropna(axis=1, how='all').dropna()
gen_res_chp = pd.read_csv(dataPath + 'capacity_factors_chp.csv', parse_dates=['t'], index_col='t').dropna(axis=1, how='all').dropna()

# HPS
gen_hps = pd.read_csv(dataPath + 'installed_units_hps.csv', index_col='name').dropna(axis=1, how='all').dropna()

# PV
gen_pv_ground = pd.read_csv(dataPath + 'installed_capacity_pv_ground.csv', index_col='name').dropna(axis=1, how='all').dropna()
gen_pv_rooftop = pd.read_csv(dataPath + 'installed_capacity_pv_rooftop.csv', index_col='name').dropna(axis=1, how='all').dropna()
gen_res_pv = pd.read_csv(dataPath + 'capacity_factors_pv.csv', parse_dates=['t'], index_col='t').dropna(axis=1, how='all').dropna()

# Wind
gen_wind_onshore = pd.read_csv(dataPath + 'installed_capacity_wind_onshore.csv', index_col='name').dropna(axis=1, how='all').dropna()
gen_wind_onshore_new = pd.read_csv(dataPath + 'installed_capacity_wind_onshore_new.csv', index_col='name').dropna(axis=1, how='all').dropna()
gen_res_wind_onshore = pd.read_csv(dataPath + 'capacity_factors_wind_onshore.csv', parse_dates=['t'], index_col='t').dropna(axis=1, how='all').dropna()
gen_res_wind_onshore_new = pd.read_csv(dataPath + 'capacity_factors_wind_onshore_new.csv', parse_dates=['t'], index_col='t').dropna(axis=1, how='all').dropna()

# Wind offshore
gen_wind_offshore = pd.read_csv(dataPath + 'installed_units_wind_off.csv', index_col='name').dropna(axis=1, how='all').dropna()
gen_res_wind_offshore = pd.read_csv(dataPath + 'capacity_factors_wind_offshore.csv', parse_dates=['t'], index_col='t').dropna(axis=1, how='all').dropna()

# DSR
gen_dsr = pd.read_csv(dataPath + 'installed_capacity_dsr.csv', index_col='name').dropna(axis=1, how='all').dropna()

# IND HC
gen_ind_hc = pd.read_csv(dataPath + 'installed_capacity_industrial_hard_coal.csv', index_col='name').dropna(axis=1, how='all').dropna()

# IND GAS
gen_ind_gas = pd.read_csv(dataPath + 'installed_capacity_industrial_gas.csv', index_col='name').dropna(axis=1, how='all').dropna()

# Biogas
gen_bg = pd.read_csv(dataPath + 'installed_capacity_biogas.csv', index_col='name').dropna(axis=1, how='all').dropna()

# Biomass
gen_bm = pd.read_csv(dataPath + 'installed_capacity_biomass.csv', index_col='name').dropna(axis=1, how='all').dropna()

# Geothermal
gen_geo = pd.read_csv(dataPath + 'installed_capacity_geothermal.csv', index_col='name').dropna(axis=1, how='all').dropna()

# ROR
gen_ror = pd.read_csv(dataPath + 'installed_capacity_hydro.csv', index_col='name').dropna(axis=1, how='all').dropna()

# Batteries
gen_storage = pd.read_csv(dataPath + 'installed_units_batteries.csv', index_col='name').dropna(axis=1, how='all').dropna()
gen_batteries_small = pd.read_csv(dataPath + 'installed_capacity_batteries_small.csv', index_col='name').dropna(axis=1, how='all').dropna()
