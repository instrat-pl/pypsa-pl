import pandas as pd
import sys
import os
from scripts.lines_buses import assign_buses_to_voivodeships, create_lines_between_buses
from scripts.distribute_over_buses import distribute_installed_capacity, distribute_load, distribute_units
from scripts.capacity_factors import pv_wind_capacity_factors, chp_capacity_factors, wind_off_capacity_factors
from scripts.cross_border_flow import create_cbf


scenario_name = 'instrat'
reference_year = '2040'

sys.path.append(os.getcwd() + '/scripts')

# Create new file with scenario assumptions
scenario_path = os.getcwd() + f'/pypsa/{scenario_name}'
if not os.path.exists(scenario_path):
    os.makedirs(scenario_path)
    os.makedirs(scenario_path + '/data')

reference_year_path = scenario_path + '/data' + f'/{reference_year}'
if not os.path.exists(reference_year_path):
    os.makedirs(reference_year_path)

# Load base files
scenario_assumptions_path = os.getcwd() + f'/inputs/{scenario_name}/scenario.xlsx'
df_scenario = pd.read_excel(scenario_assumptions_path, index_col=0)
df_scenario = df_scenario[int(reference_year)]

df_capacity_distribution = pd.read_excel('inputs/installed_capacity_distribution.xlsx', index_col=0)
df_capacity_distribution = df_capacity_distribution.fillna(0)

# Get names and coordinates of buses distributed along voivodeships
df_buses = pd.read_excel('inputs/buses.xlsx')
df_cbf = pd.read_excel('inputs/buses.xlsx', sheet_name='cbf')
df_buses = pd.concat([df_cbf, df_buses], sort=False).reset_index(drop=True)
df_buses['name'] = df_buses['name'].astype(str)
buses_in_voivodeships = assign_buses_to_voivodeships(df_buses)

if __name__ == "__main__":
    print('Creating buses and connections...')
    create_lines_between_buses(reference_year, df_buses, reference_year_path)

    print('Creating CBF buses and links...')
    create_cbf(reference_year, df_buses, df_cbf, reference_year_path, scenario_name)

    print('Creating PV & Wind capacity factors...')
    pv_wind_capacity_factors(reference_year, buses_in_voivodeships, reference_year_path)

    print('Creating RES & industrial generators...')
    distribute_installed_capacity(buses_in_voivodeships, df_capacity_distribution, df_scenario, df_buses, reference_year_path)

    print('Creating JWCD, nJWCD, CHP and HPS generators...')
    chp_units, wind_off_units = distribute_units(reference_year, df_scenario, df_buses, reference_year_path, scenario_name)
    chp_capacity_factors(reference_year, chp_units, reference_year_path)
    wind_off_capacity_factors(reference_year, wind_off_units, reference_year_path)

    print('Creating load...')
    distribute_load(reference_year, buses_in_voivodeships, df_scenario, reference_year_path)

    print('All done :>')
