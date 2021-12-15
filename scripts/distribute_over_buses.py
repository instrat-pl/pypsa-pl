import pandas as pd
from geopy import distance
import os

def assign_bus_to_lat_lon(lat, lon, cords):
    # assign the generator to the nearest bus
    temp_cords = cords.copy()
    for key in temp_cords.keys():
        temp_cords[key] = distance.distance(temp_cords[key], (lat, lon)).km
    return min(temp_cords, key=temp_cords.get)


def calculate_srmc(df_original, year, scenario_name):
    df = df_original.copy()
    year = int(year)

    scenario_assumptions_path = os.getcwd() + f'/inputs/{scenario_name}/scenario.xlsx'

    df_prices = pd.read_excel(scenario_assumptions_path, sheet_name='Prices', index_col=0)
    df_transport = pd.read_excel(scenario_assumptions_path, sheet_name='Coal transport', index_col=0)
    transport_prices = df_transport[year]
    df_constants = pd.read_excel(scenario_assumptions_path, sheet_name='Constants', index_col=0)

    GJtot = df_constants.loc['GJtot']['Value']
    EURtoPLN = df_constants.loc['EURtoPLN']['Value']
    GJtoMWh = df_constants.loc['GJtoMWh']['Value']

    co2_price = EURtoPLN*df_prices.loc['CO2 price [EUR/t]', year]

    variable_gas = df_prices.loc['Variable Gas [PLN/MWh]', year]
    variable_lignite = df_prices.loc['Variable Lignite [PLN/MWh]', year]
    variable_hard_coal = df_prices.loc['Variable Hard coal [PLN/MWh]', year]

    gas_price = GJtoMWh*df_prices.loc['Gas price [PLN/GJ]', year]
    hydrogen_price = GJtoMWh*df_prices.loc['Hydrogen price [PLN/GJ]', year]
    lignite_bel_price = df_prices.loc['Lignite Belchatow [PLN/t]', year]
    lignite_tur_price = df_prices.loc['Lignite Turow [PLN/t]', year]
    lignite_zep_price = df_prices.loc['Lignite ZEPAK [PLN/t]', year]
    hard_coal_price = GJtot*df_prices.loc['Hard coal price [PLN/GJ]', year]

    df_coal = df[df['Fuel'].isin(['Lignite', 'Hard coal'])]
    df_gas = df[df['Fuel'].isin(['Gas'])]
    df_hydrogen = df[df['Fuel'].isin(['Hydrogen'])]
    df_nuclear = df[df['Fuel'].isin(['Nuclear'])]
    df = df[~df['Fuel'].isin(['Lignite', 'Hard coal', 'Gas', 'Hydrogen', 'Nuclear'])]

    df_coal['Transport cost [PLN/t]'] = df_coal['Transport distance'].apply(lambda x: transport_prices[x] if x is not None else x)
    df_coal['Transport cost [PLN/MWh]'] = df_coal['Transport cost [PLN/t]'] / (df_coal['Fuel calorific value [kJ/kg]'] * df_coal['Net efficiency']) * 3600
    df_coal['CO2 cost [PLN/MWh]'] = df_coal['t CO2 / 1 MWh'] * co2_price
    df_coal['fuel cost [PLN/t]'] = df_coal.apply(lambda x: lignite_bel_price if "Belchatow" in x['Unit name'] else (lignite_tur_price if "Turow" in x['Unit name'] else (lignite_zep_price if x['Fuel'] == 'Lignite' else (hard_coal_price if x['Fuel'] == 'Hard coal' else None))), axis=1)
    df_coal['fuel cost [PLN/MWh]'] = df_coal['fuel cost [PLN/t]'] / (df_coal['Fuel calorific value [kJ/kg]'] * df_coal['Net efficiency']) * 3600
    df_coal['variabe cost [PLN/MWh]'] = df_coal['Fuel'].apply(lambda x: variable_lignite if x == 'Lignite' else (variable_hard_coal if x == 'Hard coal' else None))
    df_coal['SRMC'] = df_coal['Transport cost [PLN/MWh]'] + df_coal['CO2 cost [PLN/MWh]'] + df_coal['fuel cost [PLN/MWh]'] + df_coal['variabe cost [PLN/MWh]']

    df_gas['CO2 cost [PLN/MWh]'] = df_gas['t CO2 / 1 MWh'] * co2_price
    df_gas['SRMC'] = df_gas['CO2 cost [PLN/MWh]'] + gas_price/df_gas['Net efficiency'] + variable_gas

    df_hydrogen['CO2 cost [PLN/MWh]'] = 0
    df_hydrogen['SRMC'] = df_hydrogen['CO2 cost [PLN/MWh]'] + hydrogen_price/df_hydrogen['Net efficiency'] + variable_gas

    df_nuclear['CO2 cost [PLN/MWh]'] = 0
    df_nuclear['SRMC'] = 40 # PEP2040 used NREL ATB 2019 (discountinued) with SRMC around 50 PLN/MWh. Latest (2021) ATB NREL data: https://atb.nrel.gov/electricity/2021/other_technologies_(eia). VOM = 2.352 $/MWh, Fuel = 7.093 $/MWh.

    df = pd.concat([df_coal, df_gas, df_hydrogen, df_nuclear, df], sort=False)
    df['SRMC'] = df['SRMC'].fillna(0)
    return df


def add_nuclear(df_original, year, power_plants_excel_dir, plant_type):
    df = df_original.copy()
    year = int(year)
    df_nuclear = pd.read_excel(power_plants_excel_dir, sheet_name='Planned thermal')

    df_nuclear = df_nuclear[df_nuclear['Commission year'] <= year]
    df_nuclear = df_nuclear[df_nuclear['Type'] == plant_type]
    df = pd.concat([df, df_nuclear], sort=False)
    return df


def add_planned_gas(df_original, year, power_plants_excel_dir, plant_type):
    df = df_original.copy()
    year = int(year)
    df_gas = pd.read_excel(power_plants_excel_dir, sheet_name='Planned thermal')

    df_gas = df_gas[df_gas['Commission year'] <= year]
    df_gas = df_gas[df_gas['Type'] == plant_type]
    df = pd.concat([df, df_gas], sort=False)
    return df


def decomission_schedule(df_original, year):
    df = df_original.copy()
    year = int(year)

    df = df[df[f'Planned decommission date'] >= year]

    return df


def create_generators_from_given_units(df_original, coords):
    df = df_original.copy()
    df = df[df['Include in model [YES/NO]'] == 'YES']
    df['bus'] = df.apply(lambda x: assign_bus_to_lat_lon(x['Latitude'], x['Longitude'], coords), axis=1)
    df = df.rename(columns={'Unit name': 'name', 'Fuel': 'carrier', 'Installed capacity gross [MW]': 'p_nom',
                            'Net efficiency': 'efficiency', 'SRMC': 'marginal_cost',
                            'Storage capacity gross [MWh]': 'storage_capacity_mwh'})
    return df


def distribute_units(year, df_scenario, df_buses, save_directory, scenario_name):
    power_plants_excel_dir = os.getcwd() + f'/inputs/{scenario_name}/utility_units.xlsx'
    df_hps = pd.read_excel(power_plants_excel_dir, sheet_name='HPS')
    df_chp = pd.read_excel(power_plants_excel_dir, sheet_name='CHP')
    df_jwcd = pd.read_excel(power_plants_excel_dir, sheet_name='JWCD')
    df_njwcd = pd.read_excel(power_plants_excel_dir, sheet_name='nJWCD')
    df_wind_off = pd.read_excel(power_plants_excel_dir, sheet_name='Wind offshore')
    df_batteries = pd.read_excel(power_plants_excel_dir, sheet_name='Batteries')

    cords = {name: (df_buses.loc[i, 'y'], df_buses.loc[i, 'x']) for i, name in enumerate(df_buses['name'])}

    # JWCD units
    df_jwcd = add_nuclear(df_jwcd, year, power_plants_excel_dir, plant_type='EPR')
    df_jwcd = add_planned_gas(df_jwcd, year, power_plants_excel_dir, plant_type='CCGT')
    df_jwcd = add_planned_gas(df_jwcd, year, power_plants_excel_dir, plant_type='OCGT')
    df_jwcd = decomission_schedule(df_jwcd, year)
    df_jwcd = calculate_srmc(df_jwcd, year, scenario_name)
    df_jwcd = create_generators_from_given_units(df_jwcd, cords)

    # CHP units
    df_chp = add_planned_gas(df_chp, year, power_plants_excel_dir, plant_type='CHP')
    df_chp = decomission_schedule(df_chp, year)
    df_chp = calculate_srmc(df_chp, year, scenario_name)
    df_chp = create_generators_from_given_units(df_chp, cords)

    # nJWCD units
    df_njwcd = decomission_schedule(df_njwcd, year)
    df_njwcd = create_generators_from_given_units(df_njwcd, cords)

    # HPS units
    df_hps = create_generators_from_given_units(df_hps, cords)
    df_hps = df_hps[df_hps['Commission year'] <= int(year)]
    df_hps['max_hours'] = df_hps['storage_capacity_mwh'] / df_hps['p_nom']
    df_hps['p_max_pu'] = 1
    df_hps['efficiency_dispatch'] = 0.75
    df_hps['standing_loss'] = 0.01

    # Wind offshore units
    df_wind_off = create_generators_from_given_units(df_wind_off, cords)
    df_wind_off = df_wind_off[df_wind_off['Commission year'] <= int(year)]

    # Batteries units
    df_batteries = create_generators_from_given_units(df_batteries, cords)
    df_batteries = df_batteries[df_batteries['Commission year'] <= int(year)]
    df_batteries['max_hours'] = df_batteries['storage_capacity_mwh'] / df_batteries['p_nom']
    df_batteries['p_max_pu'] = 1
    df_batteries['efficiency_dispatch'] = 0.9
    df_batteries['standing_loss'] = 0.01

    # Choose only wanted columns
    df_hps = df_hps[['bus', 'name', 'carrier', 'p_nom', 'max_hours', 'p_max_pu',
                                 'efficiency_dispatch', 'standing_loss']]
    df_chp = df_chp[['bus', 'name', 'carrier', 'p_nom', 'efficiency']]
    df_jwcd = df_jwcd[['bus', 'name', 'carrier', 'p_nom', 'efficiency', 'marginal_cost']]
    df_njwcd = df_njwcd[['bus', 'name', 'carrier', 'p_nom', 'efficiency']]
    df_wind_off = df_wind_off[['bus', 'name', 'carrier', 'p_nom']]
    df_batteries = df_batteries[['bus', 'name', 'carrier', 'p_nom', 'max_hours', 'p_max_pu',
                                 'efficiency_dispatch', 'standing_loss']]

    df_hps.to_csv(f'{save_directory}/installed_units_hps.csv', index=False)
    df_chp.to_csv(f'{save_directory}/installed_units_chp.csv', index=False)
    df_jwcd.to_csv(f'{save_directory}/installed_units_jwcd.csv', index=False)
    df_njwcd.to_csv(f'{save_directory}/installed_units_njwcd.csv', index=False)
    df_wind_off.to_csv(f'{save_directory}/installed_units_wind_off.csv', index=False)
    df_batteries.to_csv(f'{save_directory}/installed_units_batteries.csv', index=False)

    return df_chp, df_wind_off


def distribute_load(year, points_in_vois, df_scenario, save_directory):
    # Remove Feb 29 from leap years
    year = int(year)
    if year % 4 == 0:
        whole_year = pd.date_range(start=f'{year}-01-01', end=f'{year + 1}-01-01', closed='left', freq='h')
        feb_29 = pd.date_range(start=f'{year}-02-29', end=f'{year}-03-01', closed='left', freq='h')
        index_date = [i for i in whole_year if i not in feb_29]
    else:
        index_date = pd.date_range(start=f'{year}-01-01', end=f'{year + 1}-01-01', closed='left', freq='h')

    # Convert yearly demand into hourly values
    load_twh = df_scenario.loc['Electricity load [TWh]']
    df_load_hourly = (load_twh/0.91) * 1000000 * pd.read_excel('inputs/load_hour_factors_2019.xlsx', index_col='t')

    # Spread load over every voivodeship
    df_load_voivodeships = pd.read_excel('inputs/load_decomposition_over_voivodeships_2019.xlsx',
                                         index_col='Bus').T
    df = pd.concat([df_load_hourly, df_load_voivodeships], sort=True).fillna(method='backfill').drop('Fraction')
    for col in df.drop('profile', axis=1).columns:
        df[col] = df[col] * df['profile']
    df = df.drop('profile', axis=1)
    df.index = index_date
    df.index.name = 't'

    # Spread load over every bus in voivodeship - to be improved using different criteria for load distribution
    df_output = pd.DataFrame(index=df.index)
    for col in df.columns:
        for key in points_in_vois[col]:
            df_output[key] = df[col] / len(points_in_vois[col])

    df_output.to_csv(f'{save_directory}/load.csv')


def distribute_installed_capacity(points_in_vois, df_capacity_distribution, df_scenario, df_buses, save_directory):
    # Distribute installed capacity across voivodeships
    fuels = df_scenario.index.to_list()
    df = df_capacity_distribution.copy()
    for col in df.columns:
        for fuel in fuels:
            if fuel.split(' [')[0] == col:
                df[col] = df[col] * df_scenario.loc[fuel]

    df_temp = df_buses['name'].copy().to_frame().rename(columns={'name': 'bus'}).set_index('bus')

    # PV - ground
    df_pv_ground = df_temp.copy()
    df_pv_ground['carrier'] = 'PV'
    df_pv_ground['name'] = df_pv_ground.index + '_PV_new_gen'

    # PV - rooftop
    df_pv_rooftop = df_temp.copy()
    df_pv_rooftop['carrier'] = 'PV'
    df_pv_rooftop['name'] = df_pv_rooftop.index + '_PV_rooftop_gen'

    # Batteries - household
    df_batteries_small = df_temp.copy()
    df_batteries_small['carrier'] = 'Battery'
    df_batteries_small['name'] = df_batteries_small.index + '_Batteries_small_gen'
    df_batteries_small['max_hours'] = 2
    df_batteries_small['p_max_pu'] = 1
    df_batteries_small['efficiency_dispatch'] = 0.9
    df_batteries_small['standing_loss'] = 0.01

    # Wind onshore
    df_wind_onshore = df_temp.copy()
    df_wind_onshore['carrier'] = 'Wind'
    df_wind_onshore['name'] = df_wind_onshore.index + '_Wind_onshore_gen'

    # Wind onshore
    df_wind_onshore_new = df_temp.copy()
    df_wind_onshore_new['carrier'] = 'Wind'
    df_wind_onshore_new['name'] = df_wind_onshore_new.index + '_Wind_onshore_new_gen'

    # Geothermal
    df_geothermal = df_temp.copy()
    df_geothermal['carrier'] = 'Geothermal'
    df_geothermal['p_max_pu'] = 1
    df_geothermal['name'] = df_geothermal.index + '_Geothermal_gen'

    # Biomass
    df_biomass = df_temp.copy()
    df_biomass['carrier'] = 'Biomass'
    df_biomass['p_max_pu'] = 1
    df_biomass['efficiency'] = 0.30
    df_biomass['name'] = df_biomass.index + '_Biomass_gen'

    # Biogas
    df_biogas = df_temp.copy()
    df_biogas['carrier'] = 'Biogas'
    df_biogas['p_max_pu'] = 1
    df_biogas['efficiency'] = 0.36
    df_biogas['name'] = df_biogas.index + '_Biogas_gen'

    # Water
    df_hydro = df_temp.copy()
    df_hydro['carrier'] = 'Hydro'
    df_hydro['p_max_pu'] = 0.35
    df_hydro['name'] = df_hydro.index + '_Hydro_gen'

    # Industrial gas
    df_ind_gas = df_temp.copy()
    df_ind_gas['carrier'] = 'Gas'
    df_ind_gas['p_max_pu'] = 1
    df_ind_gas['efficiency'] = 0.51
    df_ind_gas['name'] = df_ind_gas.index + '_Gas_ind_gen'

    # Industrial coal
    df_ind_coal = df_temp.copy()
    df_ind_coal['carrier'] = 'Hard coal'
    df_ind_coal['p_max_pu'] = 0.9
    df_ind_coal['efficiency'] = 0.37
    df_ind_coal['name'] = df_ind_coal.index + '_Hard_coal_ind_gen'

    # DSR
    df_dsr = df_temp.copy()
    df_dsr['carrier'] = 'DSR'
    df_dsr['name'] = df_dsr.index + '_DSR'

    # Spread installed capacity over every bus in voivodeship equaly - to be improved similarly to load
    for key in points_in_vois:
        df.loc[key] = df.loc[key] / len(points_in_vois[key])
        for key2 in points_in_vois[key]:
            df_pv_ground.loc[key2, 'p_nom'] = df.loc[key]['PV - ground']
            df_pv_rooftop.loc[key2, 'p_nom'] = df.loc[key]['PV - rooftop']
            df_batteries_small.loc[key2, 'p_nom'] = df.loc[key]['Batteries']
            df_wind_onshore.loc[key2, 'p_nom'] = df.loc[key]['Wind onshore']
            df_wind_onshore_new.loc[key2, 'p_nom'] = df.loc[key]['Wind onshore - new']
            df_geothermal.loc[key2, 'p_nom'] = df.loc[key]['Geothermal']
            df_biomass.loc[key2, 'p_nom'] = df.loc[key]['Biomass']
            df_biogas.loc[key2, 'p_nom'] = df.loc[key]['Biogas']
            df_hydro.loc[key2, 'p_nom'] = df.loc[key]['Hydro']
            df_ind_gas.loc[key2, 'p_nom'] = df.loc[key]['Gas industrial']
            df_ind_coal.loc[key2, 'p_nom'] = df.loc[key]['Hard coal industrial']
            df_dsr.loc[key2, 'p_nom'] = df.loc[key]['DSR']

    df_pv_ground.to_csv(f'{save_directory}/installed_capacity_pv_ground.csv')
    df_pv_rooftop.to_csv(f'{save_directory}/installed_capacity_pv_rooftop.csv')
    df_batteries_small.to_csv(f'{save_directory}/installed_capacity_batteries_small.csv')
    df_wind_onshore.to_csv(f'{save_directory}/installed_capacity_wind_onshore.csv')
    df_wind_onshore_new.to_csv(f'{save_directory}/installed_capacity_wind_onshore_new.csv')
    df_geothermal.to_csv(f'{save_directory}/installed_capacity_geothermal.csv')
    df_biomass.to_csv(f'{save_directory}/installed_capacity_biomass.csv')
    df_biogas.to_csv(f'{save_directory}/installed_capacity_biogas.csv')
    df_hydro.to_csv(f'{save_directory}/installed_capacity_hydro.csv')
    df_ind_gas.to_csv(f'{save_directory}/installed_capacity_industrial_gas.csv')
    df_ind_coal.to_csv(f'{save_directory}/installed_capacity_industrial_hard_coal.csv')
    df_dsr.to_csv(f'{save_directory}/installed_capacity_dsr.csv')