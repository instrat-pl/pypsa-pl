import pandas as pd
from scripts.create_date_index import create_date_index

capacity_factors_excel_dir = 'inputs\\capacity_factors.xlsx'


def assign_capacity_factor_based_on_voivodeship(df_original, index_date, points_in_vois):
    # Each point gets capacity factor based on voivodeship it lies in
    df = df_original.copy()
    df.index = index_date
    df_output = pd.DataFrame(index=df.index)
    for col in df.columns:
        for key in points_in_vois[col]:
            df_output[key] = df[col]
    df_output.index.name = 't'
    return df_output


def pv_wind_capacity_factors(year, points_in_vois, save_directory):
    index_date = create_date_index(year)

    # Wind onshore
    df_wind = pd.read_excel(capacity_factors_excel_dir, sheet_name='wind_2015', index_col='t')
    df_output = assign_capacity_factor_based_on_voivodeship(df_wind, index_date, points_in_vois)
    df_output.to_csv(f'{save_directory}/capacity_factors_wind_onshore.csv')

    # # Wind onshore - new
    df_wind_new = pd.read_excel(capacity_factors_excel_dir, sheet_name='wind_new_2015', index_col='t')
    df_output = assign_capacity_factor_based_on_voivodeship(df_wind_new, index_date, points_in_vois)
    df_output.to_csv(f'{save_directory}/capacity_factors_wind_onshore_new.csv')

    # PV
    df_pv = pd.read_excel(capacity_factors_excel_dir, sheet_name='pv_2015', index_col='t')
    df_output = assign_capacity_factor_based_on_voivodeship(df_pv, index_date, points_in_vois)
    df_output.to_csv(f'{save_directory}/capacity_factors_pv.csv')


def assign_capacity_factor_based_on_unit(units_original, df_original, index_date):
    df = df_original.copy()
    units = units_original.copy()
    df.index = index_date
    df.index.name = 't'
    for col in units['name'].to_list():
        df[col] = df['Generic']
    df = df.drop('Generic', axis=1)
    return df


def chp_capacity_factors(year, units, save_directory):
    index_date = create_date_index(year)

    df_capacity_chp = pd.read_excel(capacity_factors_excel_dir, sheet_name='chp_2019', index_col='t')
    df_output = assign_capacity_factor_based_on_unit(units, df_capacity_chp, index_date)
    df_output.to_csv(f'{save_directory}/capacity_factors_chp.csv')


def wind_off_capacity_factors(year, units, save_directory):
    index_date = create_date_index(year)

    df_capacity_wind_off = pd.read_excel(capacity_factors_excel_dir, sheet_name='wind_off_2019', index_col='t')
    df_output = assign_capacity_factor_based_on_unit(units, df_capacity_wind_off, index_date)
    df_output.to_csv(f'{save_directory}/capacity_factors_wind_offshore.csv')
