import pandas as pd
import os

def create_cbf(year, df_buses, df_cbf, save_directory, scenario_name):
    create_cbf_buses_and_links(year, df_buses, df_cbf, save_directory, scenario_name)

def create_cbf_buses_and_links(year, df_buses, df_cbf, save_directory, scenario_name):
    cbf_dir = os.getcwd() + f'/inputs/{scenario_name}/cbf_links.xlsx'

    # create DataFrame which divides time into quarters, months and hours
    buses = df_buses['name'].to_list()

    # create links but only for buses that exist
    df_links = pd.read_excel(cbf_dir, sheet_name=str(year))
    df_links['bus0'] = df_links['bus0'].astype(str)
    df_links['bus1'] = df_links['bus1'].astype(str)
    df_links = df_links[(df_links['bus0'].isin(buses)) & (df_links['bus1'].isin(buses))]
    df_links_in = df_links[df_links['name'].str.contains('IN')]
    df_links_out = df_links[df_links['name'].str.contains('OUT')]

    df_cbf.to_csv(f'{save_directory}/cbf_buses.csv', index=False)
    df_links_in.to_csv(f'{save_directory}/cbf_links_in.csv', index=False)
    df_links_out.to_csv(f'{save_directory}/cbf_links_out.csv', index=False)