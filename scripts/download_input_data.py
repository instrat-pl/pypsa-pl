import logging
import gspread
import pandas as pd

from pypsa_pl.config import data_dir


def gsheet_to_df(url, sheet_name):
    gc = gspread.oauth()
    gs = gc.open_by_url(url)
    logging.info(f'Downloading sheet {sheet_name} from "{gs.title}"')
    ws = gs.worksheet(sheet_name)
    df = pd.DataFrame(ws.get_all_records())
    df = df.replace("", pd.NA)
    return df


def ignore_commented_rows_columns(df):
    df = df.loc[
        ~df.iloc[:, 0].fillna(" ").str.startswith("#"), ~df.columns.str.startswith("#")
    ]
    return df


# Public google sheet URLs
urls = {
    "technology_carrier_definitions": "https://docs.google.com/spreadsheets/d/1oM4T3LirR-XGO1fQ_KhiuQXW8t3I4AKj8q0n8P0s-aE",
    "technology_cost_data": "https://docs.google.com/spreadsheets/d/1P-CGOaUUJt3J-6DfelAx5ilRSy0r2gCyJp_ZeHu1wbI",
    "installed_capacity": "https://docs.google.com/spreadsheets/d/1fwosQK76x_FoXRSI6tphexjMchXSIX0NqAfHNCDI_BA",
    "annual_energy_flows": "https://docs.google.com/spreadsheets/d/1OWm53wIPTVJf0PGUrUxhjpzfVJgyMhwdBLg5cuRzvZY",
    "capacity_utilisation": "https://docs.google.com/spreadsheets/d/1OTZmzscUlB6uxuaWvN5Et1qpixFMubnh2m4-qbZD7rk",
    "capacity_addition_potentials": "https://docs.google.com/spreadsheets/d/1z2pfJ6VwmjsgGgChJexISJZ-OYlrVllMUe1Q14Y5eR0",
}


def download_input_data():
    for name, variant in [
        # technology_carrier_definitions
        ("technology_carrier_definitions", "full"),
        ("technology_carrier_definitions", "mini"),
        # technology_cost_data
        ("technology_cost_data", "instrat_2024"),
        # installed_capacity
        ("installed_capacity", "historical_totals"),
        ("installed_capacity", "historical+instrat_projection"),
        ("installed_capacity", "neighbours"),
        # annual_energy_flows
        ("annual_energy_flows", "historical"),
        ("annual_energy_flows", "instrat_projection"),
        ("annual_energy_flows", "constraints"),
        ("annual_energy_flows", "neighbours"),
        # capacity_utilisation
        ("capacity_utilisation", "historical"),
        ("capacity_utilisation", "instrat_projection"),
        ("capacity_utilisation", "neighbours"),
        # capacity_addition_potentials
        ("capacity_addition_potentials", "instrat_projection"),
    ]:
        df = gsheet_to_df(urls[name], sheet_name=variant)
        df = ignore_commented_rows_columns(df)
        df.to_csv(data_dir("input", f"{name};variant={variant}.csv"), index=False)


if __name__ == "__main__":

    download_input_data()
