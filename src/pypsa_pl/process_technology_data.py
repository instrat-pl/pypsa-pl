import numpy as np
from pypsa_pl.config import data_dir
from pypsa_pl.io import read_excel


def get_technology_year(investment_year):
    return 5 * (np.ceil(investment_year / 5) - 1)


def load_technology_data(source, years):
    file = data_dir("input", f"technology_data;source={source}.xlsx")
    df = read_excel(file, sheet_var="technology")
    df = df[["parameter", "technology", *years]]
    df = df.melt(
        id_vars=["parameter", "technology"],
        var_name="year",
        value_name="value",
    )
    df["year"] = df["year"].astype(int)
    df = df.pivot(
        index=["year", "technology"],
        columns="parameter",
        values="value",
    ).reset_index()
    return df
