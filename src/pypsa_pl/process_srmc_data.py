import pandas as pd
import numpy as np

from pypsa_pl.config import data_dir
from pypsa_pl.io import read_excel
from pypsa_pl.process_technology_data import load_technology_data, get_technology_year


def process_srmc_data(
    df,
    years,
    source_prices,
    source_technology,
    srmc_wind=None,
    srmc_pv=None,
    srmc_dsr=None,
    srmc_only_JWCD=False,
    random_seed=None,
):
    # Read prices
    file = data_dir("input", f"prices;source={source_prices}.xlsx")
    df_prices = read_excel(file, sheet_name="prices")
    # assert not df_prices.isna().any().any()
    df_prices = df_prices[["Price Type", *years]]
    df_prices = df_prices.set_index("Price Type").transpose().reset_index(names="year")
    # Convert price units from PLN/GJ to PLN/MWh_t
    for col in df_prices.columns:
        if "PLN/GJ" in col:
            df_prices[col.replace("PLN/GJ", "PLN/MWh_t")] = 3.6 * df_prices[col]
    # Cross join with generation units dataframe
    df = pd.merge(df, df_prices, how="cross")

    if (
        source_prices == "pypsa_pl_v1"
        and "Fuel transport distance (round) [km]" in df.columns
    ):
        assert sum(year % 5 == 0 for year in years) == len(years)
        # Read transport cost table and merge
        file = data_dir("input", f"transport_prices;source={source_prices}.xlsx")
        df_transport = read_excel(file, sheet_name="coal_transport")
        df_transport = df_transport[["Transport distance [km]", *years]]
        df_transport = df_transport.melt(
            id_vars="Transport distance [km]",
            var_name="year",
            value_name="Fuel transport cost [PLN/t]",
        )
        df = pd.merge(
            df,
            df_transport,
            left_on=["Fuel transport distance (round) [km]", "year"],
            right_on=["Transport distance [km]", "year"],
            how="left",
        )
        df["Fuel transport cost [PLN/MWh_t]"] = (
            df["Fuel transport cost [PLN/t]"]
            / df["Fuel calorific value [MWh_t/t or MWh_t/1000m3 for gas]"]
        )

    # Read technological assumptions and merge (but only when data missing)
    df["technology_year"] = get_technology_year(df["year"]).astype(int)
    technology_years = df["technology_year"].drop_duplicates().tolist()
    df_tech = load_technology_data(source_technology, years=technology_years)
    param_cols = [
        "Fuel CO2 emission factor [tCO2/MWh_t]",
        "Variable cost [PLN/MWh_e]",
        "Fuel transport cost [PLN/MWh_t]",
    ]
    df_tech = df_tech[["year", "technology", *param_cols]]
    df_tech = df_tech.rename(columns={"year": "technology_year"})

    df = (
        df.set_index("name")
        .combine_first(
            pd.merge(
                df.drop(columns=[col for col in param_cols if col in df.columns]),
                df_tech,
                on=["technology_year", "technology"],
                how="left",
            ).set_index("name"),
        )
        .reset_index()
    )

    # Calculate generation costs for each unit
    df["Net electrical efficiency"] = df["efficiency"]

    # CO2 cost
    df["CO2 cost [PLN/MWh_e]"] = (
        df["CO2 price [PLN/tCO2]"]
        * df["Fuel CO2 emission factor [tCO2/MWh_t]"]
        / df["Net electrical efficiency"]
    )

    # Fuel cost
    for fuel in [
        "Natural gas",
        "Hard coal",
        "Lignite",
        "Biomass straw",
        "Biomass wood chips",
        "Biogas",
        "Hydrogen",
        "Oil",
    ]:
        if fuel == "Lignite":
            is_lignite = df["carrier"] == "Lignite"
            df.loc[is_lignite, "Fuel cost [PLN/MWh_t]"] = df.loc[
                is_lignite, "Lignite others price [PLN/MWh_t]"
            ]
            for lignite_location in ["Belchatow", "Turow"]:
                is_location = is_lignite & df["name"].str.contains(lignite_location)
                df.loc[is_location, "Fuel cost [PLN/MWh_t]"] = df.loc[
                    is_location, f"Lignite {lignite_location} price [PLN/MWh_t]"
                ]
        elif fuel == "Biogas":
            is_biogas = df["carrier"] == "Biogas"
            df.loc[is_biogas, "Fuel cost [PLN/MWh_t]"] = df.loc[
                is_biogas, f"Biogas substrate price [PLN/MWh_t]"
            ]
        else:
            is_fuel = df["carrier"] == fuel
            df.loc[is_fuel, "Fuel cost [PLN/MWh_t]"] = df.loc[
                is_fuel, f"{fuel} price [PLN/MWh_t]"
            ]

    df["Fuel cost [PLN/MWh_e]"] = (
        df["Fuel cost [PLN/MWh_t]"] / df["Net electrical efficiency"]
    )

    df["Fuel transport cost [PLN/MWh_e]"] = (
        df["Fuel transport cost [PLN/MWh_t]"] / df["Net electrical efficiency"]
    )

    # Total short run marginal cost
    df["SRMC [PLN/MWh_e]"] = df[
        [
            "CO2 cost [PLN/MWh_e]",
            "Fuel cost [PLN/MWh_e]",
            "Fuel transport cost [PLN/MWh_e]",
            "Variable cost [PLN/MWh_e]",
        ]
    ].sum(axis=1)

    # Ignore SRMC for non-JWCD units (PyPSA-PL v1 assumption)
    if srmc_only_JWCD:
        df.loc[df["category"] != "JWCD", "SRMC [PLN/MWh_e]"] = 0

    # Override final SRMC values
    assert srmc_dsr > 0
    for prefix, srmc in [("Wind", srmc_wind), ("PV", srmc_pv), ("DSR", srmc_dsr)]:
        if srmc is not None:
            df.loc[df["category"].str.startswith(prefix), "SRMC [PLN/MWh_e]"] = srmc

    # Keep only non-zero values if no noise is added
    if random_seed is None:
        df = df[df["SRMC [PLN/MWh_e]"] > 0]
    # Otherwise set minimum SRMC to 1
    else:
        df.loc[df["SRMC [PLN/MWh_e]"] < 1, "SRMC [PLN/MWh_e]"] = 1

    df["year"] = df["year"].astype(int)
    df = df.pivot(index="year", columns="name", values="SRMC [PLN/MWh_e]")
    assert df.notna().all().all()

    if not random_seed is None:
        # Add random noise from [-1, 1] to distinguish units with the same SRMC
        rng = np.random.default_rng(seed=random_seed)
        noise = pd.Series(
            rng.choice(
                np.linspace(-1, 1, len(df.columns)),
                size=len(df.columns),
                replace=False,
            ),
            index=df.columns,
        )
        df += noise

    df = df.round(3).reset_index()
    return df
