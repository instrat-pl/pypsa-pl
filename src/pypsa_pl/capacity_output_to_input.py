import pandas as pd
import numpy as np

from pypsa_pl.config import data_dir
from pypsa_pl.io import read_excel
from pypsa_pl.plot_results import read_variable


def capacity_output_to_input(
    run_name,
    source_aggregate_units,
    source_sectoral_units,
    extendable_and_decommission_technologies,
    year,
    output_capacity_name,
    fill_future=True,
    fill_future_tech_exceptions=["ICE vehicle"],
    max_year=2050,
):
    # Read the original input file for aggregate capacities (electricity sector)
    df_agg = read_excel(
        data_dir(
            "input",
            f"aggregate_units;source={source_aggregate_units}.xlsx",
        ),
        sheet_var="name",
    ).reset_index(drop=True)
    assert str(year) in df_agg.columns

    # Read the original input file for sectoral capacities (other sectors)
    df_sec = read_excel(
        data_dir(
            "input",
            f"sectoral_units;source={source_sectoral_units}.xlsx",
        ),
        sheet_var="name",
    ).reset_index(drop=True)
    assert str(year) in df_sec.columns

    # Read optimal capacities for the selected run
    output_dir = data_dir("runs", run_name, "output")
    dfs = {}
    for attr in [
        "technology",
        "category",
        "area",
        "build_year",
        "p_nom_opt",
        "p_nom_extendable",
        "e_nom_opt",
        "e_nom_extendable",
    ]:
        dfs[attr] = pd.concat(
            [
                read_variable(output_dir, components, attr)
                for components in ["generators", "storage_units", "links", "stores"]
            ]
        )
    df = pd.concat(dfs.values(), axis=1).reset_index(names=["unit"])
    df = df[
        (df["build_year"] == int(year))
        & df["technology"].isin(extendable_and_decommission_technologies)
    ]

    # Iterate over rows of the original input files and modify capacities for the selected year according to optimal ones
    # Then save to new input files with source=run_name
    for prefix, df_units in [("aggregate_units", df_agg), ("sectoral_units", df_sec)]:
        for row in df_units.itertuples():
            if row.technology not in extendable_and_decommission_technologies:
                continue

            sel = (
                (df["technology"] == row.technology)
                & (df["category"] == row.category)
                & (df["area"] == row.area)
                & df["unit"].str.contains(row.name)
            )

            if sum(sel) == 0:
                continue
            elif sum(sel) > 1:
                assert sum(sel) <= 1
            else:
                if "storage" in row.technology:
                    nom_opt = df.loc[sel, "e_nom_opt"].values[0]
                else:
                    nom_opt = df.loc[sel, "p_nom_opt"].values[0]

                df_units.loc[row.Index, str(year)] = np.round(nom_opt, 1)
                if (
                    fill_future
                    and (row.value_type == "total")
                    and row.technology not in fill_future_tech_exceptions
                ):
                    future_year_columns = [
                        str(y) for y in range(year + 5, max_year + 1, 5)
                    ]
                    df_units.loc[row.Index, future_year_columns] = np.minimum(
                        np.round(nom_opt, 1),
                        df_units.loc[row.Index, future_year_columns].fillna(0),
                    )

        with pd.ExcelWriter(
            data_dir(
                "input",
                f"{prefix};source={output_capacity_name}.xlsx",
            ),
        ) as writer:
            for name, subdf in df_units.groupby("name"):
                subdf.drop(columns=["name"]).to_excel(
                    writer, sheet_name=name, index=False
                )
