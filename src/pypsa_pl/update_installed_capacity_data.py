import pandas as pd
import numpy as np

from pypsa_pl.config import data_dir
from pypsa_pl.build_network import concat_inputs


def update_installed_capacity_data(
    network,
    variant,
    original_variant_list,
    force_original=False,
    custom_operation=None,
):
    inf = network.meta.get("inf", 999999.0)

    file = data_dir("input", f"installed_capacity;variant={variant}.csv")

    if not file.exists() or force_original:
        df = concat_inputs("installed_capacity", original_variant_list)
    else:
        df = pd.read_csv(file)

    # Get all extendable capacities from the network
    dfs = []
    for component, nom_attr in [
        ("Generator", "p_nom"),
        ("Link", "p_nom"),
        ("Store", "e_nom"),
    ]:
        df_c = (
            network.df(component)
            .reset_index(names=["name_in_network"])
            .rename(columns={f"{nom_attr}_opt": "nom_opt"})
        )
        df_c = df_c[df_c[f"{nom_attr}_extendable"]]
        df_c["cumulative"] = df_c["lifetime"] == 1
        df_c = df_c[
            [
                "name_in_network",
                "area",
                "technology",
                "qualifier",
                "build_year",
                "cumulative",
                "nom_opt",
            ]
        ]
        dfs.append(df_c)

    df_c = pd.concat(dfs)
    for attr in ["qualifier"]:
        df[attr] = df[attr].fillna("")
        df_c[attr] = df_c[attr].fillna("")

    df_c["qualifier"] = df_c["qualifier"].replace("prosumer", "")

    df = df.merge(
        df_c,
        how="left",
        on=["area", "technology", "qualifier", "build_year", "cumulative"],
    )

    df["nom"] = df["nom_opt"].abs().round(0).fillna(df["nom"]).replace(inf, np.inf)
    df["name"] = df["name"].fillna(df["name_in_network"])

    for attr in ["qualifier"]:
        df[attr] = df[attr].replace("", pd.NA)

    df = df.drop(columns=["name_in_network", "nom_opt"])

    df["retire_year"] = df["retire_year"].astype("Int64")
    df["cumulative"] = df["cumulative"].astype("str").str.upper()

    if custom_operation is not None:
        df = custom_operation(df, network.meta)

    return df
