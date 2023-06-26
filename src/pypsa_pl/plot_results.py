import logging
import os
import pandas as pd
import numpy as np
import pypsa

import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio

from pypsa_pl.colors import technology_colors
from pypsa_pl.config import data_dir
from pypsa_pl.io import product_dict, dict_to_str


def runs_dir(*path):
    return data_dir("runs", *path)


def output_dir(*path):
    return data_dir("output", *path)


def make_pypsa_pl_template():
    template = go.layout.Template()
    template.layout.title.font = dict(
        family="Work Sans Medium, sans-serif", size=14, color="black"
    )
    template.layout.font = dict(
        family="Work Sans Light, sans-serif", size=10, color="black"
    )
    template.layout.colorway = [
        "#c1843d",
        "#535ce3",
        "#d9d9d9",
        "#212121",
    ]
    return template


def plot_bars(
    df,
    x_var,
    y_var,
    cat_var,
    cat_vals,
    text_var=None,
    colors=None,
    hover_vars=None,
    x_range=None,
    y_range=None,
    x_label=None,
    y_label=None,
    cat_label=None,
    title=None,
    figsize=None,
    template=None,
    barmode="relative",
    total_digits_round=2,
    show_total=True,
    commas=False,
):
    if figsize is None:
        figsize = (800, 600)

    if template is None:
        template = make_pypsa_pl_template()  # pio.templates[pio.templates.default]

    if colors is None:
        colors = {
            cat_val: color for cat_val, color in zip(cat_vals, template.layout.colorway)
        }

    fig = px.bar(
        df,
        x=x_var,
        y=y_var,
        color=cat_var,
        hover_data=hover_vars,
        category_orders={cat_var: cat_vals},
        color_discrete_map=colors,
        text=y_var if text_var is None else text_var,
        template=template,
        barmode=barmode,
    )

    fig.for_each_trace(lambda t: t.update(text=[]) if t.name == "" else ())

    fig.update_traces(textposition="inside", textangle=0)
    fig.update_layout(
        title=title,
        legend={"traceorder": "reversed"},
        uniformtext_minsize=template.layout.font.size - 2,
        uniformtext_mode="hide",
    )

    if show_total:
        df_total = (
            df.groupby(x_var)[[y_var]].sum().round(total_digits_round).reset_index()
        )
        df_total.loc[df_total[y_var] == 0, y_var] = np.nan
        df_total_y = df[df[y_var] > 0].groupby(x_var)[[y_var]].sum().reset_index()
        if total_digits_round == 0:
            df_total[y_var] = df_total[y_var].astype("Int64")

        fig.add_trace(
            go.Scatter(
                x=df_total[x_var],
                y=df_total_y[y_var],
                text="<b>" + df_total[y_var].astype(str) + "</b>",
                mode="text",
                textposition="top center",
                textfont=dict(
                    size=template.layout.font.size,
                ),
                showlegend=False,
            )
        )

    if commas and total_digits_round > 0:
        fig.for_each_trace(
            lambda t: t.update(text=[str(label).replace(".", ",") for label in t.text])
        )

    if figsize is not None:
        fig.update_layout(
            autosize=False,
            width=figsize[0],
            height=figsize[1],
        )
    fig.update_xaxes(type="category")
    fig.update_layout(
        xaxis_title=x_label, yaxis_title=y_label, legend_title_text=cat_label
    )
    if y_range:
        fig.update_yaxes(range=y_range)
    if x_range:
        fig.update_xaxes(range=x_range)

    if x_label is None:
        fig.update_layout(margin=dict(b=40))
    if y_label is None:
        fig.update_layout(margin=dict(l=40))

    return fig


def plot_lines(
    df,
    x_var,
    y_var,
    cat_var,
    cat_vals,
    colors=None,
    style_var=None,
    hover_vars=None,
    x_range=None,
    y_range=None,
    x_label=None,
    y_label=None,
    cat_label=None,
    title=None,
    figsize=None,
    template=None,
    markers=True,
    x_categorical=True,
    label_position="top center",
    commas=False,
):
    if figsize is None:
        figsize = (800, 600)

    if template is None:
        template = make_pypsa_pl_template()  # pio.templates[pio.templates.default]

    if colors is None:
        colors = {
            cat_val: color for cat_val, color in zip(cat_vals, template.layout.colorway)
        }

    fig = px.line(
        df,
        x=x_var,
        y=y_var,
        color=cat_var,
        **({"line_dash": style_var} if style_var else {}),
        hover_data=hover_vars,
        category_orders={cat_var: cat_vals},
        color_discrete_map=colors,
        **({"text": y_var, "markers": True} if x_categorical else {}),
        template=template,
    )
    fig.update_traces(textposition=label_position)
    fig.update_layout(
        title=title,
        # legend={"traceorder": "reversed"},
        uniformtext_minsize=template.layout.font.size - 4,
        uniformtext_mode="hide",
    )
    if figsize is not None:
        fig.update_layout(
            autosize=False,
            width=figsize[0],
            height=figsize[1],
        )
    if x_categorical:
        fig.update_xaxes(type="category")
        # fig.update_traces(connectgaps=True)
    if commas:
        fig.for_each_trace(
            lambda t: t.update(
                text=[str(label).replace(".", ",") for label in t.text]
                if t.text is not None
                else []
            )
        )
    fig.update_layout(
        xaxis_title=x_label, yaxis_title=y_label, legend_title_text=cat_label
    )
    if y_range:
        fig.update_yaxes(range=y_range)
    if x_range:
        fig.update_xaxes(range=x_range)

    if x_label is None:
        fig.update_layout(margin=dict(b=40))
    if y_label is None:
        fig.update_layout(margin=dict(l=40))

    return fig


def read_variable(
    output_dir, component, variable, time_dependent=False, default_value=np.nan
):
    if not time_dependent:
        file = output_dir.joinpath(f"{component}.csv")
        if file.exists():
            df = pd.read_csv(
                file,
                index_col=0,
            )
            if variable not in df.columns:
                df[variable] = default_value
            df = df[[variable]].squeeze("columns")
        else:
            df = pd.DataFrame()
    else:
        file = output_dir.joinpath(f"{component}-{variable}.csv")
        if file.exists():
            df = pd.read_csv(file, index_col=0)
        else:
            df = pd.DataFrame()
    return df


def get_capacity(output_dir, aggregate_by="Technology"):
    df_country = pd.concat(
        [
            read_variable(output_dir, components, "bus")
            for components in ["generators", "storage_units"]
        ]
    )
    df_country = df_country.str[:2]
    df_country.name = "Country"

    df_tech = pd.concat(
        [
            read_variable(output_dir, components, "carrier")
            for components in ["generators", "storage_units"]
        ]
    )
    df_tech.name = "Technology"
    df_tech = df_tech.dropna()

    df_cap = (
        pd.concat(
            [
                read_variable(output_dir, components, "p_nom_opt")
                for components in ["generators", "storage_units"]
            ]
        )
        / 1e3
    )
    df_cap.name = "Installed Capacity [GW]"

    df_cap = pd.merge(df_cap, df_tech, left_index=True, right_index=True, how="right")
    df_cap = pd.merge(df_cap, df_country, left_index=True, right_index=True, how="left")

    # Drop virtual DSR
    df_cap = df_cap[~df_cap.index.str.endswith("Virtual DSR")]

    df_cap = df_cap.groupby(["Country", "Technology"]).sum()

    links = output_dir.joinpath("links.csv")
    if links.exists():
        df_trade = read_variable(output_dir, "links", "p_nom").to_frame()
        # TODO: select only trade links in a more robust way
        df_trade = df_trade[df_trade.index.str[2] == "-"].copy()
        df_trade["Installed Capacity [GW]"] = df_trade["p_nom"] / 1e3
        df_trade = df_trade.drop(columns=["p_nom"])
        df_trade["Exporter"] = df_trade.index.str[:2]
        df_trade["Importer"] = df_trade.index.str[3:5]
        df_imports = (
            df_trade.groupby("Importer")
            .agg({"Installed Capacity [GW]": "sum"})
            .reset_index()
        )
        df_imports["Technology"] = "Import"
        df_exports = (
            df_trade.groupby("Exporter")
            .agg({"Installed Capacity [GW]": "sum"})
            .reset_index()
        )
        df_exports["Technology"] = "Export"
        df_cap = pd.concat(
            [
                df_cap,
                df_imports.rename(columns={"Importer": "Country"}).set_index(
                    ["Country", "Technology"]
                ),
                df_exports.rename(columns={"Exporter": "Country"}).set_index(
                    ["Country", "Technology"]
                ),
            ]
        )

    # Round
    df_cap = df_cap.round(2)

    return df_cap.reset_index()


def get_line_load(output_dir):
    df_s_nom = read_variable(output_dir, "lines", "s_nom")
    df_s_nom.name = "s_nom"
    df_s_nom = df_s_nom.reset_index()

    df_p0 = read_variable(output_dir, "lines", "p0", time_dependent=True)
    df_p0 = df_p0.reset_index(names="snapshot_id").melt(
        id_vars="snapshot_id", var_name="name", value_name="p0"
    )
    df = pd.merge(df_p0, df_s_nom, on="name", how="left")
    is_p0_positive = df["p0"] > 0

    df["p0_pu"] = df["p0"] / df["s_nom"]
    df.loc[is_p0_positive, "p0_pos_pu"] = df.loc[is_p0_positive, "p0_pu"]
    df.loc[~is_p0_positive, "p0_neg_pu"] = df.loc[~is_p0_positive, "p0_pu"]

    df = df.groupby("name").agg(
        {"p0_pu": ["mean", "max", "min"], "p0_pos_pu": "mean", "p0_neg_pu": "mean"}
    )
    return df


def get_curtailment(output_dir, aggregate=True):
    df_bus = read_variable(output_dir, "generators", "bus")
    df_bus.name = "bus"
    df_bus = df_bus[df_bus.str.startswith("PL")]
    df_bus = df_bus.reset_index()

    df_p_nom = read_variable(output_dir, "generators", "p_nom_opt")
    df_p_nom.name = "p_nom"
    df_p_nom = df_p_nom.reset_index()

    df_p_nom = pd.merge(df_p_nom, df_bus, on="name", how="right")

    df_p_max_pu = read_variable(
        output_dir, "generators", "p_max_pu", time_dependent=True
    )
    df_p_max_pu = df_p_max_pu.reset_index(names="snapshot_id").melt(
        id_vars="snapshot_id", var_name="name", value_name="p_max_pu"
    )

    df_p_max = pd.merge(df_p_nom, df_p_max_pu, how="inner", on="name")
    df_p_max["p_max"] = df_p_max["p_nom"] * df_p_max["p_max_pu"]

    df_p = read_variable(output_dir, "generators", "p", time_dependent=True)
    df_p = df_p.reset_index(names="snapshot_id").melt(
        id_vars="snapshot_id", var_name="name", value_name="p"
    )

    df = pd.merge(df_p_max, df_p, how="left", on=["snapshot_id", "name"])
    df["p_curtailed"] = df["p_max"] - df["p"]

    df_tech = read_variable(output_dir, "generators", "carrier")
    df_tech.name = "Technology"
    df_tech = df_tech.reset_index()

    df = pd.merge(df, df_tech, how="left", on="name")
    for tech in ["PV", "Wind"]:
        df.loc[df["Technology"].str.startswith(tech), "Technology"] = tech

    df = (
        df.groupby(["Technology", "snapshot_id"])[["p_curtailed", "p_max"]]
        .sum()
        .round(2)
        .reset_index()
    )

    if aggregate:
        df = df.drop(columns="snapshot_id").groupby("Technology").sum()
        df = (
            df.rename(
                columns={
                    "p_curtailed": "Curtailment [GWh]",
                    "p_max": "Maximum production [GWh]",
                }
            )
            / 1e3
        )
        df = df.reset_index()

        df["Curtailment share [%]"] = (
            df["Curtailment [GWh]"] / df["Maximum production [GWh]"] * 100
        ).round(1)
        df = df[["Technology", "Curtailment share [%]"]]

    else:
        snapshots = read_variable(output_dir, "snapshots", "timestep")
        df = pd.merge(
            df, snapshots, left_on="snapshot_id", right_index=True, how="left"
        )
        df = df[["Technology", "timestep", "p_curtailed"]]

    return df


def get_fuel_consumption(output_dir, fuel_emissions=False):
    df = read_variable(output_dir, "generators", "p", time_dependent=True)
    df = df.reset_index(names="snapshot").melt(
        id_vars=["snapshot"], var_name="unit", value_name="p"
    )

    df_country = read_variable(output_dir, "generators", "bus")
    df_country = df_country.str[:2]
    df_country.name = "Country"

    df_fuel = read_variable(output_dir, "generators", "carrier")
    df_fuel.name = "Fuel"
    df_fuel = df_fuel.dropna()

    df_eff = read_variable(output_dir, "generators", "efficiency")
    df_eff.name = "Efficiency"

    df_eff_fuel = pd.merge(df_eff, df_fuel, left_index=True, right_index=True)

    for df_unit in [df_country, df_eff_fuel]:
        df = pd.merge(df, df_unit, left_on="unit", right_index=True, how="left")

    df = df[df["Country"] == "PL"].drop(columns="Country")

    df_carrier = read_variable(output_dir, "carriers", "co2_emissions")
    df_carrier.name = "CO2 emission factor [t/MWh_t]"
    df = pd.merge(df, df_carrier, left_on="Fuel", right_index=True, how="left")

    df["Fuel consumption [MWh_t]"] = df["p"] / df["Efficiency"]
    df["CO2 emissions [t]"] = (
        df["Fuel consumption [MWh_t]"] * df["CO2 emission factor [t/MWh_t]"]
    )

    df = (
        df.groupby("Fuel")[["Fuel consumption [MWh_t]", "CO2 emissions [t]"]]
        .sum()
        .reset_index()
    )

    df["CO2 emissions [Mt]"] = (df["CO2 emissions [t]"] / 1e6).round(2)
    df["Fuel consumption [TWh_t]"] = (df["Fuel consumption [MWh_t]"] / 1e6).round(2)
    if not fuel_emissions:
        df = df[["Fuel", "Fuel consumption [TWh_t]"]]
    else:
        df = df[["Fuel", "CO2 emissions [Mt]"]]

    return df


def get_co2_emissions(output_dir):
    return get_fuel_consumption(output_dir, fuel_emissions=True)


def get_production(output_dir, aggregate=True, aggregate_by="Technology"):
    df_country = pd.concat(
        [
            read_variable(output_dir, components, "bus")
            for components in ["generators", "storage_units"]
        ]
    )
    df_country = df_country.str[:2]
    df_country.name = "Country"

    df_tech = pd.concat(
        [
            read_variable(output_dir, components, "carrier")
            for components in ["generators", "storage_units"]
        ]
    )
    df_tech.name = "Technology"
    df_tech = df_tech.dropna()

    dfs = []

    for df, p_type in zip(
        [
            read_variable(output_dir, "generators", "p", time_dependent=True).assign(
                p_type="dispatch only"
            ),
            read_variable(
                output_dir, "storage_units", "p_dispatch", time_dependent=True
            ).assign(p_type="dispatch"),
            read_variable(
                output_dir, "storage_units", "p_store", time_dependent=True
            ).assign(p_type="store"),
        ],
        ["dispatch only", "dispatch", "store"],
    ):
        df = df.reset_index(names="snapshot").melt(
            id_vars=["snapshot"], var_name="unit", value_name="p"
        )
        df["p_type"] = p_type
        dfs.append(df)

    df_prod = pd.concat(dfs)

    df_prod.loc[df_prod["p_type"] == "store", "p"] *= -1

    # df_prod.name = "Electricity Production [TWh]"
    # df_prod = df_prod.dropna()

    df_prod = pd.merge(df_prod, df_tech, left_on="unit", right_index=True, how="right")
    df_prod = pd.merge(
        df_prod, df_country, left_on="unit", right_index=True, how="left"
    )

    has_p_type = df_prod["p_type"].isin(["dispatch", "store"])
    df_prod.loc[has_p_type, "Technology"] += " " + df_prod.loc[has_p_type, "p_type"]

    df_prod = df_prod.groupby(["Country", aggregate_by, "snapshot"]).sum().reset_index()

    links_p0 = output_dir.joinpath("links-p0.csv")
    if links_p0.exists() and aggregate_by == "Technology":
        df_trade = read_variable(output_dir, "links", "p0", time_dependent=True)
        # df_trade = df_trade.sum().to_frame() / 1e6
        df_trade = df_trade.reset_index(names="snapshot").melt(
            id_vars=["snapshot"], var_name="link", value_name="p"
        )

        # df_trade.columns = ["Electricity Production [TWh]"]
        # TODO: select only trade links in a more robust way
        df_trade = df_trade[df_trade["link"].str[2] == "-"].copy()
        df_trade["Exporter"] = df_trade["link"].str[:2]
        df_trade["Importer"] = df_trade["link"].str[3:5]
        df_imports = (
            df_trade.groupby(["Importer", "snapshot"]).agg({"p": "sum"}).reset_index()
        )
        df_imports["Technology"] = "Import"
        df_exports = (
            df_trade.groupby(["Exporter", "snapshot"]).agg({"p": "sum"}) * (-1)
        ).reset_index()
        df_exports["Technology"] = "Export"
        df_prod = pd.concat(
            [
                df_prod,
                df_imports.rename(columns={"Importer": "Country"}),
                df_exports.rename(columns={"Exporter": "Country"}),
            ]
        )

    # Round
    df_prod["p"] = df_prod["p"].round(2)

    if aggregate:
        df_prod = (
            df_prod.drop(columns="snapshot")
            .groupby(["Country", aggregate_by])
            .sum()
            .reset_index()
        )
        df_prod["p"] = (df_prod["p"] / 1e6).round(2)
        df_prod = df_prod.rename(columns={"p": "Electricity Production [TWh]"})
    else:
        snapshots = read_variable(output_dir, "snapshots", "timestep")
        df_prod = pd.merge(
            df_prod, snapshots, left_on="snapshot", right_index=True, how="left"
        )
        df_prod = df_prod[["Country", aggregate_by, "timestep", "p"]]

    return df_prod


def get_reserve(output_dir):
    dfs = []

    for components in ["generators", "storage_units"]:
        if not (
            output_dir.joinpath(f"{components}-p.csv").exists()
            or output_dir.joinpath(f"{components}-p_dispatch.csv").exists()
        ):
            continue

        df_bus = read_variable(output_dir, components, "bus")
        df_bus.name = "bus"
        df_bus = df_bus[df_bus.str.startswith("PL")]
        df_bus = df_bus.reset_index()

        df_p_nom = read_variable(output_dir, components, "p_nom_opt")
        df_p_nom.name = "p_nom"
        df_p_nom = df_p_nom.reset_index()

        df_p_nom = pd.merge(df_p_nom, df_bus, on="name", how="right")

        df_is_reserve = read_variable(output_dir, components, "is_warm_reserve")
        df_is_reserve = df_is_reserve[df_is_reserve == True]

        df_p_nom = pd.merge(df_p_nom, df_is_reserve, on="name", how="right")

        df_p = read_variable(
            output_dir,
            components,
            "p" if components == "generators" else "p_dispatch",
            time_dependent=True,
        )
        df_p = df_p.reset_index(names="snapshot_id").melt(
            id_vars="snapshot_id", var_name="name", value_name="p"
        )

        df_p_max_pu_static = read_variable(
            output_dir, components, "p_max_pu", default_value=1.0
        )
        df_p_max_pu_static.name = "p_max_pu"
        df_p_max_pu_static = df_p_max_pu_static.reset_index()

        if output_dir.joinpath(f"{components}-p_max_pu.csv").exists():
            df_p_max_pu = read_variable(
                output_dir, components, "p_max_pu", time_dependent=True
            )
            df_p_max_pu = df_p_max_pu.reset_index(names="snapshot_id").melt(
                id_vars="snapshot_id", var_name="name", value_name="p_max_pu"
            )

            static_p_max_pu_i = set(df_p_max_pu_static["name"].drop_duplicates()) - set(
                df_p_max_pu["name"].drop_duplicates()
            )
            df_p_max_pu_static = df_p_max_pu_static[
                df_p_max_pu_static["name"].isin(static_p_max_pu_i)
            ]

        df_snapshot = df_p["snapshot_id"].drop_duplicates()
        df_p_max_pu_static = pd.merge(df_snapshot, df_p_max_pu_static, how="cross")

        if output_dir.joinpath(f"{components}-p_max_pu.csv").exists():
            df_p_max_pu = pd.concat([df_p_max_pu, df_p_max_pu_static])
        else:
            df_p_max_pu = df_p_max_pu_static

        df_p_max = pd.merge(df_p_nom, df_p_max_pu, how="left", on="name")

        df_p_max["p_max"] = df_p_max["p_nom"] * df_p_max["p_max_pu"]

        df = pd.merge(df_p_max, df_p, how="left", on=["snapshot_id", "name"])
        df["p"] = df["p"].fillna(0)
        df["r"] = df["p_max"] - df["p"]

        if components == "generators":
            max_r_over_p = 1.0
            is_r_max = df["r"] > max_r_over_p * df["p"]
            df.loc[is_r_max, "r"] = max_r_over_p * df.loc[is_r_max, "p"]
            # TODO: include also the SOC condition, as implemented in pypsa constraint

        dfs.append(df)

    df = pd.concat(dfs)
    return df


def get_reserve_by_technology(output_dir, aggregate=True, trim=False):
    df_tech = pd.concat(
        [
            read_variable(output_dir, components, "carrier")
            for components in ["generators", "storage_units"]
        ]
    )
    df_tech.name = "Technology"
    df_tech = df_tech.dropna()

    df_reserve = get_reserve(output_dir)
    df_reserve = pd.merge(
        df_reserve, df_tech, left_on="name", right_index=True, how="right"
    )
    df_reserve = (
        df_reserve.groupby(["snapshot_id", "Technology"])[["r"]].sum().reset_index()
    )

    # Load
    df_load = read_variable(output_dir, "loads", "p_set", time_dependent=True)
    df_load = df_load.reset_index(names="snapshot_id").melt(
        id_vars="snapshot_id", var_name="name", value_name="load"
    )

    df_bus = read_variable(output_dir, "loads", "bus")
    df_bus.name = "bus"
    df_bus = df_bus[df_bus.str.startswith("PL")]
    df_bus = df_bus.reset_index()
    df_load = pd.merge(df_load, df_bus, how="right", on="name")
    df_load = df_load.groupby("snapshot_id").agg({"load": "sum"}).reset_index()

    # Calculate reserve and reserve needed
    df_delta = df_reserve.groupby("snapshot_id").sum().reset_index()
    df_delta = pd.merge(df_delta, df_load, how="left", on="snapshot_id")
    df_delta["r_min"] = 0.09 * df_delta["load"]
    df_delta["delta_r"] = df_delta["r"] - df_delta["r_min"]
    df_delta = df_delta[["snapshot_id", "delta_r"]]

    technologies = df_reserve["Technology"].drop_duplicates()
    df = df_reserve.pivot(index="snapshot_id", columns="Technology", values="r")
    df = pd.merge(df, df_delta, how="left", on="snapshot_id")

    df_deficit = df[df["delta_r"] <= 0].copy()
    df = df[df["delta_r"] > 0].copy()

    # Reduce r by delta_r
    if trim:
        order = ["Lignite", "Hard coal", "Natural gas", "Biomass wood chips"]
        for tech in order:
            if tech in df.columns:
                reduction = np.minimum(df["delta_r"], df[tech])
                reduction = np.maximum(reduction, 0)
                df[tech] -= reduction
                df["delta_r"] -= reduction
        df["r"] = df[technologies].sum(axis=1)
        for tech in technologies:
            df[tech] -= df[tech] / df["r"] * df["delta_r"]
        df = df.drop(columns=["r"])

    df = pd.concat([df, df_deficit])

    df = (
        df.drop(columns=["delta_r"])
        .melt(
            id_vars="snapshot_id",
            value_vars=technologies,
            var_name="Technology",
            value_name="r",
        )
        .reset_index()
    )

    df["r"] = df["r"].round(2)

    if aggregate:
        df = df.groupby("Technology").mean()["r"] / 1e3
        df.name = "Mean Reserve [GW]"
        df = df.dropna()
        df = df.round(2).reset_index()

    else:
        snapshots = read_variable(output_dir, "snapshots", "timestep")
        df = pd.merge(
            df, snapshots, left_on="snapshot_id", right_index=True, how="left"
        )
        df = df[["Technology", "timestep", "r"]]

    return df


def get_reserve_margin(output_dir):
    df = get_reserve(output_dir)
    df = df.groupby(["snapshot_id", "bus"]).agg({"r": "sum"}).reset_index()

    df_load = read_variable(output_dir, "loads", "p_set", time_dependent=True)
    df_load = df_load.reset_index(names="snapshot_id").melt(
        id_vars="snapshot_id", var_name="name", value_name="load"
    )

    df_bus = read_variable(output_dir, "loads", "bus")
    df_bus.name = "bus"
    df_bus = df_bus[df_bus.str.startswith("PL")]
    df_bus = df_bus.reset_index()
    df_load = pd.merge(df_load, df_bus, how="right", on="name").drop(columns=["name"])

    df = pd.merge(df, df_load, how="left", on=["snapshot_id", "bus"])
    df = df.groupby("snapshot_id").agg({"r": "sum", "load": "sum"}).reset_index()

    df["reserve_margin"] = (df["r"] / df["load"]).round(4)
    df.loc[df["reserve_margin"] < 0, "reserve_margin"] = 0

    reserve_ranges = [("<6%", 0.0), ("6-9%", 0.06), ("9-12%", 0.09), (">=12%", 0.12)]
    for name, value in reserve_ranges:
        df.loc[df["reserve_margin"] >= value, "Reserve margin"] = name

    df = df.groupby("Reserve margin").agg({"snapshot_id": "count"}).reset_index()
    df = df.rename(columns={"snapshot_id": "Number of hours"})
    df["Number of hours"] = df["Number of hours"].astype(int)

    df_reserve = pd.Series([el[0] for el in reserve_ranges], name="Reserve margin")

    df = pd.merge(df, df_reserve, on="Reserve margin", how="right").fillna(0)

    return df.reset_index()


def combine_data(get_data_function, scenario_name, extra_params=None, fillna=0):
    if extra_params is None:
        output_dir = runs_dir(scenario_name, "output")
        logging.info(f"Reading {output_dir}")
        df = get_data_function(output_dir)
    else:
        dfs = []
        internal_indices = []
        runs = product_dict(**extra_params)
        for run in runs:
            output_dir = runs_dir(f"{scenario_name};{dict_to_str(run)}", "output")
            logging.info(f"Reading {output_dir}")
            df = get_data_function(output_dir)
            # It is assumed that the value column is the last one
            internal_indices.append(df.iloc[:, :-1].drop_duplicates())
            for key, value in run.items():
                df[key] = value
            dfs.append(df)
        df = pd.concat(dfs)
        internal_indices = pd.concat(internal_indices).drop_duplicates()
        external_indices = pd.DataFrame.from_dict(runs)
        df_index = pd.merge(external_indices, internal_indices, how="cross")
        index_cols = df_index.columns.to_list()
        df_index = df_index.sort_values(by=index_cols)
        df = pd.merge(df_index, df, on=index_cols, how="left").fillna(fillna)
    return df


def plot_production(
    scenario_name,
    extra_params_values=None,
    extra_params=None,
    force=False,
    extension="png",
    plot_name="production",
    country="PL",
    x_var="year",
    cat_var="Technology",
    y_range=(-80, 280),
    exclude_technologies=None,
):
    for dir in ["data", "plots"]:
        os.makedirs(output_dir(scenario_name, dir), exist_ok=True)

    suffix_dict = {"scenario": scenario_name}
    if extra_params_values is not None:
        suffix_dict.update(extra_params_values)
    suffix = dict_to_str(suffix_dict)
    if country != "PL":
        suffix += f";country={country}"

    data_file = output_dir(scenario_name, "data", f"{plot_name};{suffix}.csv")
    plot_file = output_dir(scenario_name, "plots", f"{plot_name};{suffix}.{extension}")

    y_var = "Electricity Production [TWh]"

    if not data_file.exists() or force:
        df = combine_data(get_production, scenario_name, extra_params)
        df.to_csv(data_file, index=False)
    else:
        df = pd.read_csv(data_file)
    if "Country" in df.columns:
        df = df[df["Country"] == country].drop(columns=["Country"])
    if extra_params_values is not None:
        for var, val in extra_params_values.items():
            df = df[df[var] == val].drop(columns=[var])
    df = df[[x_var, cat_var, y_var]]
    df[y_var] = df[y_var].round(1)

    if cat_var == "Technology" and exclude_technologies:
        df = df[~df["Technology"].isin(exclude_technologies)]

    cat_vals = [val for val in technology_colors.keys() if val in set(df[cat_var])]
    colors = {cat_val: technology_colors[cat_val] for cat_val in cat_vals}

    fig = plot_bars(
        df,
        x_var=x_var,
        y_var=y_var,
        cat_var=cat_var,
        cat_vals=cat_vals,
        colors=colors,
        title=y_var,
        y_label="",
        x_label="",
        template=make_pypsa_pl_template(),
        y_range=y_range,
    )
    fig.write_image(plot_file)
    fig.show()


def plot_curtailment(
    scenario_name,
    extra_params_values=None,
    extra_params=None,
    force=False,
    extension="png",
    plot_name="curtailment",
    country="PL",
    x_var="year",
    cat_var="Technology",
):
    for dir in ["data", "plots"]:
        os.makedirs(output_dir(scenario_name, dir), exist_ok=True)

    suffix_dict = {"scenario": scenario_name}
    if extra_params_values is not None:
        suffix_dict.update(extra_params_values)
    suffix = dict_to_str(suffix_dict)

    data_file = output_dir(scenario_name, "data", f"{plot_name};{suffix}.csv")
    plot_file = output_dir(scenario_name, "plots", f"{plot_name};{suffix}.{extension}")

    y_var = "Curtailment share [%]"

    if not data_file.exists() or force:
        df = combine_data(get_curtailment, scenario_name, extra_params)
        df.to_csv(data_file, index=False)
    else:
        df = pd.read_csv(data_file)

    if "Country" in df.columns:
        df = df[df["Country"] == country].drop(columns=["Country"])
        if extra_params_values is not None:
            for var, val in extra_params_values.items():
                df = df[df[var] == val].drop(columns=[var])
        df = df[[x_var, cat_var, y_var]]

    df[cat_var] = df[cat_var].astype(str)

    cat_vals = [val for val in technology_colors.keys() if val in set(df[cat_var])]
    colors = {cat_val: technology_colors[cat_val] for cat_val in cat_vals}

    fig = plot_bars(
        df,
        x_var=x_var,
        y_var=y_var,
        cat_var=cat_var,
        cat_vals=cat_vals,
        colors=colors,
        title=y_var,
        y_label="",
        x_label=x_var,
        template=make_pypsa_pl_template(),
        barmode="group",
        # y_range=(-80, 280),
    )
    fig.write_image(plot_file)
    fig.show()


def plot_reserve_by_technology(
    scenario_name,
    extra_params_values=None,
    extra_params=None,
    force=False,
    extension="png",
    plot_name="reserve_by_technology",
    x_var="year",
    cat_var="Technology",
):
    for dir in ["data", "plots"]:
        os.makedirs(output_dir(scenario_name, dir), exist_ok=True)

    suffix_dict = {"scenario": scenario_name}
    if extra_params_values is not None:
        suffix_dict.update(extra_params_values)
    suffix = dict_to_str(suffix_dict)

    data_file = output_dir(scenario_name, "data", f"{plot_name};{suffix}.csv")
    plot_file = output_dir(scenario_name, "plots", f"{plot_name};{suffix}.{extension}")

    y_var = "Mean Reserve [GW]"
    cat_var = "Technology"

    if not data_file.exists() or force:
        df = combine_data(get_reserve_by_technology, scenario_name, extra_params)
        if extra_params_values is not None:
            for var, val in extra_params_values.items():
                df = df[df[var] == val].drop(columns=[var])
        df = df[[x_var, cat_var, y_var]]
        df.to_csv(data_file, index=False)
    else:
        df = pd.read_csv(data_file)

    cat_vals = [val for val in technology_colors.keys() if val in set(df[cat_var])]
    colors = {cat_val: technology_colors[cat_val] for cat_val in cat_vals}

    fig = plot_bars(
        df,
        x_var=x_var,
        y_var=y_var,
        cat_var=cat_var,
        cat_vals=cat_vals,
        colors=colors,
        title=y_var,
        y_label="",
        x_label="",
        y_range=(0, 4),
        template=make_pypsa_pl_template(),
    )
    fig.write_image(plot_file)
    fig.show()


def plot_capacity(
    scenario_name,
    extra_params_values=None,
    extra_params=None,
    force=False,
    extension="png",
    plot_name="capacity",
    country="PL",
    x_var="year",
    cat_var="Technology",
    y_range=(0, 150),
    exclude_technologies=None,
):
    for dir in ["data", "plots"]:
        os.makedirs(output_dir(scenario_name, dir), exist_ok=True)

    suffix_dict = {"scenario": scenario_name}
    if extra_params_values is not None:
        suffix_dict.update(extra_params_values)
    suffix = dict_to_str(suffix_dict)
    if country != "PL":
        suffix += f";country={country}"

    data_file = output_dir(scenario_name, "data", f"{plot_name};{suffix}.csv")
    plot_file = output_dir(scenario_name, "plots", f"{plot_name};{suffix}.{extension}")

    y_var = "Installed Capacity [GW]"

    if not data_file.exists() or force:
        df = combine_data(get_capacity, scenario_name, extra_params)
        df.to_csv(data_file, index=False)

    else:
        df = pd.read_csv(data_file)

    if "Country" in df.columns:
        df = df[df["Country"] == country].drop(columns=["Country"])
    if extra_params_values is not None:
        for var, val in extra_params_values.items():
            df = df[df[var] == val].drop(columns=[var])
    df = df[[x_var, cat_var, y_var]]
    df[y_var] = df[y_var].round(1)

    if cat_var == "Technology" and exclude_technologies:
        df = df[~df["Technology"].isin(exclude_technologies)]

    cat_vals = [val for val in technology_colors.keys() if val in set(df[cat_var])]
    colors = {cat_val: technology_colors[cat_val] for cat_val in cat_vals}

    fig = plot_bars(
        df,
        x_var=x_var,
        y_var=y_var,
        cat_var=cat_var,
        cat_vals=cat_vals,
        colors=colors,
        title=y_var,
        y_range=y_range,
        y_label="",
        x_label="",
        template=make_pypsa_pl_template(),
    )
    fig.write_image(plot_file)
    fig.show()


def plot_capacity_utilisation(
    scenario_name,
    extra_params_values=None,
    extra_params=None,
    force=False,
    extension="png",
    plot_name="capacity_utilisation",
    country="PL",
    x_var="year",
    cat_var="Technology",
):
    for dir in ["data", "plots"]:
        os.makedirs(output_dir(scenario_name, dir), exist_ok=True)

    suffix_dict = {"scenario": scenario_name}
    if extra_params_values is not None:
        suffix_dict.update(extra_params_values)
    suffix = dict_to_str(suffix_dict)

    data_file = output_dir(scenario_name, "data", f"{plot_name};{suffix}.csv")
    plot_file = output_dir(scenario_name, "plots", f"{plot_name};{suffix}.{extension}")

    cat_var = "Technology"
    y_var = "Capacity Utilisation [%]"

    if not data_file.exists() or force:
        df_prod = combine_data(get_production, scenario_name, extra_params)
        df_cap = combine_data(get_capacity, scenario_name, extra_params)
        if extra_params_values is not None:
            for var, val in extra_params_values.items():
                df_prod = df_prod[df_prod[var] == val].drop(columns=[var])
                df_cap = df_cap[df_cap[var] == val].drop(columns=[var])
        merge_on = (
            [x_var, "Technology"]
            if "Country" not in df_prod.columns
            else [x_var, "Country", "Technology"]
        )
        # Fix data for storage units and imports/exports to calculate utilization
        df_prod.loc[
            df_prod["Technology"] == "Export", "Electricity Production [TWh]"
        ] *= -1
        df_prod["Technology"] = df_prod["Technology"].str.replace(" dispatch", "")
        df_prod = df_prod[~df_prod["Technology"].str.endswith(" store")]

        df = pd.merge(df_prod, df_cap, on=merge_on, how="left")
        df["Capacity Utilisation [%]"] = (
            100
            * df["Electricity Production [TWh]"]
            / df["Installed Capacity [GW]"]
            * 1000
            / 8760
        ).round(1)
        df.to_csv(data_file, index=False)
    else:
        df = pd.read_csv(data_file)

    if "Country" in df.columns:
        df = df[df["Country"] == country].drop(columns=["Country"])
    df = df[[x_var, cat_var, y_var]]

    cat_vals = [val for val in technology_colors.keys() if val in set(df[cat_var])]
    colors = {cat_val: technology_colors[cat_val] for cat_val in cat_vals}

    fig = plot_bars(
        df,
        x_var=x_var,
        y_var=y_var,
        cat_var=cat_var,
        cat_vals=cat_vals,
        colors=colors,
        title=y_var,
        y_label="",
        x_label="",
        y_range=(0, 100),
        template=make_pypsa_pl_template(),
        barmode="group",
        figsize=(3000, 600),
    )
    fig.write_image(plot_file)
    fig.show()


def plot_fuel_consumption(
    scenario_name,
    extra_params_values=None,
    extra_params=None,
    force=False,
    extension="png",
    plot_name="fuel_consumption",
    country="PL",
    x_var="year",
    cat_var="Fuel",
    fuels=["Natural gas", "Hard coal", "Lignite"],
):
    for dir in ["data", "plots"]:
        os.makedirs(output_dir(scenario_name, dir), exist_ok=True)

    suffix_dict = {"scenario": scenario_name}
    if extra_params_values is not None:
        suffix_dict.update(extra_params_values)
    suffix = dict_to_str(suffix_dict)

    data_file = output_dir(scenario_name, "data", f"{plot_name};{suffix}.csv")
    plot_file = output_dir(scenario_name, "plots", f"{plot_name};{suffix}.{extension}")

    cat_var = "Fuel"
    y_var = "Fuel consumption [TWh_t]"

    if not data_file.exists() or force:
        df = combine_data(get_fuel_consumption, scenario_name, extra_params)
        df = df[df["Fuel"].isin(fuels)]
        if extra_params_values is not None:
            for var, val in extra_params_values.items():
                df = df[df[var] == val].drop(columns=[var])
        df.to_csv(data_file, index=False)
    else:
        df = pd.read_csv(data_file)

    if "Country" in df.columns:
        df = df[df["Country"] == country].drop(columns=["Country"])
    df = df[[x_var, cat_var, y_var]]

    cat_vals = [val for val in technology_colors.keys() if val in set(df[cat_var])]
    colors = {cat_val: technology_colors[cat_val] for cat_val in cat_vals}

    fig = plot_bars(
        df,
        x_var=x_var,
        y_var=y_var,
        cat_var=cat_var,
        cat_vals=cat_vals,
        colors=colors,
        title=y_var,
        y_label="",
        x_label="",
        y_range=(0, 170),
        template=make_pypsa_pl_template(),
        barmode="group",
    )
    fig.write_image(plot_file)
    fig.show()


def plot_co2_emissions(
    scenario_name,
    extra_params_values=None,
    extra_params=None,
    force=False,
    extension="png",
    plot_name="co2_emissions",
    country="PL",
    x_var="year",
    cat_var="Fuel",
    fuels=["Natural gas", "Hard coal", "Lignite"],
):
    for dir in ["data", "plots"]:
        os.makedirs(output_dir(scenario_name, dir), exist_ok=True)

    suffix_dict = {"scenario": scenario_name}
    if extra_params_values is not None:
        suffix_dict.update(extra_params_values)
    suffix = dict_to_str(suffix_dict)

    data_file = output_dir(scenario_name, "data", f"{plot_name};{suffix}.csv")
    plot_file = output_dir(scenario_name, "plots", f"{plot_name};{suffix}.{extension}")

    cat_var = "Fuel"
    y_var = "CO2 emissions [Mt]"

    if not data_file.exists() or force:
        df = combine_data(get_co2_emissions, scenario_name, extra_params)
        df = df[df["Fuel"].isin(fuels)]
        if extra_params_values is not None:
            for var, val in extra_params_values.items():
                df = df[df[var] == val].drop(columns=[var])
        df.to_csv(data_file, index=False)
    else:
        df = pd.read_csv(data_file)

    if "Country" in df.columns:
        df = df[df["Country"] == country].drop(columns=["Country"])
    df = df[[x_var, cat_var, y_var]]

    cat_vals = [val for val in technology_colors.keys() if val in set(df[cat_var])]
    colors = {cat_val: technology_colors[cat_val] for cat_val in cat_vals}

    fig = plot_bars(
        df,
        x_var=x_var,
        y_var=y_var,
        cat_var=cat_var,
        cat_vals=cat_vals,
        colors=colors,
        title=y_var,
        y_label="",
        x_label="",
        y_range=(0, 110),
        template=make_pypsa_pl_template(),
    )
    fig.write_image(plot_file)
    fig.show()


def plot_reserve_margin(
    scenario_name,
    extra_params_values=None,
    extra_params=None,
    force=False,
    extension="png",
    plot_name="reserve",
    x_var="year",
    cat_var="Reserve margin",
):
    for dir in ["data", "plots"]:
        os.makedirs(output_dir(scenario_name, dir), exist_ok=True)

    suffix_dict = {"scenario": scenario_name}
    if extra_params_values is not None:
        suffix_dict.update(extra_params_values)
    suffix = dict_to_str(suffix_dict)

    data_file = output_dir(scenario_name, "data", f"{plot_name};{suffix}.csv")
    plot_file = output_dir(scenario_name, "plots", f"{plot_name};{suffix}.{extension}")

    cat_var = "Reserve margin"
    y_var = "Number of hours"

    if not data_file.exists() or force:
        df = combine_data(get_reserve_margin, scenario_name, extra_params)
        if extra_params_values is not None:
            for var, val in extra_params_values.items():
                df = df[df[var] == val].drop(columns=[var])
        df = df[[x_var, cat_var, y_var]]
        df.to_csv(data_file, index=False)
    else:
        df = pd.read_csv(data_file)

    cat_vals = df[cat_var].drop_duplicates().to_list()
    colors = ["#cc6d00", "#ff8800", "#49b675", "#3a925e"]
    colors = {cat_val: color for cat_val, color in zip(cat_vals, colors)}

    fig = plot_bars(
        df,
        x_var=x_var,
        y_var=y_var,
        cat_var=cat_var,
        cat_vals=cat_vals,
        colors=colors,
        title="Number of hours with a given power reserve margin",
        y_label="",
        x_label="",
        y_range=(0, 6500),
        template=make_pypsa_pl_template(),
        barmode="group",
        figsize=(1200, 600),
    )
    fig.write_image(plot_file)
    fig.show()
