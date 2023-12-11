import logging
import os
import json
import pandas as pd
import numpy as np
from adjustText import adjust_text
from datetime import datetime

from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
from seaborn import objects as so

from pypsa_pl.config import data_dir
from pypsa_pl.io import dict_to_str, extend_param_list, read_excel
from pypsa_pl.colors import technology_colors
from pypsa_pl.process_technology_data import get_technology_year
from pypsa_pl.helper_functions import calculate_annuity


plt.set_loglevel("WARNING")


def runs_dir(*path):
    return data_dir("runs", *path)


def output_dir(*path):
    return data_dir("output", *path)


dark_color = "#231f20"
light_color = "#6c7178"
bg_color = "#dcddde"
default_size = 10


def set_instrat_template(font_scale):
    palette = [
        "#c1843d",
        "#535ce3",
        "#d9d9d9",
        "#212121",
        "#2979ff",
        "#01e677",
        "#e040fb",
        "#4b7689",
    ]

    sns.set_theme(
        context="notebook",
        style="whitegrid",
        font="Work Sans",
        font_scale=font_scale,
        palette=palette,
        rc={
            "font.weight": 400,  # 300,
            "axes.labelweight": 400,  # 300,
            "axes.labelsize": default_size,
            "axes.labelcolor": light_color,
            "axes.edgecolor": "#939598",
            "xtick.color": light_color,
            "ytick.color": light_color,
            "ytick.labelsize": default_size,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "figure.titlesize": "x-large",
            "grid.color": bg_color,
            "grid.linewidth": 0.5,
            "axes.linewidth": 0.5,
        },
    )

    # font_size = mpl.rcParams["font.size"]
    # mpl.rcParams["figure.titlesize"] = font_size + 10

    # If font is not found clear the matplotlib cache
    # rm -r ~/.cache/matplotlib/

    # https://github.com/seaborn/seaborn/blob/d1c04f2c2c4dbd11ede016405b5ea51380e37f51/seaborn/rcmod.py#L386


def make_timestamp():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return timestamp


def calculate_luminance(rgb):
    # https://www.w3.org/TR/AERT/#color-contrast
    return rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114


def determine_label_locations(values, small_value, n_locs=5, mirror=False):
    mid_loc = n_locs // 2
    locs = [mid_loc] * len(values)
    for i in range(len(locs)):
        if abs(values[i]) < small_value:
            if i == 0 or abs(values[i - 1]) >= small_value:
                locs[i] = 0
            else:
                new_loc = (locs[i - 1] + 1) % n_locs
                if new_loc == mid_loc:
                    new_loc = (new_loc + 1) % n_locs
                locs[i] = new_loc
    locs = [loc - mid_loc for loc in locs]
    if mirror:
        locs = [-loc for loc in locs]
    return locs


def format_labels(digits=1, commas=False):
    n_digits = lambda val: digits if abs(val) >= 10 ** (-digits) else digits + 1
    return lambda val: f"{val:.{n_digits(val)}f}".replace(".", "," if commas else ".")


def add_bar_labels(
    ax,
    bar_containers,
    horizontal=False,
    n_locs=5,
    loc_width=40,
    small_value=10,
    label_digits=1,
    mirror=False,
    show_total=True,
    total_offset=10,
    label_scale=1,
    total_label_scale=None,
    commas=False,
):
    if total_label_scale is None:
        total_label_scale = label_scale
    containers_per_x = {}
    for container in bar_containers:
        patches = container.patches
        values = container.datavalues
        x_values = [
            (patch.get_x() if not horizontal else patch.get_y()) for patch in patches
        ]
        for patch, value, x in zip(patches, values, x_values):
            if x not in containers_per_x.keys():
                containers_per_x[x] = {"patches": [], "values": []}
            containers_per_x[x]["patches"].append(patch)
            containers_per_x[x]["values"].append(value)
    containers_per_x = {
        x: mpl.container.BarContainer(
            container["patches"],
            datavalues=container["values"],
            orientation="vertical" if not horizontal else "horizontal",
        )
        for x, container in containers_per_x.items()
    }

    for container in containers_per_x.values():
        # Bar labels
        locs = determine_label_locations(
            container.datavalues,
            small_value,
            n_locs=n_locs,
            mirror=mirror,
        )
        colors = [patch.get_facecolor() for patch in container.patches]
        labels = ax.bar_label(
            container,
            fmt=format_labels(digits=label_digits, commas=commas),
            label_type="center",
        )
        for label, color, loc in zip(labels, colors, locs):
            label.set(
                **{"x" if not horizontal else "y": loc * loc_width},
                bbox=dict(
                    boxstyle="square,pad=0.1",
                    facecolor=color,
                    linewidth=0,
                ),
                color="white" if calculate_luminance(color) < 0.5 else "black",
                fontsize=mpl.rcParams["ytick.labelsize"] * label_scale,
            )
        # Total labels
        if show_total:
            # TODO: figure out why setting datavalues to [total] does not influence the labels
            total = sum(container.datavalues)
            top_container = mpl.container.BarContainer(
                container.patches[-1:],
                datavalues=[total],
                orientation="vertical" if not horizontal else "horizontal",
            )
            labels = ax.bar_label(
                top_container,
                labels=[format_labels(digits=label_digits, commas=commas)(total)],
            )
            for label in labels:
                label.set(
                    **{
                        "y"
                        if not horizontal
                        else "x": total_offset * (-1 if mirror else 1)
                    },
                    bbox=dict(
                        boxstyle="square,pad=0.1",
                        alpha=0,
                        linewidth=0,
                    ),
                    color=dark_color,
                    fontsize=mpl.rcParams["ytick.labelsize"] * total_label_scale,
                    fontweight=400,  # 500,
                )


def add_text_labels(
    ax,
    x_list,
    y_list,
    value_list,
    move=(1, 0),
    commas=False,
    label_digits=1,
):
    objects = []
    for x, y, value in zip(x_list, y_list, value_list):
        txt = format_labels(digits=label_digits, commas=commas)(value)
        obj = ax.annotate(
            txt,
            (x, y),
            fontsize=mpl.rcParams["ytick.labelsize"],
            fontweight=400,  # 500,
            textcoords="offset fontsize",
            xytext=(0.5, -1),
        )
        objects.append(obj)
    adjust_text(
        objects,
        autoalign="y",
        only_move={"points": "y", "text": "y", "objects": "y"},
        expand_text=(1.5, 1.5),
        force_text=(0, 1),
        force_points=(0, 0),
        force_objects=(0, 0),
        ax=ax,
    )
    # Move all labels to the right
    for obj in objects:
        obj.set_position(
            (obj.get_position()[0] + move[0], obj.get_position()[1] + move[1])
        )


def plot_lines(
    df,
    x_var,
    y_var,
    cat_var,
    cat_style_dict=technology_colors,
    band=False,
    lines=True,
    markers=False,
    y_var_min=None,
    y_var_max=None,
    x_range=None,
    y_range=None,
    x_label=None,
    x_ticks=None,
    y_unit=None,
    cat_label=None,
    title=None,
    footer=None,
    figsize=None,
    data_labels=False,
    label_digits=1,
    commas=False,
    font_scale=0.85,
    move=(1, 0),
):
    set_instrat_template(font_scale=font_scale)

    if figsize is None:
        figsize = (16, 16)
    if x_label is None:
        x_label = x_var
    if y_unit is None:
        y_unit = y_var.split(" [")[1][:-1]
    # if title is None:
    #     title = y_var

    fig, ax = plt.subplots(figsize=figsize)

    # Determine ordering and styling of categorical variable
    df_plot = df.copy()

    cat_vals = df_plot[cat_var].unique()
    cat_vals = [v for v in cat_style_dict.keys() if v in cat_vals]
    cat_style = {v: cat_style_dict[v] for v in cat_vals}
    df_plot[cat_var] = pd.Categorical(df_plot[cat_var], cat_vals)
    df_plot = df_plot.sort_values(cat_var)

    plot = so.Plot(df_plot, x=x_var, y=y_var, color=cat_var)
    if markers:
        plot = plot.add(so.Dot(pointsize=6))

    if lines:
        plot = plot.add(so.Line())

    if band:
        plot = plot.add(so.Band(), ymin=y_var_min, ymax=y_var_max)

    (
        plot.scale(
            color=so.Nominal(cat_style, order=cat_vals),
        )
        .on(ax)
        .plot()
    )

    if data_labels:
        df_labels = df_plot[[x_var, y_var]].drop_duplicates()
        add_text_labels(
            ax,
            df_labels[x_var],
            df_labels[y_var],
            df_labels[y_var],
            commas=commas,
            label_digits=label_digits,
            move=move,
        )

    legend = fig.legends.pop()
    handles = legend.legendHandles
    labels = [t.get_text() for t in legend.get_texts()]

    legend = ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        title=cat_label,
        title_fontproperties={"weight": "normal"},
        alignment="left",
    )

    ax.set_ylabel(y_unit, ha="left", y=1, rotation=0, labelpad=0)
    ax.set_xlabel(x_label)

    if x_range is not None:
        ax.set_xlim(x_range)

    if y_range is not None:
        ax.set_ylim(y_range)

    if x_ticks is not None:
        ax.set_xticks([tick[0] for tick in x_ticks])
        ax.set_xticklabels([tick[1] for tick in x_ticks])

    x_header = 0.02
    if title is not None:
        fig.suptitle(title, ha="left", x=x_header)

    footer_height = 0
    if footer is not None:
        footer_height = 0.03 + footer.count("\n") * 0.015
        fig.text(
            x_header,
            0.01,
            footer,
            ha="left",
            va="bottom",
            multialignment="left",
            fontdict={"size": mpl.rcParams["legend.fontsize"]},
        )

    fig.tight_layout(rect=(0, footer_height, 1, 1))

    return fig


def plot_bars(
    df,
    x_var,
    y_var,
    cat_var,
    cat_style_dict=technology_colors,
    x_var_categorical=True,
    bar_width=0.8,
    area=False,
    x_hours=None,
    text_var=None,
    x_range=None,
    y_range=None,
    x_label=None,
    x_ticks_rotation=0,
    y_unit=None,
    cat_label=None,
    cat_label_positive=None,
    cat_label_negative=None,
    positive_only=False,
    title=None,
    footer=None,
    figsize=None,
    stack=True,
    label_digits=1,
    label_scale=1,
    total_label_scale=None,
    show_total=None,
    one_total=False,
    commas=False,
    small_value=20,
    label_kwargs={},
    data_labels=True,
    font_scale=0.85,
    horizontal=False,
    n_cols_legend=None,
):
    set_instrat_template(font_scale=font_scale)

    if figsize is None:
        figsize = (16, 16)
    if cat_label_positive is None:
        cat_label_positive = cat_label
    if show_total is None:
        show_total = stack
    if x_label is None:
        x_label = x_var
    if y_unit is None:
        y_unit = y_var.split(" [")[1][:-1]
    # if title is None:
    #     title = y_var

    fig, ax = plt.subplots(figsize=figsize)

    for val_type in ["negative", "positive"]:
        if val_type == "negative":
            df_plot = df[df[y_var].round(label_digits + 1) < 0].copy()
            if df_plot.empty:
                positive_only = True
                continue
        else:
            df_plot = df[df[y_var].round(label_digits + 1) > 0].copy()

        # Determine ordering and styling of categorical variable
        cat_vals = df_plot[cat_var].unique()
        cat_vals = [v for v in cat_style_dict.keys() if v in cat_vals]
        cat_style = {v: cat_style_dict[v] for v in cat_vals}
        df_plot[cat_var] = pd.Categorical(df_plot[cat_var], cat_vals)
        df_plot = df_plot.sort_values(cat_var)

        if stack:
            if (
                val_type == "positive"
                and not horizontal
                or val_type == "negative"
                and horizontal
            ):
                cat_vals_legend = cat_vals[::-1]
            else:
                cat_vals_legend = cat_vals
        else:
            cat_vals_legend = cat_vals

        if x_var_categorical:
            x_vals = df_plot[x_var].drop_duplicates().sort_values()
            df_plot[x_var] = pd.Categorical(df_plot[x_var], x_vals)

        if not horizontal:
            xy_args = {"x": x_var, "y": y_var}
        else:
            xy_args = {"x": y_var, "y": x_var}

        if area:
            plot_type = so.Area(alpha=1, edgewidth=0)
            assert stack
            assert not data_labels
        else:
            plot_type = so.Bar(alpha=1, edgewidth=0, width=bar_width)

        (
            so.Plot(df_plot, **xy_args, color=cat_var)
            .add(plot_type, so.Stack() if stack else so.Dodge())
            .scale(
                color=so.Nominal(cat_style, order=cat_vals_legend),
            )
            .on(ax)
            .plot()
        )

        if data_labels:
            if not one_total:
                bar_containers = ax.containers[-1:]
            else:
                bar_containers = ax.containers

            if not one_total or val_type == "positive":
                add_bar_labels(
                    ax,
                    bar_containers,
                    horizontal=horizontal,
                    n_locs=5
                    if "n_locs" not in label_kwargs.keys()
                    else label_kwargs["n_locs"],
                    loc_width=30
                    if "loc_width" not in label_kwargs.keys()
                    else label_kwargs["loc_width"],
                    small_value=small_value,
                    label_digits=label_digits,
                    label_scale=label_scale,
                    total_label_scale=total_label_scale,
                    mirror=val_type == "negative",
                    show_total=show_total,
                    total_offset=0.75 * figsize[1],
                    commas=commas,
                )

        legend = fig.legends.pop()
        handles = legend.legendHandles
        labels = [t.get_text() for t in legend.get_texts()]

        if val_type == "negative" and cat_label_negative is not None:
            legend_title = cat_label_negative
        elif val_type == "positive" and cat_label_positive is not None:
            legend_title = cat_label_positive
        else:
            legend_title = val_type

        if not horizontal:
            x_legend = 1.05
            y_legend = (
                0.13 if val_type == "negative" else (0.5 if positive_only else 0.68)
            )
        else:
            x_legend = 0.5
            y_legend = (
                1.02 if val_type == "negative" else (1.10 if positive_only else 1.02)
            )

        if n_cols_legend is None:
            n_cols_legend = 1 if not horizontal else len(handles)
        # https://stackoverflow.com/questions/66783109/matplotlibs-legend-how-to-order-entries-by-row-first-rather-than-by-column
        reorder = lambda hl: (
            sum((lis[i::n_cols_legend] for i in range(n_cols_legend)), []) for lis in hl
        )

        legend = ax.legend(
            *reorder((handles, labels)),
            loc="center left" if not horizontal else "lower center",
            bbox_to_anchor=(x_legend, y_legend),
            title=legend_title,
            title_fontproperties={"weight": "normal"},
            alignment="left",
            ncol=(1 if not horizontal else len(handles))
            if n_cols_legend is None
            else n_cols_legend,
            # mode="expand",
        )
        if val_type == "negative" and not positive_only:
            artist = ax.add_artist(legend)
            artist.set_in_layout(True)

    if not horizontal:
        ax.set_ylabel(y_unit, ha="left", y=1, rotation=0, labelpad=0)
        ax.set_xlabel(x_label)
        if x_ticks_rotation != 0:
            for label in ax.get_xmajorticklabels():
                label.set_rotation(x_ticks_rotation)
            ax.tick_params(axis="x", labelrotation=x_ticks_rotation)
        if x_var_categorical:
            for label in ax.get_xticklabels():
                label.set(fontweight=400)  # 500)
        if x_hours is not None:
            locator = mpl.dates.HourLocator(byhour=x_hours)
            fmt = mpl.dates.DateFormatter("%d.%m\n%H:%M")
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(fmt)

        if x_range is not None:
            ax.set_xlim(x_range)

        if y_range is not None:
            ax.set_ylim(y_range)
    else:
        ax.set_xlabel(y_unit)  # , va="bottom", x=1, rotation=0, labelpad=0)
        ax.set_ylabel(x_label)
        if x_ticks_rotation != 0:
            for label in ax.get_ymajorticklabels():
                label.set_rotation(x_ticks_rotation)
            ax.tick_params(axis="y", labelrotation=x_ticks_rotation)
        if x_var_categorical:
            for label in ax.get_yticklabels():
                label.set(fontweight=400)  # 500)
        if x_hours is not None:
            locator = mpl.dates.HourLocator(byhour=x_hours)
            fmt = mpl.dates.DateFormatter("%d.%m %H:%M")
            ax.yaxis.set_major_locator(locator)
            ax.yaxis.set_major_formatter(fmt)

        if x_range is not None:
            ax.set_ylim(x_range)

        if y_range is not None:
            ax.set_xlim(y_range)

    x_header = 0.02
    if title is not None:
        fig.suptitle(title, ha="left", x=x_header)

    footer_height = 0
    if footer is not None:
        footer_height = 0.03 + footer.count("\n") * 0.015
        fig.text(
            x_header,
            0.01,
            footer,
            ha="left",
            va="bottom",
            multialignment="left",
            fontdict={"size": mpl.rcParams["legend.fontsize"]},
        )

    fig.tight_layout(rect=(0, footer_height, 1, 1))

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


def get_costs(output_dir, cost_component="opex", sector="electricity"):
    # TODO: use statistics_fixed_capacity.csv rather than statistics.csv
    meta = json.load(open(output_dir.joinpath("meta.json")))
    max_price = meta["srmc_dsr"]
    df = pd.read_csv(output_dir.joinpath("statistics.csv"))
    if cost_component == "opex":
        column = "Operational Expenditure"
    if cost_component == "capex":
        column = "Capital Expenditure"
    df = df[df["area"].fillna("PL").str.startswith("PL")].drop(columns=["area"])

    df = df[df["component"] != "Load"]
    df = df[~df["carrier"].isin(["AC", "DC"])]
    df = df[~df["carrier"].str.startswith("Virtual DSR")]

    # Map special sectoral buses to the actual sectoral carriers
    bus_carriers_per_sector = {
        "heat": ["District heating", "Heat pump small"],
        "electricity": ["biogas"],
    }
    for sector_iter, carriers in bus_carriers_per_sector.items():
        sectoral_carrier = sector_iter if sector_iter != "electricity" else "AC"
        for carrier in carriers:
            for bus in ["bus", "bus0", "bus1"]:
                df[f"{bus}_carrier"] = df[f"{bus}_carrier"].replace(
                    carrier, sectoral_carrier
                )

    df["carrier"] = df["carrier"].replace(
        "District heating output", "District heating distribution"
    )
    df = df[~df["carrier"].str.endswith("output")]

    sectoral_carrier = sector if sector != "electricity" else "AC"
    df = df[
        (df["bus_carrier"] == sectoral_carrier)
        | ((df["p0_sign"] > 0) & (df["bus1_carrier"] == sectoral_carrier))
        | ((df["p0_sign"] < 0) & (df["bus0_carrier"] == sectoral_carrier))
    ]

    df = df[["carrier", column]].groupby("carrier").sum().reset_index()
    # TODO: verify why statistics are per snapshot
    df[column] = df[column] * 8760 / 1e9
    df = df.rename(columns={"carrier": "Technology", column: f"{column} [bln PLN]"})

    if sector == "electricity":
        df = df.set_index("Technology")
        df.loc["Import", f"{column} [bln PLN]"] = 0
        df.loc["Export", f"{column} [bln PLN]"] = 0
        df = df.reset_index()

        # Opex is special because one needs to calculate import costs and export revenues under certain assumptions on trade prices
        if cost_component == "opex" and output_dir.joinpath("links-p0.csv").exists():
            df_trade = read_variable(output_dir, "links", "p0", time_dependent=True)
            df_price = read_variable(
                output_dir, "buses", "marginal_price", time_dependent=True
            )
            df_bus_carrier = read_variable(output_dir, "buses", "carrier")
            electricity_buses = df_bus_carrier[df_bus_carrier == "AC"].index
            price_domestic = df_price[
                [bus for bus in electricity_buses if bus.startswith("PL")]
            ].mean(axis=1)
            price_domestic = price_domestic.where(price_domestic < max_price, max_price)
            df_bus0 = read_variable(output_dir, "links", "bus0")
            df_bus1 = read_variable(output_dir, "links", "bus1")
            df_bus = pd.concat([df_bus0, df_bus1], axis=1)
            df_bus = df_bus[
                (
                    ~df_bus["bus0"].str.startswith("PL")
                    & df_bus["bus1"].str.startswith("PL")
                )
                | df_bus["bus0"].str.startswith("PL")
                & ~df_bus["bus1"].str.startswith("PL")
            ]
            neighbors = [
                n
                for n in set(df_bus["bus0"]).union(set(df_bus["bus1"]))
                if not n.startswith("PL")
            ]

            for neighbor in neighbors:
                imports_i = df_bus[df_bus["bus0"] == neighbor].index.intersection(
                    df_trade.columns
                )
                exports_i = df_bus[df_bus["bus1"] == neighbor].index.intersection(
                    df_trade.columns
                )
                imports = df_trade[imports_i].sum(axis=1)
                exports = df_trade[exports_i].sum(axis=1)
                price_neighbor = df_price[neighbor]
                price_neighbor = price_neighbor.where(
                    price_neighbor < max_price, max_price
                )
                imports_value = (imports * price_domestic).sum()
                exports_value = (exports * price_neighbor).sum()
                df = df.set_index("Technology")
                df.loc["Import", f"{column} [bln PLN]"] += imports_value / 1e9
                df.loc["Export", f"{column} [bln PLN]"] += -exports_value / 1e9
                df = df.reset_index()

    df[f"{column} [bln PLN]"] = df[f"{column} [bln PLN]"].round(2)
    df = df.rename(
        columns={
            f"{column} [bln PLN]": f"{column.replace('Expenditure', 'costs')} [bln PLN]"
        }
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

        df["Curtailment [%]"] = (
            df["Curtailment [GWh]"] / df["Maximum production [GWh]"] * 100
        ).round(2)
        df = df[["Technology", "Curtailment [%]"]]

    else:
        snapshots = read_variable(output_dir, "snapshots", "timestep")
        df = pd.merge(
            df, snapshots, left_on="snapshot_id", right_index=True, how="left"
        )
        df = df[["Technology", "timestep", "p_curtailed"]]

    return df


def list_fuels(sector):
    fuels = []
    if sector == "electricity":
        fuels = [
            "Natural gas",
            "Hard coal",
            "Lignite",
            "Biomass straw",
            "Biomass wood chips",
            "Biogas",
        ]
    elif sector == "heat":
        fuels = ["Conventional boiler", "Biomass boiler", "Conventional heating plant"]
    elif sector == "hydrogen":
        fuels = ["Natural gas reforming"]
    elif sector == "light vehicles":
        fuels = ["ICE vehicle"]
    return fuels


def get_fuel_consumption(output_dir, sector="electricity", fuel_emissions=False):
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
    df_fuel = df_fuel.replace("Biogas plant", "Biogas")

    df_eff = read_variable(output_dir, "generators", "efficiency")
    df_eff.name = "Efficiency"

    df_eff_fuel = pd.merge(df_eff, df_fuel, left_index=True, right_index=True)

    for df_unit in [df_country, df_eff_fuel]:
        df = pd.merge(df, df_unit, left_on="unit", right_index=True, how="left")

    df = df[df["Country"] == "PL"].drop(columns="Country")
    df = df[df["Fuel"].isin(list_fuels(sector))]

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


def get_co2_emissions(output_dir, sector="electricity"):
    return get_fuel_consumption(output_dir, sector=sector, fuel_emissions=True)


def get_co2_costs(output_dir, sector="electricity"):
    meta = json.load(open(output_dir.joinpath("meta.json")))
    source_prices = meta["prices"]

    year = meta["years"][-1]

    df_prices = read_excel(data_dir("input", f"prices;source={source_prices}.xlsx"))
    co2_price = df_prices.loc[
        df_prices["Price Type"] == "CO2 price [PLN/tCO2]", year
    ].values[0]

    df = get_co2_emissions(output_dir, sector=sector)

    df["CO2 costs [bln PLN]"] = (df["CO2 emissions [Mt]"] * co2_price / 1e3).round(2)
    df = df.drop(columns=["CO2 emissions [Mt]"])

    return df


def get_supply_and_withdrawal_timeseries(
    output_dir,
    sector="electricity",
    groupby="carrier",
):
    sectoral_carriers = [sector if sector != "electricity" else "AC"]
    if sector == "heat":
        sectoral_carriers += ["District heating", "Heat pump small"]

    df_bus_carrier = read_variable(output_dir, "buses", "carrier").to_frame()
    df_bus_carrier = df_bus_carrier[df_bus_carrier.index.str.startswith("PL")]

    # Withdrawal

    df_load_bus = read_variable(output_dir, "loads", "bus").to_frame()
    df_load_bus = df_load_bus.merge(
        df_bus_carrier, left_on="bus", right_index=True, how="inner"
    ).rename(columns={"carrier": "bus_carrier"})
    df_load_bus = df_load_bus[df_load_bus["bus_carrier"].isin(sectoral_carriers)]

    df_load = read_variable(output_dir, "loads", "p_set", time_dependent=True)
    df_load = df_load[df_load.columns.intersection(df_load_bus.index)]

    # Fixed loads
    df_load_p_set = read_variable(output_dir, "loads", "p_set").to_frame()
    df_load_p_set = df_load_p_set[df_load_p_set["p_set"] > 0]
    df_load_p_set = df_load_p_set.merge(
        df_load_bus, left_index=True, right_index=True, how="inner"
    )
    for bus, p_set in df_load_p_set["p_set"].items():
        df_load.loc[:, bus] = p_set

    # Supply

    df_gen_bus = read_variable(output_dir, "generators", "bus").to_frame()
    df_gen_bus = df_gen_bus.merge(
        df_bus_carrier, left_on="bus", right_index=True, how="inner"
    ).rename(columns={"carrier": "bus_carrier"})
    df_gen_bus = df_gen_bus[df_gen_bus["bus_carrier"].isin(sectoral_carriers)]

    df_gen = read_variable(output_dir, "generators", "p", time_dependent=True)
    df_gen = df_gen[df_gen.columns.intersection(df_gen_bus.index)]

    # Mixed - storage

    df_su_bus = read_variable(output_dir, "storage_units", "bus").to_frame()
    df_su_bus = df_su_bus.merge(
        df_bus_carrier, left_on="bus", right_index=True, how="inner"
    ).rename(columns={"carrier": "bus_carrier"})
    df_su_bus = df_su_bus[df_su_bus["bus_carrier"].isin(sectoral_carriers)]

    df_su_withdrawal = read_variable(
        output_dir, "storage_units", "p_store", time_dependent=True
    )
    df_su_withdrawal = df_su_withdrawal[
        df_su_withdrawal.columns.intersection(df_su_bus.index)
    ]
    df_su_supply = read_variable(
        output_dir, "storage_units", "p_dispatch", time_dependent=True
    )
    df_su_supply = df_su_supply[df_su_supply.columns.intersection(df_su_bus.index)]

    df_store_bus = read_variable(output_dir, "stores", "bus").to_frame()
    df_store_bus = df_store_bus.merge(
        df_bus_carrier, left_on="bus", right_index=True, how="inner"
    ).rename(columns={"carrier": "bus_carrier"})
    df_store_bus = df_store_bus[df_store_bus["bus_carrier"].isin(sectoral_carriers)]

    df_store = read_variable(output_dir, "stores", "p", time_dependent=True)
    df_store = df_store[df_store.columns.intersection(df_store_bus.index)]
    df_store_withdrawal = df_store.clip(upper=0)
    df_store_supply = df_store.clip(lower=0)

    # Mixed - links

    df_link_bus0 = read_variable(output_dir, "links", "bus0").to_frame()
    df_link_bus0 = df_link_bus0.merge(
        df_bus_carrier, left_on="bus0", right_index=True, how="inner"
    ).rename(columns={"carrier": "bus0_carrier"})
    df_link_bus0 = df_link_bus0[df_link_bus0["bus0_carrier"].isin(sectoral_carriers)]

    df_link0 = read_variable(output_dir, "links", "p0", time_dependent=True)
    df_link0 = df_link0[df_link0.columns.intersection(df_link_bus0.index)]
    df_link0_withdrawal = df_link0.clip(lower=0)
    df_link0_supply = df_link0.clip(upper=0)

    df_link_bus1 = read_variable(output_dir, "links", "bus1").to_frame()
    df_link_bus1 = df_link_bus1.merge(
        df_bus_carrier, left_on="bus1", right_index=True, how="inner"
    ).rename(columns={"carrier": "bus1_carrier"})
    df_link_bus1 = df_link_bus1[df_link_bus1["bus1_carrier"].isin(sectoral_carriers)]

    df_link1 = read_variable(output_dir, "links", "p1", time_dependent=True)
    df_link1 = df_link1[df_link1.columns.intersection(df_link_bus1.index)]
    df_link1_withdrawal = df_link1.clip(lower=0)
    df_link1_supply = df_link1.clip(upper=0)

    df_supply = pd.concat(
        [
            df_gen,
            df_su_supply,
            df_store_supply,
            -df_link0_supply,
            -df_link1_supply,
        ],
        axis=1,
    )
    df_supply = df_supply.loc[:, (df_supply != 0).any(axis=0)]

    df_withdrawal = pd.concat(
        [
            -df_load,
            -df_su_withdrawal,
            df_store_withdrawal,
            -df_link0_withdrawal,
            -df_link1_withdrawal,
        ],
        axis=1,
    )
    df_withdrawal = df_withdrawal.loc[:, (df_withdrawal != 0).any(axis=0)]

    if groupby is not None:
        for flow in ["dispatch", "store"]:
            dfs_groupby = []
            for components in [
                "loads",
                "generators",
                "links",
                "storage_units",
                "stores",
            ]:
                df_groupby = read_variable(output_dir, components, groupby)
                if components == "loads":
                    df_groupby[:] = sector.replace("_", " ").capitalize() + " final use"
                if components == "links":
                    df_groupby[df_groupby.isin(["AC", "DC"])] = (
                        "Import" if flow == "dispatch" else "Export"
                    )
                if components in ["storage_units", "stores"]:
                    df_groupby[:] += f" {flow}"
                dfs_groupby.append(df_groupby)
            df_groupby = pd.concat(dfs_groupby)

            if flow == "dispatch":
                df_supply = df_supply.T.groupby(df_groupby).sum().T
            if flow == "store":
                df_withdrawal = df_withdrawal.T.groupby(df_groupby).sum().T

        if sector == "heat":
            df_withdrawal["District heating loss"] = (
                df_withdrawal["District heating output"]
                + df_supply["District heating output"]
            )
            df_withdrawal = df_withdrawal.drop(
                columns=["District heating output", "Heat pump small output"]
            )
            df_supply = df_supply.drop(
                columns=["District heating output", "Heat pump small output"]
            ).rename(columns=lambda x: x.replace(" heat output", ""))

    df_snapshots = read_variable(output_dir, "snapshots", "timestep")
    df_supply = pd.merge(
        df_snapshots, df_supply, left_index=True, right_index=True, how="left"
    ).set_index("timestep")
    df_withdrawal = pd.merge(
        df_snapshots, df_withdrawal, left_index=True, right_index=True, how="left"
    ).set_index("timestep")

    return df_supply, df_withdrawal


from icecream import ic


def get_peak_demand(output_dir, sector="electricity", top_hours=40, type="total"):
    df_supply, df_withdrawal = get_supply_and_withdrawal_timeseries(
        output_dir, sector=sector, groupby="carrier"
    )

    vres_columns = [col for col in df_supply.columns if col.startswith(("PV", "Wind"))]
    if type == "total":
        df_load = df_supply.sum(axis=1)
    elif type == "residual":
        df_load = df_supply.drop(columns=vres_columns).sum(axis=1)
    elif type == "vRES":
        df_load = df_supply[vres_columns].sum(axis=1)
    elif type == "exogenous":
        df_load = -df_withdrawal[f"{sector.replace('_', ' ').capitalize()} final use"]

    n_timesteps = top_hours
    top_timesteps = df_load.sort_values(ascending=False).iloc[:n_timesteps].index

    df_supply = df_supply.loc[top_timesteps].mean(axis=0).round(1).to_frame()
    df_withdrawal = df_withdrawal.loc[top_timesteps].mean(axis=0).round(1).to_frame()

    df = pd.concat([df_supply, df_withdrawal], axis=0)
    df = (df / 1e3).round(2)
    df = df[df != 0].dropna()
    df.index.name = "Technology"
    df.columns = [f"Average power in {top_hours} hours with highest {type} load [GW]"]
    return df.reset_index()


def get_from_statistics(
    output_dir, variable="Generation [TWh]", sector="electricity", groupby="carrier"
):
    meta = json.load(open(output_dir.joinpath("meta.json")))
    year = meta["years"][-1]
    yearly_step = meta["extension_years"]

    df = pd.read_csv(output_dir.joinpath("statistics.csv"))

    df["p0_sign"] = df["p0_sign"].fillna(1.0)
    for bus in ["bus", "bus0", "bus1"]:
        df[bus] = df[bus].fillna("")

    # Map special sectoral buses to the actual sectoral carriers
    bus_carriers_per_sector = {
        "heat": ["District heating", "Heat pump small"],
    }
    for sector_iter, carriers in bus_carriers_per_sector.items():
        sectoral_carrier = sector_iter if sector_iter != "electricity" else "AC"
        for carrier in carriers:
            for bus in ["bus", "bus0", "bus1"]:
                df[f"{bus}_carrier"] = df[f"{bus}_carrier"].replace(
                    carrier, sectoral_carrier
                )

    # Set interconnector carriers
    df.loc[
        df["bus0"].str.startswith("PL") & ~df["bus1"].str.startswith("PL"), "carrier"
    ] = "Export"
    df.loc[
        ~df["bus0"].str.startswith("PL") & df["bus1"].str.startswith("PL"), "carrier"
    ] = "Import"

    # Set area for interconnectors to PL
    df.loc[df["carrier"].isin(["Export", "Import"]), "area"] = "PL"

    # Set area for load based on bus
    is_load = df["component"] == "Load"
    df.loc[is_load, "area"] = df.loc[is_load, "bus"].str[:2]

    # Set sector for load based on bus_carrier
    df.loc[is_load, "sector"] = df.loc[is_load, "bus_carrier"].replace(
        "AC", "electricity"
    )

    # Set carrier for load based on sector
    df.loc[is_load, "carrier"] = (
        df.loc[is_load, "sector"].str.capitalize() + " final use"
    )

    # Fill missing area with bus and then PL
    df["area"] = df["area"].fillna(df["bus"]).replace("", "PL")

    # Remove "foreign" from carrier names
    df["carrier"] = df["carrier"].str.replace(" foreign", "")

    # Set Country
    df["Country"] = df["area"].str[:2]

    if sector == "heat":
        is_chp = df["technology"].fillna("").str.contains("CHP")
        df.loc[is_chp, "carrier"] = df.loc[is_chp, "carrier"].str.replace(
            "heat output", "CHP"
        )

    if sector == "electricity":
        # Change carrier to technology for combustibles
        is_ng = df["carrier"].isin(
            [
                "Hard coal",
                "Lignite",
                "Natural gas",
                "Hydrogen",
                "Biogas",
                "Biomass straw",
                "Biomass wood chips",
            ]
        )
        df.loc[is_ng, "carrier"] = df.loc[is_ng, "technology"].str.replace(
            "Biogas engine", "Biogas"
        )
        if variable in []:
            df.loc[is_ng, "carrier"] = df.loc[is_ng, "carrier"].str.replace(" CHP", "")

    # Define sectoral_carrier based on the sector
    sectoral_carrier = sector if sector != "electricity" else "AC"

    index_columns = ["Country", groupby]

    if variable == "Generation [TWh]":
        # Generators
        is_sectoral_generator = (df["component"] == "Generator") & (
            df["bus_carrier"] == sectoral_carrier
        )
        df_gen = df.loc[is_sectoral_generator, index_columns + ["Supply"]].rename(
            columns={"Supply": variable}
        )

        # Stores and storage units
        is_sectoral_store_or_storage_unit = df["component"].isin(
            ["Store", "StorageUnit"]
        ) & (df["bus_carrier"] == sectoral_carrier)
        df_dispatch = df.loc[
            is_sectoral_store_or_storage_unit, index_columns + ["Supply"]
        ].rename(columns={"Supply": variable})
        df_dispatch[groupby] += " dispatch"
        df_store = df.loc[
            is_sectoral_store_or_storage_unit, index_columns + ["Withdrawal"]
        ].rename(columns={"Withdrawal": variable})
        df_store[groupby] += " store"

        # Links
        is_sectoral_supplying_link = (
            (df["component"] == "Link")
            & (
                ((df["p0_sign"] > 0) & (df["bus1_carrier"] == sectoral_carrier))
                | ((df["p0_sign"] < 0) & (df["bus0_carrier"] == sectoral_carrier))
            )
            & ~(df["carrier"] == "Export")
        )
        df_link_supply = df.loc[
            is_sectoral_supplying_link, index_columns + ["Supply"]
        ].rename(columns={"Supply": variable})
        is_sectoral_withdrawing_link = (
            (df["component"] == "Link")
            & (
                ((df["p0_sign"] > 0) & (df["bus0_carrier"] == sectoral_carrier))
                | ((df["p0_sign"] < 0) & (df["bus1_carrier"] == sectoral_carrier))
            )
            & ~(df["carrier"] == "Import")
        )
        df_link_withdrawal = df.loc[
            is_sectoral_withdrawing_link, index_columns + ["Withdrawal"]
        ].rename(columns={"Withdrawal": variable})

        # Loads
        is_sectoral_load = (df["component"] == "Load") & (
            df["bus_carrier"] == sectoral_carrier
        )
        df_load = df.loc[is_sectoral_load, index_columns + ["Withdrawal"]].rename(
            columns={"Withdrawal": variable}
        )

        df = pd.concat(
            [df_gen, df_store, df_dispatch, df_link_supply, df_link_withdrawal, df_load]
        )

        if sector == "heat":
            df[groupby] = df[groupby].replace(
                "District heating output", "District heating loss"
            )

        # Convert mean power in MW to summed production in TWh
        df[variable] *= 8760 / 1e6

        precision = 2
        df = df.groupby(index_columns)[[variable]].sum().round(precision).reset_index()

        if sector == "electricity":
            if "BEV V2G" in df[groupby].unique():
                df = df.set_index(groupby)
                df.loc["BEV charger - recovered", variable] = -df.loc[
                    "BEV V2G", variable
                ]
                df.loc["BEV charger - net", variable] = (
                    df.loc["BEV charger", variable] + df.loc["BEV V2G", variable]
                ).round(precision)
                df = df.drop(index=["BEV charger"])
                df["Country"] = df["Country"].fillna("PL")
                df = df.reset_index()
            else:
                df[groupby] = df[groupby].replace("BEV charger", "BEV charger - net")

    if variable in [
        "Installed capacity [GW]",
        "New installed capacity [GW]",
        "Capacity utilisation [%]",
        "Curtailment [%]",
    ]:
        is_not_store_nor_load = ~df["component"].isin(["Store", "Load"])
        is_sector_supplying = (
            (df["bus_carrier"] == sectoral_carrier)
            | ((df["p0_sign"] > 0) & (df["bus1_carrier"] == sectoral_carrier))
            | ((df["p0_sign"] < 0) & (df["bus0_carrier"] == sectoral_carrier))
        )
        is_not_virtual = ~df["carrier"].str.contains("Virtual")
        is_not_technology_bundle_link = ~df["carrier"].isin(
            ["District heating output", "Heat pump small output"]
        )

        if variable != "Generation [TWh]" and sector == "electricity":
            is_sector_supplying |= df["carrier"].isin(["Biogas plant"])

        sel = (
            is_sector_supplying
            & is_not_store_nor_load
            & is_not_virtual
            & is_not_technology_bundle_link
        )

        if variable in ["Installed capacity [GW]", "New installed capacity [GW]"]:
            column_name = "Optimal Capacity"
            scale_factor = 0.001
            precision = 2

            df = df.loc[
                sel,
                index_columns + [column_name, "build_year", "lifetime"],
            ].rename(columns={column_name: variable})

            if variable == "New installed capacity [GW]":
                df = df[
                    (df["build_year"] > year - yearly_step)
                    & (df["build_year"] <= year)
                    & (df["lifetime"] > 1)
                ]

            df[variable] *= scale_factor
            df = (
                df.groupby(index_columns)[[variable]]
                .sum()
                .round(precision)
                .reset_index()
            )

        if variable == "Capacity utilisation [%]":
            column_name = "Capacity Factor"
            weight_column_name = "Optimal Capacity"
            scale_factor = 100
            precision = 1

            df = df.loc[
                sel,
                index_columns + [column_name, weight_column_name],
            ].rename(columns={column_name: variable})

            df[variable] *= df[weight_column_name] * scale_factor
            df = df.groupby(index_columns)[[variable, weight_column_name]].sum()
            df[variable] /= df[weight_column_name]
            df = df[[variable]].round(precision).reset_index()

        if variable == "Curtailment [%]":
            column_name = "Curtailment"
            generation_column_name = "Supply"
            scale_factor = 100
            precision = 1

            df = df.loc[
                sel,
                index_columns + [column_name, generation_column_name],
            ].rename(columns={column_name: variable})
            df = df[df[variable] > 0]
            df = df.groupby(index_columns)[[variable, generation_column_name]].sum()
            df[variable] /= df[generation_column_name] + df[variable]
            df[variable] *= scale_factor
            df = df[[variable]].round(precision).reset_index()

    if variable in [
        "Installed store capacity [GWh]",
        "New installed store capacity [GWh]",
    ]:
        is_store = df["component"] == "Store"
        is_sectoral = df["bus_carrier"] == sectoral_carrier

        if variable == "New installed store capacity [GWh]" and sector == "electricity":
            is_sectoral |= df["carrier"].isin(["Biogas storage"])

        sel = is_store & is_sectoral

        column_name = "Optimal Capacity"
        scale_factor = 0.001
        precision = 1

        df = df.loc[
            sel,
            index_columns + [column_name, "build_year", "lifetime"],
        ].rename(columns={column_name: variable})

        if variable == "New installed store capacity [GWh]":
            df = df[
                (df["build_year"] > year - yearly_step)
                & (df["build_year"] <= year)
                & (df["lifetime"] > 1)
            ]

        df[variable] *= scale_factor
        df = df.groupby(index_columns)[[variable]].sum().round(precision).reset_index()

    df = df.rename(columns={groupby: "Technology"})

    return df


def get_reserve(output_dir, reserve_name, aggregate=True, aggregate_by="Technology"):
    dfs = []

    for components in ["generators", "storage_units"]:
        if not output_dir.joinpath(f"{components}-r_{reserve_name}.csv").exists():
            continue

        df_bus = read_variable(output_dir, components, "bus")
        df_bus.name = "bus"
        df_bus = df_bus[df_bus.str.startswith("PL")]
        df_bus = df_bus.reset_index()

        df_r = read_variable(
            output_dir,
            components,
            f"r_{reserve_name}",
            time_dependent=True,
        )
        df_r = df_r.reset_index(names="snapshot_id").melt(
            id_vars="snapshot_id", var_name="name", value_name="r"
        )

        dfs.append(df_r)

    if len(dfs) == 0:
        return pd.DataFrame()
    df_r = pd.concat(dfs)

    df_tech = pd.concat(
        [
            read_variable(output_dir, components, "carrier")
            for components in ["generators", "storage_units"]
        ]
    )
    df_tech.name = "Technology"
    df_tech = df_tech.dropna()

    df_r = pd.merge(df_r, df_tech, left_on="name", right_index=True, how="right")

    if aggregate:
        df_r = (
            df_r.groupby(["snapshot_id", aggregate_by])
            .agg({"r": "sum"})
            .reset_index()
            .groupby(aggregate_by)
            .agg({"r": "mean"})
            .reset_index()
        )
        df_r["r"] = df_r["r"].round(2)
        df_r = df_r.rename(
            columns={"r": f"Mean {reserve_name.replace('_', ' ').title()} Reserve [MW]"}
        )
    else:
        snapshots = read_variable(output_dir, "snapshots", "timestep")
        df_r = pd.merge(
            df_r, snapshots, left_on="snapshot_id", right_index=True, how="left"
        )
        df_r = df_r[[aggregate_by, "timestep", "r"]]
    return df_r


def combine_data(get_data_function, project, runs_list=None, fillna=0):
    if runs_list is None:
        output_dir = runs_dir(project, "output")
        logging.info(f"Reading {output_dir}")
        df = get_data_function(output_dir)
    else:
        dfs = []
        internal_indices = []
        for run in runs_list:
            output_dir = runs_dir(f"{project};{dict_to_str(run)}", "output")
            logging.info(f"Reading {output_dir}")
            df = get_data_function(output_dir)
            # It is assumed that the value column is the last one
            internal_indices.append(df.iloc[:, :-1].drop_duplicates())
            for key, value in run.items():
                df[key] = value
            dfs.append(df)
        df = pd.concat(dfs)
        internal_indices = pd.concat(internal_indices).drop_duplicates()
        external_indices = pd.DataFrame.from_dict(runs_list)
        df_index = pd.merge(external_indices, internal_indices, how="cross")
        index_cols = df_index.columns.to_list()
        df_index = df_index.sort_values(by=index_cols)
        df = pd.merge(df_index, df, on=index_cols, how="left").fillna(fillna)
    return df


def get_investment_costs(output_dir, sector, store=True, annuitized=False):
    meta = json.load(open(output_dir.joinpath("meta.json")))
    investment_year = meta["years"][-1]
    discount_rate = meta["discount_rate"]
    technology_year = int(get_technology_year(investment_year))
    source_technology_data = meta["technology_data"]

    capacity_col = (
        "New installed capacity [GW]"
        if not store
        else "New installed store capacity [GWh]"
    )

    df = get_from_statistics(
        output_dir,
        variable=capacity_col,
        sector=sector,
        groupby="technology",
    )

    df_tech = read_excel(
        data_dir("input", f"technology_data;source={source_technology_data}.xlsx"),
        sheet_var="Technology",
    )
    df_tech = df_tech[["Technology", "parameter", str(technology_year)]]
    df_cost = df_tech[
        df_tech["parameter"].str.startswith("Investment cost [MPLN/MW")
    ].copy()
    # TODO: fix electrolyser data
    df_cost = df_cost[
        (df_cost["Technology"] != "Electrolyser")
        | (df_cost["parameter"] == "Investment cost [MPLN/MW_t]")
    ]
    df_cost = df_cost[["Technology", str(technology_year)]].rename(
        columns={str(technology_year): "unit_cost"}
    )

    df = pd.merge(df, df_cost, on="Technology", how="left")
    df["Investment cost [bln PLN]"] = df[capacity_col] * df["unit_cost"]
    df = df[["Technology", "Investment cost [bln PLN]"]]

    col, digits = "Investment cost [bln PLN]", 2
    if annuitized:
        df_lifetime = df_tech[df_tech["parameter"] == "Lifetime [years]"].copy()
        df_lifetime = df_lifetime[["Technology", str(technology_year)]].rename(
            columns={str(technology_year): "lifetime"}
        )
        df = pd.merge(df, df_lifetime, on="Technology", how="left")
        df["Annuitised investment cost [bln PLN]"] = df[
            "Investment cost [bln PLN]"
        ] * calculate_annuity(df["lifetime"], discount_rate)
        col, digits = "Annuitised investment cost [bln PLN]", 3

    df["Technology"] = (
        df["Technology"]
        # .str.replace(" CHP", "")
        .str.replace(" 1h", "")
        .str.replace(" 2h", "")
        .str.replace(" 4h", "")
    )
    df = df[[col, "Technology"]].groupby("Technology").sum().reset_index()
    df[col] = df[col].round(digits)

    return df


def get_fixed_costs(output_dir, sector, store=True):
    meta = json.load(open(output_dir.joinpath("meta.json")))
    investment_year = meta["years"][-1]
    technology_year = int(get_technology_year(investment_year))
    source_technology_data = meta["technology_data"]

    capacity_col = (
        "Installed capacity [GW]" if not store else "Installed store capacity [GWh]"
    )

    df = get_from_statistics(
        output_dir,
        variable=capacity_col,
        sector=sector,
        groupby="technology",
    )
    df = df[df["Country"] == "PL"].drop(columns=["Country"])

    df_tech = read_excel(
        data_dir("input", f"technology_data;source={source_technology_data}.xlsx"),
        sheet_var="Technology",
    )
    df_tech = df_tech[["Technology", "parameter", str(technology_year)]]
    df_tech = df_tech[df_tech["parameter"].str.startswith("Fixed cost [PLN/MW")]
    # TODO: fix electrolyser data
    df_tech = df_tech[
        (df_tech["Technology"] != "Electrolyser")
        | (df_tech["parameter"] == "Fixed cost [PLN/MW_t/year]")
    ]
    df_tech = df_tech[["Technology", str(technology_year)]].rename(
        columns={str(technology_year): "unit_cost"}
    )

    df = pd.merge(df, df_tech, on="Technology", how="left")
    df["Fixed cost [bln PLN]"] = df[capacity_col] * df["unit_cost"] / 1e6
    df = df[["Technology", "Fixed cost [bln PLN]"]]

    df["Technology"] = (
        df["Technology"]
        # .str.replace(" CHP", "")
        .str.replace(" 1h", "")
        .str.replace(" 2h", "")
        .str.replace(" 4h", "")
    )
    df = df.groupby("Technology").sum().reset_index()
    df["Fixed cost [bln PLN]"] = df["Fixed cost [bln PLN]"].round(2)

    return df


def plot_variable(
    project,
    x_var="year",
    x_vals=[2030],
    non_x_params={},
    cat_var="Technology",
    cat_style_dict=technology_colors,
    cat_labels=("Technology", ""),
    name="generation",
    get_run_data=lambda output_dir: get_from_statistics(
        output_dir, variable="Generation [TWh]", sector="electricity"
    ),
    nice_name="Generation",
    unit="TWh",
    sector="electricity",
    country="PL",
    digits=1,
    stack=True,
    y_range=None,
    fig_height=12,
    width_per_x=3,
    legend_width=4,
    extension="png",
    dpi=300,
    force=True,
):
    for dir in ["data", "plots"]:
        os.makedirs(output_dir(project, dir, name), exist_ok=True)

    suffix = f"x={x_var}"
    if non_x_params:
        suffix += f";{dict_to_str(non_x_params)}"

    prefix = f"{sector.replace(' ', '_')}_{name}"
    data_file = output_dir(project, "data", name, f"{prefix};{suffix}.csv")
    plot_file = output_dir(project, "plots", name, f"{prefix};{suffix}.{extension}")
    if country != "PL":
        os.makedirs(output_dir(project, "plots", name, "neighbours"), exist_ok=True)
        plot_file = output_dir(
            project,
            "plots",
            name,
            "neighbours",
            f"{prefix};country={country};{suffix}.{extension}",
        )

    if not data_file.exists() or force:
        runs_list = extend_param_list([non_x_params], **{x_var: x_vals})
        df = combine_data(get_run_data, project, runs_list)
        if country == "PL":
            df.to_csv(data_file, index=False)
    else:
        df = pd.read_csv(data_file)
    if "Country" in df.columns:
        df = df[df["Country"] == country].drop(columns=["Country"])

    y_var = f"{nice_name} [{unit}]"
    df = df[[x_var, cat_var, y_var]]
    df[y_var] = df[y_var].round(digits + 1)

    title = f"{nice_name} – {sector}"
    if country != "PL":
        title += f" – {country}"
    title += f" [{unit}]"

    footer = f"Project: {project} • Parameters: {dict_to_str(non_x_params)}"
    footer += "\n" + f"PyPSA-PL • Instrat • Generated: {make_timestamp()} "

    if stack:
        totals_positive, totals_negative = (
            df[df[y_var] > 0].groupby(x_var)[y_var].sum(),
            df[df[y_var] < 0].groupby(x_var)[y_var].sum(),
        )
        max_val = 0 if totals_positive.empty else max(totals_positive)
        min_val = 0 if totals_negative.empty else min(totals_negative)
        height = max_val - min_val
        small_value = 0.025 * height * 12 / fig_height
    else:
        small_value = 0

    if df[y_var].abs().sum() == 0:
        return

    fig = plot_bars(
        df,
        x_var=x_var,
        y_var=y_var,
        cat_var=cat_var,
        x_label=x_var,
        cat_style_dict=cat_style_dict,
        title=title,
        footer=footer,
        cat_label_positive=cat_labels[0],
        cat_label_negative=cat_labels[1],
        stack=stack,
        y_range=y_range,
        small_value=small_value,
        figsize=(len(x_vals) * width_per_x + legend_width, fig_height),
    )
    fig.savefig(plot_file, dpi=dpi)
    plt.close()


def plot_generation(*args, **kwargs):
    nice_name, unit = "Generation", "TWh"
    plot_variable(
        *args,
        name="generation",
        get_run_data=lambda output_dir: get_from_statistics(
            output_dir,
            variable=f"{nice_name} [{unit}]",
            sector=kwargs.get("sector", "electricity"),
        ),
        nice_name=nice_name,
        unit=unit,
        cat_labels=("Supply", "Withdrawal"),
        fig_height=16,
        **kwargs,
    )


def plot_capacity(*args, **kwargs):
    nice_name, unit = "Installed capacity", "GW"
    plot_variable(
        *args,
        name="capacity",
        get_run_data=lambda output_dir: get_from_statistics(
            output_dir,
            variable=f"{nice_name} [{unit}]",
            sector=kwargs.get("sector", "electricity"),
        ),
        nice_name=nice_name,
        unit=unit,
        **kwargs,
    )


def plot_store_capacity(*args, **kwargs):
    sector = kwargs.get("sector", "electricity")
    if sector not in ["heat", "hydrogen"]:
        return
    nice_name, unit = "Installed store capacity", "GWh"
    plot_variable(
        *args,
        name="store_capacity",
        get_run_data=lambda output_dir: get_from_statistics(
            output_dir,
            variable=f"{nice_name} [{unit}]",
            sector=sector,
        ),
        nice_name=nice_name,
        unit=unit,
        **kwargs,
    )


def plot_utilisation(*args, **kwargs):
    nice_name, unit = "Capacity utilisation", "%"
    plot_variable(
        *args,
        name="capacity_utilisation",
        get_run_data=lambda output_dir: get_from_statistics(
            output_dir,
            variable=f"{nice_name} [{unit}]",
            sector=kwargs.get("sector", "electricity"),
        ),
        nice_name=nice_name,
        unit=unit,
        stack=False,
        y_range=(0, 105),
        width_per_x=10,
        **kwargs,
    )


def plot_curtailment(*args, **kwargs):
    sector = kwargs.get("sector", "electricity")
    if sector != "electricity":
        return
    nice_name, unit = "Curtailment", "%"
    plot_variable(
        *args,
        name="curtailment",
        get_run_data=lambda output_dir: get_from_statistics(
            output_dir,
            variable=f"{nice_name} [{unit}]",
            sector=kwargs.get("sector", "electricity"),
        ),
        nice_name=nice_name,
        unit=unit,
        stack=False,
        width_per_x=5,
        **kwargs,
    )


def plot_fuel_consumption(*args, **kwargs):
    nice_name, unit = "Fuel consumption", "TWh_t"
    plot_variable(
        *args,
        name="fuel_consumption",
        get_run_data=lambda output_dir: get_fuel_consumption(
            output_dir, sector=kwargs.get("sector", "electricity")
        ),
        nice_name=nice_name,
        unit=unit,
        cat_var="Fuel",
        cat_labels=("Fuel", ""),
        stack=False,
        width_per_x=5,
        **kwargs,
    )


def plot_emissions(*args, **kwargs):
    nice_name, unit = "CO2 emissions", "Mt"
    plot_variable(
        *args,
        name="emissions",
        get_run_data=lambda output_dir: get_co2_emissions(
            output_dir, sector=kwargs.get("sector", "electricity")
        ),
        nice_name=nice_name,
        unit=unit,
        cat_var="Fuel",
        cat_labels=("Fuel", ""),
        **kwargs,
    )


def plot_opex(*args, **kwargs):
    nice_name, unit = "Operational costs", "bln PLN"
    plot_variable(
        *args,
        name="opex",
        get_run_data=lambda output_dir: get_costs(
            output_dir,
            cost_component="opex",
            sector=kwargs.get("sector", "electricity"),
        ),
        nice_name=nice_name,
        unit=unit,
        cat_labels=("Costs", "Revenues"),
        **kwargs,
    )


def plot_co2_costs(*args, **kwargs):
    nice_name, unit = "CO2 costs", "bln PLN"
    plot_variable(
        *args,
        name="co2_costs",
        get_run_data=lambda output_dir: get_co2_costs(
            output_dir,
            sector=kwargs.get("sector", "electricity"),
        ),
        nice_name=nice_name,
        cat_var="Fuel",
        cat_labels=("Fuel", ""),
        unit=unit,
        **kwargs,
    )


def plot_new_capacity(*args, **kwargs):
    sector = kwargs.get("sector", "electricity")
    nice_name, unit = "New installed capacity", "GW"
    if sector == "light vehicles":
        return
    plot_variable(
        *args,
        name="new_capacity",
        get_run_data=lambda output_dir: get_from_statistics(
            output_dir,
            variable=f"{nice_name} [{unit}]",
            sector=sector,
        ),
        nice_name=nice_name,
        unit=unit,
        **kwargs,
    )


def plot_new_store_capacity(*args, **kwargs):
    sector = kwargs.get("sector", "electricity")
    nice_name, unit = "New installed store capacity", "GWh"
    if sector not in ["electricity", "heat", "hydrogen"]:
        return
    plot_variable(
        *args,
        name="new_store_capacity",
        get_run_data=lambda output_dir: get_from_statistics(
            output_dir,
            variable=f"{nice_name} [{unit}]",
            sector=sector,
        ),
        nice_name=nice_name,
        unit=unit,
        **kwargs,
    )


def plot_investment_costs(*args, **kwargs):
    sector = kwargs.get("sector", "electricity")
    nice_name, unit = "Investment cost", "bln PLN"
    if sector == "light vehicles":
        return
    plot_variable(
        *args,
        name="investment_costs",
        get_run_data=lambda output_dir: get_investment_costs(
            output_dir,
            sector=sector,
            store=False,
        ),
        nice_name=nice_name,
        unit=unit,
        **kwargs,
    )


def plot_annuitised_investment_costs(*args, **kwargs):
    sector = kwargs.get("sector", "electricity")
    nice_name, unit = "Annuitised investment cost", "bln PLN"
    if sector == "light vehicles":
        return
    plot_variable(
        *args,
        name="annuitised_investment_costs",
        get_run_data=lambda output_dir: get_investment_costs(
            output_dir,
            sector=sector,
            store=False,
            annuitized=True,
        ),
        nice_name=nice_name,
        unit=unit,
        **kwargs,
    )


def plot_store_investment_costs(*args, **kwargs):
    sector = kwargs.get("sector", "electricity")
    nice_name, unit = "Investment cost", "bln PLN"
    if sector not in ["electricity", "heat", "hydrogen"]:
        return
    plot_variable(
        *args,
        name="store_investment_costs",
        get_run_data=lambda output_dir: get_investment_costs(
            output_dir,
            sector=sector,
            store=True,
        ),
        nice_name=nice_name,
        unit=unit,
        **kwargs,
    )


def plot_annuitised_store_investment_costs(*args, **kwargs):
    sector = kwargs.get("sector", "electricity")
    nice_name, unit = "Annuitised investment cost", "bln PLN"
    if sector not in ["electricity", "heat", "hydrogen"]:
        return
    plot_variable(
        *args,
        name="annuitised_store_investment_costs",
        get_run_data=lambda output_dir: get_investment_costs(
            output_dir,
            sector=sector,
            store=True,
            annuitized=True,
        ),
        nice_name=nice_name,
        unit=unit,
        **kwargs,
    )


def plot_fixed_costs(*args, **kwargs):
    sector = kwargs.get("sector", "electricity")
    nice_name, unit = "Fixed cost", "bln PLN"
    plot_variable(
        *args,
        name="fixed_costs",
        get_run_data=lambda output_dir: get_fixed_costs(
            output_dir,
            sector=sector,
            store=False,
        ),
        nice_name=nice_name,
        unit=unit,
        **kwargs,
    )


def plot_store_fixed_costs(*args, **kwargs):
    sector = kwargs.get("sector", "electricity")
    nice_name, unit = "Fixed cost", "bln PLN"
    if sector not in []:
        return
    plot_variable(
        *args,
        name="store_fixed_costs",
        get_run_data=lambda output_dir: get_fixed_costs(
            output_dir,
            sector=sector,
            store=True,
        ),
        nice_name=nice_name,
        unit=unit,
        **kwargs,
    )


def plot_peak_total_load(*args, **kwargs):
    sector = kwargs.get("sector", "electricity")
    top_hours = 40
    nice_name, unit = (
        f"Average power in {top_hours} hours with highest total load",
        "GW",
    )
    if sector not in ["electricity", "heat"]:
        return
    plot_variable(
        *args,
        name="peak_total_load",
        get_run_data=lambda output_dir: get_peak_demand(
            output_dir,
            type="total",
            sector=sector,
            top_hours=top_hours,
        ),
        nice_name=nice_name,
        unit=unit,
        cat_labels=("Supply", "Withdrawal"),
        fig_height=16,
        **kwargs,
    )


def plot_peak_residual_load(*args, **kwargs):
    sector = kwargs.get("sector", "electricity")
    top_hours = 40
    nice_name, unit = (
        f"Average power in {top_hours} hours with highest residual load",
        "GW",
    )
    if sector != "electricity":
        return
    plot_variable(
        *args,
        name="peak_residual_load",
        get_run_data=lambda output_dir: get_peak_demand(
            output_dir,
            type="residual",
            top_hours=top_hours,
            sector=sector,
        ),
        nice_name=nice_name,
        unit=unit,
        cat_labels=("Supply", "Withdrawal"),
        fig_height=16,
        **kwargs,
    )


def plot_peak_exogenous_load(*args, **kwargs):
    sector = kwargs.get("sector", "electricity")
    top_hours = 40
    nice_name, unit = (
        f"Average power in {top_hours} hours with highest exogenous load",
        "GW",
    )
    if sector not in ["electricity", "heat"]:
        return
    plot_variable(
        *args,
        name="peak_exogenous_load",
        get_run_data=lambda output_dir: get_peak_demand(
            output_dir,
            type="exogenous",
            sector=sector,
            top_hours=top_hours,
        ),
        nice_name=nice_name,
        unit=unit,
        cat_labels=("Supply", "Withdrawal"),
        fig_height=16,
        **kwargs,
    )


def plot_peak_vres(*args, **kwargs):
    sector = kwargs.get("sector", "electricity")
    top_hours = 40
    nice_name, unit = (
        f"Average power in {top_hours} hours with highest vRES load",
        "GW",
    )
    if sector != "electricity":
        return
    plot_variable(
        *args,
        name="peak_vres",
        get_run_data=lambda output_dir: get_peak_demand(
            output_dir,
            type="vRES",
            top_hours=top_hours,
            sector=sector,
        ),
        nice_name=nice_name,
        unit=unit,
        cat_labels=("Supply", "Withdrawal"),
        fig_height=16,
        **kwargs,
    )
