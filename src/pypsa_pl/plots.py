from cycler import cycler
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt


from pypsa_pl.config import data_dir


default_size = 10
dark_color = "#231f20"
light_color = "#6c7178"
bg_color = "#dcddde"
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


mpl_style = {
    #
    "axes.axisbelow": True,
    #
    "axes.edgecolor": dark_color,
    #
    "axes.grid": False,
    "axes.grid.axis": "y",
    #
    "axes.labelcolor": light_color,
    "axes.labelsize": default_size,
    "axes.labelweight": "regular",
    #
    "axes.linewidth": 0.5,
    #
    "axes.prop_cycle": cycler("color", palette),
    #
    "axes.titlesize": default_size + 2,
    "axes.titleweight": "medium",
    #
    "axes.spines.top": False,
    "axes.spines.right": False,
    #
    "figure.titlesize": default_size + 2,
    "figure.dpi": 90,
    #
    "font.family": "sans-serif",
    "font.sans-serif": ["Work Sans", "DejaVu Sans", "Arial", "sans-serif"],
    "font.size": default_size,
    "font.weight": "regular",
    #
    "grid.color": bg_color,
    "grid.linewidth": 0.5,
    #
    "legend.fontsize": default_size - 2,
    "legend.title_fontsize": default_size,
    "legend.frameon": False,
    #
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    #
    "patch.edgecolor": dark_color,
    "patch.linewidth": 0.25,
    #
    "xtick.bottom": False,
    "xtick.color": light_color,
    "xtick.labelsize": default_size - 2,
    "xtick.major.width": 0.5,
    "xtick.major.size": 6,
    "xtick.minor.width": 0.5,
    "ytick.minor.size": 4,
    #
    "ytick.left": False,
    "ytick.color": light_color,
    "ytick.labelsize": default_size,
    "ytick.major.width": 0.5,
    "ytick.major.size": 6,
    "ytick.minor.width": 0.5,
    "ytick.minor.size": 4,
    #
}

mpl.rcParams.update(mpl_style)


def calculate_luminance(rgb):
    # https://www.w3.org/TR/AERT/#color-contrast
    return rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114


def get_order_and_colors(network=None, run_name=None, agg="carrier"):

    if network is not None:
        df = network.carriers
    elif run_name is not None:
        df = pd.read_csv(
            data_dir("runs", run_name, "input_network", "carriers.csv"), index_col=0
        )

    if agg == "carrier":
        order = df.sort_values("order").index
        colors = df["color"].to_dict()
    elif agg == "aggregation":
        df = (
            df[["aggregation", "order", "color"]]
            .sort_values("order")
            .groupby("aggregation")
            .first()
        )
        order = df.sort_values("order").index
        colors = df["color"].to_dict()
    return order, colors


def format_label(digits, commas, threshold):
    return lambda val: (
        ""
        if abs(val) < threshold  # or np.isnan(val)
        else f"{val:.{digits}f}".replace(".", "," if commas else ".")
    )


def add_bar_labels(
    ax,
    sign,
    n_cat,
    label_digits=1,
    label_commas=False,
    label_threshold=10,
    show_total=True,
    horizontal=False,
):
    if label_digits < 0:
        label_digits = 0
    if sign == "+":
        bar_containers = ax.containers
    if sign == "-":
        bar_containers = ax.containers[n_cat["+"] :]

    containers_per_bar = {}
    for container in bar_containers:
        patches = container.patches
        values = container.datavalues
        bar_locs = [
            (patch.get_x() if not horizontal else patch.get_y()) for patch in patches
        ]
        for patch, value, bar_loc in zip(patches, values, bar_locs):
            if value == 0:
                continue
            if bar_loc not in containers_per_bar.keys():
                containers_per_bar[bar_loc] = {"patches": [], "values": []}
            containers_per_bar[bar_loc]["patches"].append(patch)
            containers_per_bar[bar_loc]["values"].append(value)
    containers_per_bar = {
        bar_loc: mpl.container.BarContainer(
            container["patches"],
            datavalues=container["values"],
            orientation="vertical" if not horizontal else "horizontal",
        )
        for bar_loc, container in containers_per_bar.items()
    }

    # Do not display patches and labels of bars with "" label
    axis_labels = [x.get_text() for x in ax.get_xticklabels()]
    bar_locs = sorted(containers_per_bar.keys())

    for axis_label, bar_loc in zip(axis_labels, bar_locs):
        if axis_label in ["", " ", "\n"]:
            for patch in containers_per_bar[bar_loc].patches:
                patch.set_visible(False)
            continue
        container = containers_per_bar[bar_loc]
        # Bar labels
        colors = [patch.get_facecolor() for patch in container.patches]
        labels = ax.bar_label(
            container,
            fmt=format_label(label_digits, label_commas, label_threshold),
            label_type="center",
            fontsize=default_size - 2,
        )
        for label, color in zip(labels, colors):
            label.set(
                color="white" if calculate_luminance(color) < 0.5 else "black",
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
                labels=[format_label(label_digits, label_commas, 0)(total)],
                fontsize=default_size,
                fontweight="medium",
                padding=3,
            )
            for label in labels:
                label.set(color=dark_color)


def add_legend(ax, sign, n_cat, max_characters=30):
    handles, labels = ax.get_legend_handles_labels()

    if sign == "+":
        handles = handles[::-1]
        labels = labels[::-1]
        pos = (n_cat["-"] + 0.5 * n_cat["+"]) / (n_cat["-"] + n_cat["+"])
    if sign == "-":
        handles = handles[n_cat["+"] :]
        labels = labels[n_cat["+"] :]
        pos = 0.5 * n_cat["-"] / (n_cat["-"] + n_cat["+"])

    # Add dummy label to ensure thet each legend has approx. the same width
    # One whitespace character has a width of around 1.5 of an average text character
    dummy_label = " " * int(1.5 * max_characters + 0.5)
    labels.append(dummy_label)
    dummy_handle = mpl.patches.Patch(facecolor="none", edgecolor="none")
    handles.append(dummy_handle)

    legend = ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1, pos),
        title=sign if n_cat["+"] > 0 and n_cat["-"] > 0 else None,
        title_fontproperties={"weight": "medium"},
        alignment="left",
        ncol=1,
    )
    if sign == "+" and n_cat["-"] > 0:
        artist = ax.add_artist(legend)
        artist.set_in_layout(True)


def plot_bar(
    df,
    x_var="year",
    cat_var="carrier",
    y_var="value",
    title="Electricity production [TWh]",
    cat_order=None,
    cat_colors=None,
    label_digits=1,
    label_threshold=0,
    label_commas=False,
    show_total_label=True,
    bar_width=0.8,
    figsize=(5, 8),
):

    fig, ax = plt.subplots(figsize=figsize)

    df = df.pivot(index=x_var, columns=cat_var, values=y_var)

    cats = {"+": df.columns[(df > 0).any()], "-": df.columns[(df < 0).any()]}

    n_cat = {sign: len(cats[sign]) for sign in ["+", "-"]}

    for sign in ["+", "-"]:

        subdf = df[cats[sign]].copy()
        if sign == "+":
            subdf[subdf < 0] = 0
        elif sign == "-":
            subdf[subdf > 0] = 0

        if subdf.empty:
            continue

        order = (
            cat_order.intersection(subdf.columns)
            if cat_order is not None
            else subdf.columns
        )
        subdf = subdf[order]
        subdf.plot.bar(
            stacked=True,
            ax=ax,
            title=title,
            color=cat_colors if cat_colors is not None else {},
            linewidth=mpl_style["patch.linewidth"],
            edgecolor=mpl_style["patch.edgecolor"],
            width=bar_width,
            rot=0,
        )

        add_bar_labels(
            ax,
            sign=sign,
            n_cat=n_cat,
            label_digits=label_digits,
            label_threshold=label_threshold,
            show_total=show_total_label,
            label_commas=label_commas,
        )

        add_legend(ax, sign, n_cat)

    ax.grid(axis="y")
    ax.spines["left"].set_visible(False)

    return fig


def plot_area(
    df,
    is_wide=False,
    x_var="hour",
    cat_var="carrier",
    y_var="value",
    title="Electricity generation [GW]",
    cat_order=None,
    cat_colors=None,
    ylim=None,
    figsize=(8, 8),
):

    fig, ax = plt.subplots(figsize=figsize)

    if not is_wide:
        df = df.pivot(index=x_var, columns=cat_var, values=y_var)

    cats = {"+": df.columns[(df > 0).any()], "-": df.columns[(df < 0).any()]}

    n_cat = {sign: len(cats[sign]) for sign in ["+", "-"]}

    for sign in ["+", "-"]:

        subdf = df[cats[sign]].copy()
        if sign == "+":
            subdf[subdf < 0] = 0
        elif sign == "-":
            subdf[subdf > 0] = 0

        order = (
            cat_order.intersection(subdf.columns)
            if cat_order is not None
            else subdf.columns
        )
        subdf = subdf[order]
        subdf.plot.area(
            ax=ax,
            title=title,
            color=cat_colors if cat_colors is not None else {},
            ylim=ylim,
            linewidth=0,
        )

        add_legend(ax, sign, n_cat)

    ax.grid(axis="y")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", which="major", bottom=True)

    return fig


def plot_line(
    df, title="Marginal cost of electricity production [MWh]", figsize=(8, 8)
):
    fig, ax = plt.subplots(figsize=figsize)

    df.plot.line(
        ax=ax,
        title=title,
        color="#535ce3",
    )
    ax.grid(axis="y")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", which="major", bottom=True)

    return fig
