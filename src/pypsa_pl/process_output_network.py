import logging
import numpy as np
import pandas as pd

# from pypsa_pl_mini.config import data_dir
from pypsa_pl.custom_statistics import make_capex_calculator, make_opex_calculator
from pypsa_pl.build_network import concat_inputs


def get_attr(attr):
    def getter(n, c):
        df = n.df(c)
        if attr in df:
            values = df[attr].fillna("")
        else:
            values = pd.Series("", index=df.index)
        return values.rename(attr)

    return getter


def get_bus_attr(bus, attr):
    def getter(n, c):
        df = n.df(c)
        if bus in df:
            values = df[bus].map(n.buses[attr]).fillna("")
        else:
            values = pd.Series("", index=df.index)
        return values.rename(f"{bus}_{attr}")

    return getter


def is_in_bus_carriers(df, bus_carriers, column_name="bus_carrier"):
    if isinstance(bus_carriers, list):
        return df[column_name].isin(bus_carriers)
    else:
        return df[column_name] == bus_carriers


def check_qualifiers(df, bus_qualifiers, column_name="qualifier"):
    if isinstance(bus_qualifiers, list):
        return df[column_name].isin(bus_qualifiers)
    if isinstance(bus_qualifiers, str):
        return df[column_name] == bus_qualifiers
    else:
        return df[column_name] == df[column_name]


def make_custom_groupby(extra_attrs=[], extra_bus_attrs=[], buses=[]):
    attrs = ["area", "aggregation", "carrier", "technology", "qualifier"] + extra_attrs
    bus_attrs = ["carrier", "qualifier"] + extra_bus_attrs

    def custom_groupby(n, c, **kwargs):
        return [get_attr(attr)(n, c) for attr in attrs] + [
            get_bus_attr(bus, bus_attr)(n, c) for bus in buses for bus_attr in bus_attrs
        ]

    return custom_groupby


def attribute_bus_carrier_to_component_atribute(
    network, attr="carrier", output_bus=True
):
    technology_carrier_definitions = network.meta["technology_carrier_definitions"]
    df = concat_inputs("technology_carrier_definitions", technology_carrier_definitions)
    bus_carrier_column = "output_carrier" if output_bus else "input_carrier"
    df = df[[attr, bus_carrier_column, "bus_carrier"]]
    df[bus_carrier_column] = df[bus_carrier_column].fillna(df["bus_carrier"])
    df = df.drop(columns="bus_carrier").drop_duplicates()
    if len(df) > df[attr].nunique():
        logging.warning(
            f"Attribution of bus carrier to `{attr}` attribute is ambiguous."
        )
    return df.set_index(attr)[bus_carrier_column]


def calculate_statistics(network, bus_carriers=None):
    index = ["component", "area", "aggregation", "carrier", "technology", "qualifier"]
    df = (
        network.statistics(
            groupby=make_custom_groupby(),
            bus_carrier=bus_carriers,
        )
        .stack(future_stack=True)
        .reset_index(names=index + ["year"])
    )

    # TODO: statistics seem to be broken in the current version of pypsa - verify impacts
    # Temporary fix: swap supply with withdrawal and change sign if both are either NaN or negative
    to_swap = (df["Supply"].isna() | (df["Supply"] < 0)) & (
        df["Withdrawal"].isna() | (df["Withdrawal"] < 0)
    )
    df.loc[to_swap, ["Supply", "Withdrawal"]] = -df.loc[
        to_swap, ["Withdrawal", "Supply"]
    ].values

    # Replace all NaNs with 0
    value_columns = [
        "Optimal Capacity",
        "Installed Capacity",
        "Dispatch",
        "Withdrawal",
        "Curtailment",
        "Supply",
        "Capital Expenditure",
        "Operational Expenditure",
        "Revenue",
    ]
    df[value_columns] = df[value_columns].fillna(0)

    # Keep curtailment only for wind and solar
    is_vres = df["carrier"].str.startswith(("wind", "solar PV"))
    df.loc[~is_vres, "Curtailment"] = np.nan

    # Fix for stores
    is_store = df["component"] == "Store"
    df.loc[
        is_store,
        ["Dispatch", "Withdrawal", "Supply", "Operational Expenditure", "Revenue"],
    ] *= network.snapshot_weightings["generators"].values[0]

    df = df.set_index(["year"] + index).reset_index()
    return df


def calculate_fuel_consumption(
    network,
    show_intermediate_consumption=False,
    unit="PJ",
):
    assert unit in ["PJ", "TWh"]
    # Define here intermediate fuel supply carriers
    fuel_supply_carriers = {
        "biogas production": "biogas supply",
        "biogas upgrading": "biomethane supply",
    }
    fuels = network.carriers[network.carriers.index.str.endswith("supply")].index
    fuels = fuels.union(fuel_supply_carriers.values()).str[: -len(" supply")]
    df = pd.concat(
        [
            calculate_flows(network, bus_carriers=fuel).assign(bus_carrier=fuel)
            for fuel in fuels
        ],
        ignore_index=True,
    )
    index = ["year", "area", "carrier", "aggregation", "bus_carrier"]
    df = df.groupby(index).agg({"value": "sum"}).reset_index()

    # Supply - upstream fuels
    df_supply = df.loc[df["value"] > 0].copy()
    df_supply["fuel"] = (
        df_supply["carrier"].replace(fuel_supply_carriers).str[: -len(" supply")]
    )
    df_supply = df_supply.drop(columns="carrier")

    # Withdrawal - downstream carriers
    df_withdrawal = df.loc[df["value"] < 0].copy()
    df_withdrawal["value"] *= -1
    df_withdrawal["fuel"] = df_withdrawal["bus_carrier"]
    bus_carrier_map = attribute_bus_carrier_to_component_atribute(
        network, attr="carrier"
    )
    df_withdrawal["downstream_bus_carrier"] = (
        df_withdrawal["carrier"].map(bus_carrier_map).fillna(df["bus_carrier"])
    )

    # Calculate shares in supply
    dfs = []
    for _, subdf in df_supply.groupby(["year", "area", "bus_carrier"]):
        subdf["value"] = (subdf["value"] / subdf["value"].sum()).fillna(1)
        dfs.append(subdf)
    df_supply_shares = pd.concat(dfs).rename(columns={"value": "supply_share"})

    # Propagate supply shares to withdrawal
    df_withdrawal = df_withdrawal.drop(columns="fuel").merge(
        df_supply_shares[["year", "area", "bus_carrier", "fuel", "supply_share"]],
        how="left",
        on=["year", "area", "bus_carrier"],
    )
    df_withdrawal["value"] *= df_withdrawal["supply_share"]
    df_withdrawal = df_withdrawal.drop(columns="supply_share")

    df = df_withdrawal

    if not show_intermediate_consumption:
        # By default, it should not be shown as it can lead to double counting
        df = df[~df["carrier"].isin(fuel_supply_carriers.keys())]

    index = ["area", "year", "fuel", "carrier", "aggregation", "downstream_bus_carrier"]
    df = df.groupby(index).agg({"value": "sum"}).reset_index()
    df = df.rename(columns={"downstream_bus_carrier": "bus_carrier"})
    if unit == "PJ":
        df["value"] *= 3.6

    return df


def calculate_fuel_opex(
    network, cost_attr="marginal_cost", bus_carriers=None, area="PL"
):
    df = calculate_fuel_consumption(
        network, show_intermediate_consumption=True, unit="TWh"
    )
    df_cost = network.generators.copy()
    df_cost = df_cost.loc[
        df_cost["carrier"].str.endswith("supply") & (df_cost["area"] == area),
        ["build_year", "carrier", cost_attr],
    ].rename(columns={"build_year": "year", "carrier": "fuel_carrier"})
    df["fuel_carrier"] = df["fuel"] + " supply"
    df = df.merge(
        df_cost,
        how="left",
        on=["year", "fuel_carrier"],
    )
    # Unit cost is in PLN/MWh, provide result in PLN
    df["value"] *= df[cost_attr] * 1e6
    df = df[df["value"].abs() > 0]
    df = df.drop(columns=["fuel_carrier"])
    if bus_carriers is not None:
        df = df[df["bus_carrier"].isin(bus_carriers)]
    return df


def calculate_co2_emissions(network):
    df_co2 = network.carriers[["co2_emissions"]].dropna()
    df = calculate_fuel_consumption(network, unit="TWh")
    df["fuel_carrier"] = df["fuel"] + " supply"
    # *** Add emissions not related to "supply" technologies
    # Define here extra sources of CO2 emissions with input and downstream carriers
    extra_carriers = {
        "biogas upgrading": ("biogas", "biogas"),
        "biomass agriculture CHP CC": ("biomass agriculture", "electricity in"),
    }
    dfs_extra = [
        calculate_flows(network, bus_carriers=bus_carrier)
        .assign(
            fuel=carrier,
            fuel_carrier=carrier,
            bus_carrier=downstream_bus_carrier,
            value=lambda x: -x["value"],
        )
        .query("carrier == @carrier")
        for carrier, (bus_carrier, downstream_bus_carrier) in extra_carriers.items()
    ]
    df = pd.concat([df] + dfs_extra)[df.columns]
    # ***
    df = df.merge(
        df_co2["co2_emissions"],
        how="inner",
        left_on="fuel_carrier",
        right_index=True,
    )
    df["value"] *= df["co2_emissions"]
    df = df.drop(columns=["co2_emissions", "fuel_carrier"])
    df = df[df["value"].abs() > 0]
    return df


def calculate_opex(network, cost_attr="marginal_cost", bus_carriers=None):
    index = ["component", "area", "aggregation", "carrier", "technology", "qualifier"]
    df = (
        make_opex_calculator(attr=cost_attr)(
            network, groupby=make_custom_groupby(), bus_carrier=bus_carriers
        )
        .rename_axis("year", axis=1)
        .stack(future_stack=True)
        .rename("value")
        .reset_index()
    )

    is_store = df["component"] == "Store"
    df.loc[is_store, "value"] *= network.snapshot_weightings["generators"].values[0]

    df = df.set_index(["year"] + index).reset_index()
    return df


def calculate_capex(network, cost_attr="capital_cost", bus_carriers=None):
    index = ["component", "area", "carrier", "technology", "qualifier"]
    df = (
        make_capex_calculator(attr=cost_attr)(
            network, groupby=make_custom_groupby(), bus_carrier=bus_carriers
        )
        .rename_axis("year", axis=1)
        .stack(future_stack=True)
        .rename("value")
        .reset_index()
    )
    df = df.set_index(["year"] + index).reset_index()
    return df


def calculate_electricity_trade_revenue(network, domestic_area="PL"):
    buses = network.buses[
        network.buses["carrier"].str.startswith(("electricity in", "electricity out"))
    ]
    domestic_bus_carriers = {
        bus_carrier: buses.loc[
            buses["carrier"].str.contains(bus_carrier)
            & (buses["area"].str.startswith(domestic_area)),
            "carrier",
        ].values.tolist()
        for bus_carrier in ["electricity in", "electricity out"]
    }
    neighbour_bus_carriers = {
        bus_carrier: buses.loc[
            buses["carrier"].str.contains(bus_carrier)
            & ~buses["area"].str.startswith(domestic_area),
            "carrier",
        ].values.tolist()
        for bus_carrier in ["electricity in", "electricity out"]
    }
    # Use domestic electricity prices to calculate import costs
    # Use neighbour electricity prices to calculate export revenues
    dfs = []
    for carrier, bus_carriers, sign in [
        ("electricity import", domestic_bus_carriers["electricity in"], 1),
        ("electricity export", neighbour_bus_carriers["electricity in"], -1),
    ]:
        df = calculate_statistics(network, bus_carriers=bus_carriers)
        df = df[df["carrier"] == carrier]
        df["Revenue"] *= sign
        df["area"] = domestic_area
        dfs.append(df)

    df = pd.concat(dfs)
    df = (
        df.groupby(["year", "area", "aggregation", "carrier"])
        .agg(value=("Revenue", "sum"), flow=("Dispatch", "sum"))
        .reset_index()
    )
    df["marginal_cost"] = (df["value"] / df["flow"]).round(2)
    df = df.drop(columns="flow")
    df = df[df["value"].abs() > 0]
    return df


def define_sectors(network):
    sectoral_bus_carriers = {
        "electricity": [
            "electricity in",
            "electricity out",
            "hydro PSH electricity",
            "battery large electricity",
        ],
        "heat centralised": [
            "heat centralised in",
            "heat centralised out",
            "heat storage large tank heat",
        ],
        "heat decentralised": ["heat decentralised", "heat storage small heat"],
        "light vehicle mobility": [
            "light vehicle mobility",
            "BEV electricity",
            "ICE vehicle fuel",
        ],
        "biogas": ["biogas substrate", "biogas"],
        "heating": ["space heating", "water heating", "other heating"],
        # "biomethane": ["biomethane"], # TODO: create separate bus for biomethane
        "hydrogen": ["hydrogen"],
        "natural gas": ["natural gas"],
        "hard coal": ["hard coal"],
        "lignite": ["lignite"],
        "other fuel": ["other fuel"],
        "biomass wood": ["biomass wood"],
        "biomass agriculture": ["biomass agriculture"],
        "process emissions": ["process emissions"],
        "lulucf": ["lulucf"],
    }
    df = network.buses[["area", "carrier"]].drop_duplicates()
    df = df[df["area"].str.startswith("PL")].drop(columns="area")
    df["sector"] = (
        df["carrier"]
        .map(
            {
                bus_carrier: sector
                for sector, bus_carriers in sectoral_bus_carriers.items()
                for bus_carrier in bus_carriers
            }
        )
        .fillna(df["carrier"])
    )
    df = (
        df.sort_values(["sector", "carrier"])
        .reset_index(drop=True)
        .rename(columns={"carrier": "bus_carrier"})
    )
    return df


def calculate_generation_shares_in_chp(network):
    # TODO: implement more precise cost allocation, based on individual CHP units

    sector_bus_carriers = {
        "electricity": ["electricity in"],
        "heat centralised": ["heat centralised in"],
    }

    dfs = []

    for sector, bus_carriers in sector_bus_carriers.items():
        df = calculate_flows(network, bus_carriers=bus_carriers)
        df["sector"] = sector
        dfs.append(df)

    df = pd.concat(dfs)

    df = df[df["carrier"].str.contains("CHP")]

    index = [
        "year",
        "area",
        "carrier",
        "sector",
    ]
    df = df.groupby(index).agg({"value": "sum"}).reset_index()

    df = df.pivot(index=index[:3], columns="sector", values="value").reset_index()
    df["total"] = df[sector_bus_carriers.keys()].sum(axis=1)
    for sector in sector_bus_carriers.keys():
        df[sector] /= df["total"]
    df = df.drop(columns="total")
    df = df.melt(id_vars=index[:3], var_name="sector", value_name="share").dropna(
        subset=["share"]
    )
    df = df.sort_values(index).reset_index(drop=True)
    return df


def calculate_sectoral_costs(network):
    df_sector = define_sectors(network)

    dfs = []

    # (1) Capital and operational costs of infrastructure, fuel supply and CO2 costs
    costs = [
        ("variable_cost", "Var. O&M", calculate_opex),
        ("co2_cost", "CO₂", calculate_opex),
        ("fixed_cost", "Fix. O&M", calculate_capex),
        ("annual_investment_cost", "Ann. invest.", calculate_capex),
    ]
    for sector, subdf_sector in df_sector.groupby("sector"):
        bus_carriers = subdf_sector["bus_carrier"].tolist()
        for cost_attr, label, calculate_cost in costs:
            df = calculate_cost(network, cost_attr=cost_attr, bus_carriers=bus_carriers)
            df["cost component"] = label
            df["sector"] = sector
            dfs.append(df)

    # (2) Electricity trade revenues
    df = calculate_electricity_trade_revenue(network)
    df["sector"] = "electricity"
    df["cost component"] = "Trade"
    dfs.append(df)

    df = pd.concat(dfs)

    # (3) Split costs of CHP between electricity and heat centralised sectors
    df_chp = calculate_generation_shares_in_chp(network).rename(
        columns={"sector": "chp_sector"}
    )
    df = df.merge(df_chp, how="left", on=["year", "area", "carrier"])
    is_chp = df["carrier"].str.contains("CHP")
    df.loc[is_chp, "sector"] = df.loc[is_chp, "chp_sector"]
    df.loc[is_chp, "value"] *= df.loc[is_chp, "share"]
    df = df.drop(columns=["chp_sector", "share"])

    index = ["area", "year", "sector", "carrier", "aggregation", "cost component"]
    df = df.groupby(index).agg({"value": "sum"}).reset_index()

    df["value"] = (df["value"] / 1e9).round(3)
    df = df[df["value"].abs() > 0].reset_index(drop=True)
    return df


def calculate_sectoral_flows(network):
    df_sector = define_sectors(network)

    dfs = []

    for sector, subdf_sector in df_sector.groupby("sector"):
        bus_carriers = subdf_sector["bus_carrier"].tolist()
        df = calculate_flows(network, bus_carriers=bus_carriers)
        df["sector_from"] = sector
        dfs.append(df)

    df = pd.concat(dfs)

    # Keep only negative flows, which include withdrawals by other sectors
    df = df[df["value"] < 0]
    df["value"] *= -1

    # Match carriers to sectors
    bus_carrier_map = attribute_bus_carrier_to_component_atribute(
        network, attr="carrier"
    )
    sector_map = (
        bus_carrier_map.reset_index()
        .rename(columns={"output_carrier": "bus_carrier"})
        .merge(df_sector, how="outer", on="bus_carrier")
    )
    sector_map.loc[sector_map["carrier"].str.endswith("final use"), "sector"] = (
        "final use"
    )
    sector_map = sector_map.set_index("carrier")["sector"]
    df["sector_to"] = df["carrier"].map(sector_map)

    # Keep only intersectoral flows
    df = df[df["sector_from"] != df["sector_to"]]

    # Electricity final use can include transmission losses
    electricity_transmission_loss = network.meta.get(
        "electricity_transmission_loss", 0.03
    )
    df.loc[df["carrier"] == "electricity final use", "value"] *= (
        1 - electricity_transmission_loss
    )

    # Split fuel flows to CHP between electricity and heat centralised sectors
    df_chp = calculate_generation_shares_in_chp(network).rename(
        columns={"sector": "chp_sector"}
    )
    df = df.merge(df_chp, how="left", on=["year", "area", "carrier"])
    is_chp = df["carrier"].str.contains("CHP")
    df.loc[is_chp, "sector_to"] = df.loc[is_chp, "chp_sector"]
    df.loc[is_chp, "value"] *= df.loc[is_chp, "share"]
    df = df.drop(columns=["chp_sector", "share"])

    index = [
        "year",
        "area",
        "sector_from",
        "sector_to",
        "carrier",
        "aggregation",
    ]
    df = df.groupby(index).agg({"value": "sum"}).reset_index()
    df["value"] = df["value"].round(3)
    df = df[df["value"].abs() > 0].reset_index(drop=True)

    return df


def calculate_sectoral_unit_costs(network):
    df_costs = calculate_sectoral_costs(network)
    df_flows = calculate_sectoral_flows(network)

    assert df_costs["area"].str.startswith("PL").all()
    assert df_flows["area"].str.startswith("PL").all()
    assert df_costs["year"].nunique() == 1
    assert df_flows["year"].nunique() == 1

    # (1) Calculate unit costs in a self-consistent manner

    # Unit costs should also reflect costs of upstream sectors
    # Total costs of infrastructure and operations of sector i: C_i
    # Energy flow from sector j to sector i: F_ij
    # Total demand of sector i: D_i = sum_j F_ji
    # Unit cost per sector i: c_i = C_i / D_i + sum_j (1 / D_i) * F_ij * c_j
    # c = D^-1 * C + D^-1 * F * c
    # Solve for c: (1 - D^-1 * F) * c = D^-1 * C

    sectors_i = df_flows["sector_from"].unique()
    C = df_costs.groupby("sector").agg({"value": "sum"}).reindex(sectors_i).fillna(0)
    C = C["value"].values
    D = (
        df_flows.groupby("sector_from")
        .agg({"value": "sum"})
        .reindex(sectors_i)
        .fillna(0)
    )
    inv_D = np.diag((1 / D["value"]).values)
    F = (
        df_flows.groupby(["sector_from", "sector_to"])
        .agg({"value": "sum"})
        .reset_index()
        .pivot(index="sector_to", columns="sector_from", values="value")
        .reindex(index=sectors_i, columns=sectors_i)
        .fillna(0)
    )
    F = F.values

    A = np.identity(len(sectors_i)) - inv_D @ F
    B = inv_D @ C

    c = np.linalg.solve(A, B)

    df_unit_costs = pd.DataFrame(c, index=sectors_i, columns=["unit_cost"]).reset_index(
        names="input_sector"
    )

    # *** Print unit costs
    # df_temp = df_unit_costs.copy()
    # df_temp["unit_cost"] = (df_temp["unit_cost"] * 1e3).round(2)
    # print(df_temp)
    # ***

    # (2) Decompose unit costs including costs of inputs from other sectors
    df_input_costs = df_flows[df_flows["sector_to"].isin(sectors_i)].copy()
    df_input_costs = df_input_costs.rename(
        columns={"sector_to": "sector", "sector_from": "input_sector"}
    )
    df_input_costs = df_input_costs.merge(df_unit_costs, how="left", on="input_sector")
    df_input_costs["value"] *= df_input_costs["unit_cost"]
    df_input_costs["cost component"] = "Fuel"

    df = pd.concat([df_costs, df_input_costs])
    df["input_sector"] = df["input_sector"].fillna("none")

    index = [
        "area",
        "year",
        "sector",
        "carrier",
        "aggregation",
        "cost component",
        "input_sector",
    ]
    df = df.groupby(index).agg({"value": "sum"}).reset_index()

    # (3) Divide by total demand per sector
    df_demand = (
        df_flows.groupby("sector_from")
        .agg(demand=("value", "sum"))
        .reset_index(names="sector")
    )
    df = df.merge(df_demand, how="left", on="sector")
    df["value"] /= df["demand"]
    df = df.drop(columns="demand")

    # (4) Change unit from GPLN/TWh to PLN/MWh
    df["value"] = (df["value"] * 1e3).round(2)

    return df


def calculate_output_capacities(
    network, bus_carriers=["electricity in"], bus_qualifiers=None, type="final"
):
    reverse_links = network.meta.get("reverse_links", False)
    inf = network.meta.get("inf", None)

    if type == "final":
        calculate_capacity = network.statistics.optimal_capacity
    elif type == "initial":
        calculate_capacity = network.statistics.installed_capacity

    # Generators
    df_gen = (
        calculate_capacity(
            comps=["Generator"],
            bus_carrier=bus_carriers,
            groupby=make_custom_groupby(
                extra_attrs=["sign"], extra_bus_attrs=["qualifier"]
            ),
        )
        .rename_axis("year", axis=1)
        .stack(future_stack=True)
        .rename("value")
        .reset_index()
    )

    df_gen = df_gen[df_gen["sign"] > 0].drop(columns="sign")
    # Links
    output_buses = ["bus1", "bus2"] if not reverse_links else ["bus0", "bus2"]
    df_link = (
        calculate_capacity(
            comps=["Link"],
            bus_carrier=bus_carriers,
            groupby=make_custom_groupby(buses=output_buses),
            at_port=True,
        )
        .rename_axis("year", axis=1)
        .stack(future_stack=True)
        .rename("value")
        .reset_index()
    )

    df_link = df_link[
        is_in_bus_carriers(
            df_link, bus_carriers, column_name=f"{output_buses[0]}_carrier"
        )
        | is_in_bus_carriers(
            df_link, bus_carriers, column_name=f"{output_buses[1]}_carrier"
        )
    ].drop(columns=[f"{bus}_carrier" for bus in output_buses])

    if bus_qualifiers is not None:
        df_gen = check_qualifiers(df_gen, bus_qualifiers)
        df_link = df_link[
            check_qualifiers(
                df_link, bus_qualifiers, column_name=f"{output_buses[0]}_qualifier"
            )
            | check_qualifiers(
                df_link, bus_qualifiers, column_name=f"{output_buses[1]}_qualifier"
            )
        ].drop(columns=[f"{bus}_qualifier" for bus in output_buses])

    df = pd.concat([df_gen, df_link])
    if inf:
        df = df[df["value"] != inf]
    return df.reset_index(drop=True)


def calculate_output_capacity_additions(
    network, bus_carriers=["electricity in"], bus_qualifiers=None
):
    df_init = calculate_output_capacities(
        network, bus_carriers, bus_qualifiers=bus_qualifiers, type="initial"
    )
    df_final = calculate_output_capacities(
        network, bus_carriers, bus_qualifiers=bus_qualifiers, type="final"
    )
    index = [col for col in df_init.columns if col != "value"]
    df = pd.merge(
        df_init, df_final, on=index, suffixes=("_init", "_final"), how="outer"
    ).fillna(0)
    df["value"] = df["value_final"] - df["value_init"]
    return df.drop(columns=["value_init", "value_final"])


def calculate_input_capacities(
    network, bus_carriers=["electricity in"], bus_qualifiers=None, type="final"
):
    reverse_links = network.meta.get("reverse_links", False)
    inf = network.meta.get("inf", None)

    if type == "final":
        calculate_capacity = network.statistics.optimal_capacity
    elif type == "initial":
        calculate_capacity = network.statistics.installed_capacity

    # Generators
    df_gen = (
        calculate_capacity(
            comps=["Generator"],
            bus_carrier=bus_carriers,
            groupby=make_custom_groupby(extra_attrs=["sign"]),
        )
        .rename_axis("year", axis=1)
        .stack(future_stack=True)
        .rename("value")
        .reset_index()
    )
    df_gen = df_gen[df_gen["sign"] < 0].drop(columns="sign")
    # Links
    input_buses = ["bus0"] if not reverse_links else ["bus1"]
    df_link = (
        calculate_capacity(
            comps=["Link"],
            bus_carrier=bus_carriers,
            groupby=make_custom_groupby(buses=input_buses),
            at_port=True,
        )
        .rename_axis("year", axis=1)
        .stack(future_stack=True)
        .rename("value")
        .reset_index()
    )
    df_link = df_link[
        is_in_bus_carriers(
            df_link, bus_carriers, column_name=f"{input_buses[0]}_carrier"
        )
    ].drop(columns=[f"{bus}_carrier" for bus in input_buses])

    if bus_qualifiers is not None:
        df_gen = check_qualifiers(df_gen, bus_qualifiers)
        df_link = df_link[
            check_qualifiers(
                df_link, bus_qualifiers, column_name=f"{input_buses[0]}_qualifier"
            )
        ].drop(columns=[f"{bus}_qualifier" for bus in input_buses])

    df = pd.concat([df_gen, df_link])
    if inf:
        df = df[df["value"] != inf]
    return df.reset_index(drop=True)


def calculate_input_capacity_additions(
    network, bus_carriers=["electricity in"], bus_qualifiers=None
):
    df_init = calculate_input_capacities(
        network, bus_carriers, bus_qualifiers=bus_qualifiers, type="initial"
    )
    df_final = calculate_input_capacities(
        network, bus_carriers, bus_qualifiers=bus_qualifiers, type="final"
    )
    index = [col for col in df_init.columns if col != "value"]
    df = pd.merge(df_init, df_final, on=index, suffixes=("_init", "_final"))
    df["value"] = df["value_final"] - df["value_init"]
    return df.drop(columns=["value_init", "value_final"])


def calculate_storage_capacities(
    network, bus_carriers=None, bus_qualifiers=None, type="final"
):
    if type == "final":
        calculate_capacity = network.statistics.optimal_capacity
    elif type == "initial":
        calculate_capacity = network.statistics.installed_capacity

    df = (
        calculate_capacity(groupby=make_custom_groupby(buses=["bus"]), storage=True)
        .rename_axis("year", axis=1)
        .stack(future_stack=True)
        .rename("value")
        .reset_index()
    )
    if bus_carriers is not None:
        df = df[is_in_bus_carriers(df, bus_carriers)]
    if bus_qualifiers is not None:
        df = df[check_qualifiers(df, bus_qualifiers)]
    return df.reset_index(drop=True)


def calculate_storage_capacity_additions(
    network, bus_carriers=None, bus_qualifiers=None
):
    df_init = calculate_storage_capacities(
        network,
        bus_carriers=bus_carriers,
        bus_qualifiers=bus_qualifiers,
        type="initial",
    )
    df_final = calculate_storage_capacities(
        network, bus_carriers=bus_carriers, bus_qualifiers=bus_qualifiers, type="final"
    )
    index = [col for col in df_init.columns if col != "value"]
    df = pd.merge(df_init, df_final, on=index, suffixes=("_init", "_final"))
    df["value"] = df["value_final"] - df["value_init"]
    return df.drop(columns=["value_init", "value_final"])


def calculate_curtailed_vres_energy(network, area="PL"):
    df = (
        network.statistics.curtailment(groupby=make_custom_groupby())
        .rename_axis("year", axis=1)
        .stack(future_stack=True)
        .rename("value")
        .reset_index()
        .drop(columns="component")
    )
    df = df[df["area"].str.startswith(area)]
    df = df[df["value"] > 0]
    df = df.set_index("year").reset_index()
    return df


def calculate_flows(
    network, bus_carriers="electricity", bus_qualifiers=None, annual=True
):

    df = (
        network.statistics.energy_balance(aggregate_bus=False, aggregate_time=False)
        .stack(level=0, future_stack=True)
        .reset_index()
        .rename(columns={"period": "year"})
    )

    df = df[is_in_bus_carriers(df, bus_carriers)].drop(columns="bus_carrier")

    if bus_qualifiers is not None:
        df = df.merge(
            network.buses[["qualifier"]], how="left", left_on="bus", right_index=True
        )
        df = df[check_qualifiers(df, bus_qualifiers)].drop(columns=["qualifier"])

    df = df.merge(
        network.buses["area"], how="left", left_on="bus", right_index=True
    ).drop(columns="bus")

    df = df.merge(
        network.carriers[["aggregation"]],
        left_on="carrier",
        right_index=True,
        how="left",
    )

    df = df.set_index(
        ["year", "component", "area", "carrier", "aggregation"]
    ).sort_index()

    # Provide annual value in TWh
    if annual:
        df *= network.snapshot_weightings["generators"].droplevel(0, axis=0)
        df = (df.sum(axis=1) / 1e6).rename("value").sort_index().reset_index()

    return df


def calculate_energy_balance_at_peak_load(
    network,
    bus_carriers=["electricity in", "electricity out"],
    bus_qualifiers=None,
    cat_var="carrier",
    year_share=40 / 8760,
    load_type="total",
    return_snapshots=False,
):
    df = calculate_flows(network, bus_carriers, bus_qualifiers, annual=False)
    df = df.groupby(level=[cat_var]).sum().transpose()

    vres_columns = [
        col for col in df.columns if col.startswith(("solar", "wind", "hydro ROR"))
    ]
    final_use_columns = [col for col in df.columns if col.endswith("final use")]
    heating_columns = [
        col
        for col in df.columns
        if col.endswith(("space heating", "water heating", "other heating"))
    ]

    if load_type == "total":
        df_load = df[df > 0].sum(axis=1)
    elif load_type == "residual":
        df_load = df.drop(columns=vres_columns)[df > 0].sum(axis=1)
    elif load_type == "vRES":
        df_load = df[vres_columns].sum(axis=1)
    elif load_type == "final use":
        df_load = -df[final_use_columns].sum(axis=1)
    elif load_type == "heating":
        df_load = -df[heating_columns].sum(axis=1)

    n_snapshots = int(year_share * len(network.snapshots) + 0.5)
    if n_snapshots == 0:
        n_snapshots = 1

    peak_snapshots = df_load.sort_values(ascending=False).iloc[:n_snapshots].index
    df = df.loc[peak_snapshots].mean(axis=0).rename("value")

    # Convert to GW
    df = (df / 1e3).round(2)
    df = df[df.abs() > 0].reset_index()

    year = network.meta.get("year", np.nan)
    df["year"] = year

    df = df[["year", cat_var, "value"]]
    if not return_snapshots:
        return df
    else:
        return df, peak_snapshots


def calculate_marginal_prices(
    network, bus_carriers=None, bus_qualifiers=None, area=["PL"]
):

    df = network.buses_t["marginal_price"].T
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["year", "timestep"])
    df = df.rename_axis("bus", axis=0).stack(level=0, future_stack=True).reset_index()

    df = df.merge(
        network.buses[["area", "carrier"]], left_on="bus", right_index=True, how="left"
    ).drop(columns="bus")

    df = df.rename(columns={"carrier": "bus_carrier"})
    df = df[is_in_bus_carriers(df, bus_carriers) & df["area"].isin(area)]
    if bus_qualifiers is not None:
        df = df[check_qualifiers(df, bus_qualifiers)]
    df = df.set_index(["year", "area", "bus_carrier"])
    return df
