import pandas as pd


def define_time_dependent_attributes(df_cap, params):

    index = ["carrier", "technology", "qualifier", "attribute"]
    df = pd.DataFrame(columns=index + ["profile_type"]).set_index(index)

    df_cap["qualifier"] = df_cap["qualifier"].fillna("none")

    # (1a) Electricity final use profiles - p_set
    electricity_final_use_df = df_cap.loc[
        df_cap["carrier"] == "electricity final use", index[:-1]
    ].drop_duplicates()
    for vals in electricity_final_use_df.itertuples(index=False):
        df.loc[(*vals, "p_set"), :] = ["electricity final use load profile"]

    # # (1b) Fuel final use profiles - p_set
    # if "p_set_annual" in df_cap.columns:
    #     fuel_final_use_df = df_cap.loc[
    #         df_cap["p_set_annual"].notna()
    #         & ~df_cap["carrier"].isin(
    #             [
    #                 "electricity final use",
    #                 "hydrogen final use",
    #                 "light vehicle mobility final use",
    #                 "space heating final use",
    #                 "water heating final use",
    #                 "other heating final use",
    #             ]
    #         ),
    #         index[:-1],
    #     ]
    #     for vals in fuel_final_use_df.itertuples(index=False):
    #         df.loc[(*vals, "p_set"), :] = ["constant load profile"]

    # (2) CHP generation following heat final use load profile - p_set
    if params["fix_public_chp"] and "p_set_annual" in df_cap.columns:
        chp_df = df_cap.loc[
            df_cap["p_set_annual"].notna() & df_cap["carrier"].str.contains("CHP"),
            index[:-1],
        ].drop_duplicates()
        for vals in chp_df.itertuples(index=False):
            df.loc[(*vals, "p_set"), :] = ["public heat final use load profile"]

    # (3) vRES availability profiles - p_max_pu
    vres_df = df_cap.loc[
        df_cap["carrier"].str.contains(("wind|solar")), index[:-1]
    ].drop_duplicates()
    for vals in vres_df.itertuples(index=False):
        df.loc[(*vals, "p_max_pu"), :] = ["vres availability profile"]

    # (4a) Fixed generation or load profiles - p_set
    # This includes constant generation profiles for industrial CHP units
    if "p_set" in df_cap.columns:
        const_generation_df = df_cap.loc[
            df_cap["p_set"].notna(),
            index[:-1],
        ].drop_duplicates()
        if not params["fix_industrial_chp"]:
            const_generation_df = const_generation_df[
                ~const_generation_df["carrier"].str.contains("CHP")
            ]
        for vals in const_generation_df.itertuples(index=False):
            df.loc[(*vals, "p_set"), :] = ["constant load profile"]

    # (4b) Fixed generation or load profiles - p_set_pu
    if "p_set_pu" in df_cap.columns:
        const_generation_pu_df = df_cap.loc[
            df_cap["p_set_pu"].notna(),
            index[:-1],
        ].drop_duplicates()
        for vals in const_generation_pu_df.itertuples(index=False):
            for attr in ["p_min_pu", "p_max_pu"]:
                df.loc[(*vals, attr), :] = ["constant load pu profile"]

    # (5) Space heating and light vehicle mobility final use load profiles - p_set_pu
    if "p_set_pu_annual" in df_cap.columns:
        space_heating_df = df_cap.loc[
            df_cap["p_set_pu_annual"].notna()
            & df_cap["carrier"].isin(
                [
                    "centralised space heating",
                    "decentralised space heating",
                    "building retrofits",
                ]
            ),
            index[:-1],
        ].drop_duplicates()
        for vals in space_heating_df.itertuples(index=False):
            for attr in ["p_min_pu", "p_max_pu"]:
                df.loc[(*vals, attr), :] = ["space heating final use load pu profile"]

        light_vehicle_mobility_df = df_cap.loc[
            df_cap["p_set_pu_annual"].notna()
            & (df_cap["carrier"] == "light vehicle mobility"),
            index[:-1],
        ].drop_duplicates()
        for vals in light_vehicle_mobility_df.itertuples(index=False):
            for attr in ["p_min_pu", "p_max_pu"]:
                df.loc[(*vals, attr), :] = [
                    "light vehicle mobility final use load pu profile"
                ]

    # (6) COP profiles / max output profiles for heat pumps - efficiency / p_max_pu
    heat_pump_df = df_cap.loc[
        df_cap["carrier"].str.startswith("heat pump"), index[:-1]
    ].drop_duplicates()
    for vals in heat_pump_df.itertuples(index=False):
        df.loc[(*vals, "efficiency"), :] = ["heat pump COP profile"]
        df.loc[(*vals, "p_max_pu"), :] = ["heat pump max output pu profile"]

    # (7a) Availability of BEV charger / V2G
    bev_charger_df = df_cap.loc[
        df_cap["carrier"].isin(["BEV charger", "BEV V2G"]), index[:-1]
    ].drop_duplicates()
    for vals in bev_charger_df.itertuples(index=False):
        df.loc[(*vals, "p_max_pu"), :] = ["BEV charger max output pu profile"]
        df.loc[(*vals, "p_min_pu"), :] = ["BEV charger min output pu profile"]

    # (7b) Max state of charge for BEV battery
    bev_battery_df = df_cap.loc[
        df_cap["carrier"] == "BEV battery", index[:-1]
    ].drop_duplicates()
    for vals in bev_battery_df.itertuples(index=False):
        df.loc[(*vals, "e_max_pu"), :] = ["BEV battery max SOC profile"]
        df.loc[(*vals, "e_min_pu"), :] = ["BEV battery min SOC profile"]
    df = df.reset_index()
    return df
