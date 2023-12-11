import itertools
import pandas as pd


import pandas as pd


def read_excel(file, sheet_var=None, **kwargs):
    """
    Read data from an Excel file and return a pandas DataFrame. Optionally concatenate data from
    multiple sheets and store sheet names in a specified column.

    :param file: The file path or file object of the Excel file to read.
    :param sheet_var: A column name to store sheet names in the final DataFrame. If None, reads the first sheet (default: None).
    :param **kwargs: Additional keyword arguments to pass to the `pd.read_excel` function.
    :return: A pandas DataFrame containing the data from the Excel file.
    """
    # If sheet_var is None, read data from the first sheet
    if sheet_var is None:
        df = pd.read_excel(file, **kwargs)
        # Remove columns that start with "#"
        df = df.drop(columns=[col for col in df.columns if str(col).startswith("#")])
        # Remove rows that start with "#"
        df = df.loc[~df.iloc[:, 0].astype(str).str.startswith("#")]
        return df
    else:
        # Read all sheets
        dfs = pd.read_excel(file, sheet_name=None, **kwargs)

        # Concatenate DataFrames from each sheet after removing columns and rows that start with "#" and adding sheet_var column
        df_full = pd.concat(
            [
                df.drop(columns=[col for col in df.columns if str(col).startswith("#")])
                .assign(**{sheet_var: sheet_name})
                .rename(columns=lambda col: str(col))
                for sheet_name, df in dfs.items()
                if not sheet_name.startswith("#")
            ]
        )
        df_full = df_full.loc[~df_full.iloc[:, 0].astype(str).str.startswith("#")]

        # Sheet names are always strings. Convert the sheet_var column to numeric if possible.
        df_full.loc[:, sheet_var] = pd.to_numeric(df_full[sheet_var], errors="ignore")
        return df_full


def order_dict(d):
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[0])}


def product_dicts(sort=False, **kwargs):
    if sort:
        kwargs = order_dict(kwargs)
    keys = kwargs.keys()
    vals = kwargs.values()
    return [dict(zip(keys, instance)) for instance in itertools.product(*vals)]


def ordered_product_dicts(**kwargs):
    return product_dicts(sort=True, **kwargs)


def extend_param_list(core_param_list, **kwargs):
    if len(kwargs) == 0:
        return core_param_list
    else:
        return [
            {**core_params, param: value}
            for core_params in core_param_list
            for param, values in kwargs.items()
            for value in values
        ]


def make_param_list(
    core_param_space, extended_param_space={}, exclude_params=[], sort=True
):
    core_param_list = product_dicts(
        **{
            param: value
            for param, value in core_param_space.items()
            if param not in exclude_params
        }
    )
    full_param_list = extend_param_list(
        core_param_list,
        **{
            param: value
            for param, value in extended_param_space.items()
            if param not in exclude_params
        },
    )
    if sort:
        full_param_list = [order_dict(p) for p in full_param_list]
    return full_param_list


def dict_to_str(d, sort=False):
    if sort:
        d = order_dict(d)
    return ";".join(f"{key}={value}" for key, value in d.items())


def dict_to_ordered_str(d):
    return dict_to_str(d, sort=True)
