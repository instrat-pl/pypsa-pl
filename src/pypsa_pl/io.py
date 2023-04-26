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
        return df
    else:
        # Read all sheets
        dfs = pd.read_excel(file, sheet_name=None, **kwargs)

        # Concatenate DataFrames from each sheet after removing columns that start with "#" and adding sheet_var column
        df_full = pd.concat(
            [
                df.drop(
                    columns=[col for col in df.columns if str(col).startswith("#")]
                ).assign(**{sheet_var: sheet_name})
                for sheet_name, df in dfs.items()
                if not sheet_name.startswith("#")
            ]
        )

        # Sheet names are always strings. Convert the sheet_var column to numeric if possible.
        df_full.loc[:, sheet_var] = pd.to_numeric(df_full[sheet_var], errors="ignore")
        return df_full


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    return [dict(zip(keys, instance)) for instance in itertools.product(*vals)]


def dict_to_str(d):
    return ";".join(f"{key}={value}" for key, value in d.items())
