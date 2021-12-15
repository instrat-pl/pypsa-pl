import pandas as pd


def create_date_index(year):
    # Remove 29 February from leap years
    year = int(year)
    if year % 4 == 0:
        whole_year = pd.date_range(start=f'{year}-01-01', end=f'{year + 1}-01-01', closed='left', freq='h')
        #whole_year = pd.date_range(start=f'{year}-01-01', end=f'{year}-02-01', closed='left', freq='h')
        feb_29 = pd.date_range(start=f'{year}-02-29', end=f'{year}-03-01', closed='left', freq='h')
        snapshots = [i for i in whole_year if i not in feb_29]
    else:
        snapshots = pd.date_range(start=f'{year}-01-01', end=f'{year + 1}-01-01', closed='left', freq='h')
        #snapshots = pd.date_range(start=f'{year}-01-01', end=f'{year}-02-01', closed='left', freq='h')
    return snapshots
